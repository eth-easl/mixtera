import multiprocessing as mp
import os
import tempfile
from pathlib import Path
from time import sleep
from unittest import TestCase

from integrationtests.utils import REPRODUCIBILITY_ITERATIONS, TestMetadataParser, setup_test_dataset
from loguru import logger
from mixtera.core.client.mixtera_client import MixteraClient, QueryExecutionArgs, ResultStreamingArgs
from mixtera.core.datacollection.datasets import JSONLDataset
from mixtera.core.query import Query, ResultChunk
from mixtera.core.query.mixture import ArbitraryMixture, MixtureKey, StaticMixture

TEST_LOCAL_INSTANCE_COUNT = 50
TEST_LOCAL_FILE_COUNT = 1
TEST_LOCAL_FRACTION_MULTIPLIER = 2


def parsing_func(sample):
    import json

    return json.loads(sample)["text"]


def run_test_arbitrarymixture(client: MixteraClient):
    job_id = f"local_checkpoint_test_0_workers"
    query = Query.for_job(job_id).select(None)
    query_execution_args = QueryExecutionArgs(
        mixture=ArbitraryMixture(chunk_size=5),
        dp_groups=1,
        nodes_per_group=1,
        num_workers=0,
    )
    client.execute_query(query, query_execution_args)
    client.wait_for_execution(job_id)
    result_streaming_args = ResultStreamingArgs(job_id=job_id)
    logger.info("Executed query.")
    # Get one chunk for each worker
    worker_iterator = client._stream_result_chunks(job_id, 0, 0, 0)
    chunks = [next(worker_iterator)]

    worker_status = [2]  # Worker has consumed 2 samples on chunk
    logger.info(f"Got first chunk, initiating checkpoint.")

    checkpoint_id = client.checkpoint(
        job_id=job_id,
        dp_group_id=0,
        node_id=0,
        worker_status=worker_status,
    )

    while not client.checkpoint_completed(job_id, checkpoint_id, True):
        sleep(0.1)

    logger.info("Checkpoint done.")

    # Now after checkpoint consume all chunks for later comparison
    try:
        while True:
            chunks.append(next(worker_iterator))
    except StopIteration:
        pass

    logger.info(f"Got all chunks.")

    client.restore_checkpoint(job_id, checkpoint_id)
    client.wait_for_execution(job_id)

    logger.info("Restored checkpoint.")

    # Obtain post checkpoint chunks
    chunks_pc: list[ResultChunk] = []
    worker_iterator_pc = client._stream_result_chunks(job_id, 0, 0, 0)
    try:
        while True:
            chunks_pc.append(next(worker_iterator_pc))
    except StopIteration:
        pass

    logger.info(f"Obtained all chunks again.")

    assert len(chunks) == len(
        chunks_pc
    ), f"len(chunks) = {len(chunks)} len(chunks_pc) = {len(chunks_pc)}\n{chunks}\n{chunks_pc}"

    # Validate for chunk 0 that worker status has been transferred correctly
    assert (
        chunks_pc[0]._samples_to_skip == 2
    ), f"Worker 0 chunk 0: skipping { chunks_pc[0]._samples_to_skip} instead of 2 samples."
    for chunk_id in range(1, len(chunks_pc)):
        assert (
            chunks_pc[chunk_id]._samples_to_skip == 0
        ), f"Worker 0 chunk {chunk_id}: skipping { chunks_pc[chunk_id]._samples_to_skip} instead of 0 samples."

    logger.info("Validated worker status.")

    # Validate for all chunks that the content is equivalent
    for j in range(len(chunks_pc)):
        pre_chunk = chunks[j]
        post_chunk = chunks_pc[j]

        TestCase().assertDictEqual(pre_chunk._result_index, post_chunk._result_index, f"worker 0 chunk {j} not equal")
        if pre_chunk._mixture is None:
            assert post_chunk._mixture is None
        else:
            TestCase().assertDictEqual(pre_chunk._mixture, post_chunk._mixture, f"worker 0 chunk {j} mixture not equal")


def run_test_staticmixture(client: MixteraClient):
    job_id = f"local_checkpoint_test_static_mixture"
    query = Query.for_job(job_id).select(None)

    # Initial mixture: only javascript code
    mixture1 = StaticMixture(5, {MixtureKey({"language": ["JavaScript"]}): 1.0})

    query_execution_args = QueryExecutionArgs(
        mixture=mixture1,
        dp_groups=1,
        nodes_per_group=1,
        num_workers=0,
    )

    # Execute the query with the initial mixture (javascript only)
    client.execute_query(query, query_execution_args)
    logger.info("Executed query with initial mixture (javascript only).")

    # Get an iterator over result chunks
    worker_iterator = client._stream_result_chunks(job_id, 0, 0, 0)

    # Get two chunks with the initial mixture
    chunks = []
    for _ in range(2):
        try:
            chunks.append(next(worker_iterator))
        except StopIteration:
            logger.error("Unexpected end of data while fetching initial chunks.")
            return

    # Update the mixture to use only html code
    mixture2 = StaticMixture(5, {MixtureKey({"language": ["HTML"]}): 1.0})

    # There currently is no official callback interface to update the mixture just yet.
    client._get_query_result(job_id).update_mixture(mixture2)
    logger.info("Updated mixture to html only.")

    # Get the third chunk with the updated mixture
    try:
        chunks.append(next(worker_iterator))
    except StopIteration:
        logger.error("Unexpected end of data after updating mixture.")
        return

    worker_status = [2]
    logger.info("Initiating checkpoint.")

    # Perform checkpoint
    checkpoint_id = client.checkpoint(
        job_id=job_id,
        dp_group_id=0,
        node_id=0,
        worker_status=worker_status,
    )

    # Wait for checkpoint to complete
    while not client.checkpoint_completed(job_id, checkpoint_id, True):
        sleep(0.1)

    logger.info("Checkpoint completed.")

    # After checkpoint, consume all remaining chunks for later comparison
    try:
        while True:
            chunks.append(next(worker_iterator))
    except StopIteration:
        pass

    logger.info("Fetched all chunks after checkpoint.")

    # Restore from checkpoint
    client.restore_checkpoint(job_id, checkpoint_id)
    logger.info("Restored from checkpoint.")

    # Obtain chunks after restoring from checkpoint
    chunks_pc = []
    worker_iterator_pc = client._stream_result_chunks(job_id, 0, 0, 0)
    try:
        while True:
            chunks_pc.append(next(worker_iterator_pc))
    except StopIteration:
        pass

    logger.info("Fetched all chunks after restoring from checkpoint.")

    # Since the mixture was changed before the checkpoint,
    # we can only compare chunks from the checkpoint onwards
    checkpoint_chunk_index = 2  # Index of the chunk at which we checkpointed

    # Ensure the number of chunks after the checkpoint matches
    num_chunks_after_checkpoint = len(chunks) - checkpoint_chunk_index
    num_chunks_pc = len(chunks_pc)
    assert num_chunks_after_checkpoint == num_chunks_pc, (
        f"Number of chunks after checkpoint: {num_chunks_after_checkpoint} "
        f"does not match the number after restore: {num_chunks_pc}"
    )

    # Validate worker status: samples_to_skip should match at the checkpoint
    chunk_checkpoint = chunks[checkpoint_chunk_index]
    chunk_pc_0 = chunks_pc[0]
    assert (
        chunk_pc_0._samples_to_skip == 2
    ), f"At checkpoint, samples to skip: expected 2, got {chunk_pc_0._samples_to_skip}"

    logger.info("Validated worker status.")

    # Compare chunks from the checkpoint onwards
    for j in range(num_chunks_pc):
        pre_chunk = chunks[checkpoint_chunk_index + j]
        post_chunk = chunks_pc[j]

        # Assert that the chunks are equal
        TestCase().assertDictEqual(
            pre_chunk._result_index, post_chunk._result_index, f"Chunk {j} result index not equal after checkpoint."
        )

        TestCase().assertDictEqual(
            pre_chunk._mixture, post_chunk._mixture, f"Chunk {j} mixture not equal after checkpoint."
        )

    logger.info("Validated chunks after checkpoint.")


def main():
    print(f"Running local checkpointing tests with {mp.get_start_method()} start method.")

    with tempfile.TemporaryDirectory() as directory:
        dir = Path(directory)
        setup_test_dataset(dir, TEST_LOCAL_INSTANCE_COUNT, TEST_LOCAL_FILE_COUNT, TEST_LOCAL_FRACTION_MULTIPLIER)
        client = MixteraClient.from_directory(dir)
        client.register_metadata_parser("TEST_PARSER", TestMetadataParser)
        client.register_dataset(
            "client_integrationtest_chunk_reader_dataset", dir, JSONLDataset, parsing_func, "TEST_PARSER"
        )

        run_test_arbitrarymixture(client)
        run_test_staticmixture(client)

    print("Local checkpointing test done.")


if __name__ == "__main__":
    main()
