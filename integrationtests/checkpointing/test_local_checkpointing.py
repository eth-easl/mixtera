# test_local_checkpointing.py

import multiprocessing as mp
import os
import tempfile
from pathlib import Path
from integrationtests.utils import REPRODUCIBILITY_ITERATIONS, TestMetadataParser, setup_test_dataset

from mixtera.core.client.mixtera_client import MixteraClient, QueryExecutionArgs, ResultStreamingArgs
from mixtera.core.datacollection.datasets import JSONLDataset
from mixtera.core.query import Query, ArbitraryMixture, ResultChunk
from unittest import TestCase

from loguru import logger
from time import sleep

TEST_LOCAL_INSTANCE_COUNT = 50
TEST_LOCAL_FILE_COUNT = 1
TEST_LOCAL_FRACTION_MULTIPLIER = 2

def parsing_func(sample):
    import json

    return json.loads(sample)["text"]


def run_test(num_workers, client: MixteraClient):
    job_id = f"local_checkpoint_test_{num_workers}_workers"
    query = Query.for_job(job_id).select(None)
    query_execution_args = QueryExecutionArgs(
        mixture=ArbitraryMixture(chunk_size=5), # 10 Chunks overall
        dp_groups=1,
        nodes_per_group=1,
        num_workers=num_workers,
    )
    client.execute_query(query, query_execution_args)
    result_streaming_args = ResultStreamingArgs(job_id=job_id)
    logger.info("Executed query.")
    # Get one chunk for each worker
    worker_iterators = [client._stream_result_chunks(job_id, 0,0, i) for i in range(max(num_workers, 1))]
    chunks = [[next(worker_iterators[i])] for i in range(max(num_workers, 1)) ]

    worker_status = [i + 2 for i in range(max(num_workers, 1))] # report each worker has consumed i + 2 samples
    logger.info(f"Got first chunks = {chunks}, initiating checkpoint.")

    checkpoint_id = client.checkpoint(
        job_id=job_id,
        dp_group_id=0,
        node_id=0,
        worker_status=worker_status,
    )

    while not client.checkpoint_completed(job_id, checkpoint_id, False):
        sleep(0.1)

    logger.info("Checkpoint done.")

    # Now after checkpoint consume all chunks for later comparison
    for i, worker_iterator in enumerate(worker_iterators):
        try:
            while True:
                chunks[i].append(next(worker_iterator))
        except StopIteration:
            continue

    logger.info(f"Got all chunks = {chunks}.")
    # Issue is with local checkpointing: main process does not have updated queryresult (always keeps initial ones), only worker fork processes update
    # HOWEVER, right now we fetch all chunks in the same process so we should not actually run into this issue? something else must be off.
    client.restore_checkpoint(job_id, checkpoint_id)

    logger.info("Restored checkpoint.")

    # Obtain post checkpoint chunks
    chunks_pc: list[list[ResultChunk]] = [[] for _ in range(max(num_workers, 1)) ]
    worker_iterators_pc = [client._stream_result_chunks(job_id, 0,0, i) for i in range(max(num_workers, 1))]
    for i, worker_iterator in enumerate(worker_iterators_pc):
        try:
            while True:
                chunks_pc[i].append(next(worker_iterator))
        except StopIteration:
            continue

    logger.info(f"Obtained all chunks again = {chunks_pc}.")

    assert len(chunks) == len(chunks_pc), f"len(chunks) = {len(chunks)} len(chunks_pc) = {len(chunks_pc)}\n{chunks}\n{chunks_pc}"

    # Validate for chunk 0 that worker status has been transferred correctly
    for i in range(max(num_workers, 1)):
        assert chunks_pc[i][0]._samples_to_skip == i + 2, f"Worker {i} chunk 0: skipping { chunks_pc[i][0]._samples_to_skip} instead of {i+2} samples."
        for chunk_id in range(1, len(chunks_pc[i])):
            assert chunks_pc[i][chunk_id]._samples_to_skip == 0,  f"Worker {i} chunk {chunk_id}: skipping { chunks_pc[i][chunk_id]._samples_to_skip} instead of 0 samples."

    logger.info("Validated worker status.")

    logger.debug(chunks)
    logger.debug(chunks_pc)

    # Validate for all chunks that the content is equivalent
    for i in range(max(num_workers, 1)):
        #assert len(chunks_pc[i]) == len(chunks[i]), f"Worker {i}: Pre Checkpoint: {len(chunks[i])} chunks, post checkpoint: {len(chunks_pc[i])} chunks\n{chunks[i]}\n{chunks_pc[i]}"
        for j in range(len(chunks_pc[i])):
            pre_chunk = chunks[i][j]
            post_chunk = chunks_pc[i][j] 
            if j == 0:
                post_chunk._samples_to_skip = 0 # to make objects equal

            logger.debug(f"chunk {j}")
            logger.debug(pre_chunk._result_index)
            logger.debug(post_chunk._result_index)
            logger.debug("")

            #TestCase().assertDictEqual(pre_chunk._result_index, post_chunk._result_index, f"worker {i} chunk {j} not equal")
            #TestCase().assertEqual(pre_chunk, post_chunk, f"worker {i} chunk {j} not equal")


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

        for num_workers in [0,1,8]:
            run_test(num_workers, client)

    print("Local checkpointing tests done.")

if __name__ == "__main__":
    main()