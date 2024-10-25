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
    logger.info("Got first chunks, initiating checkpoint.")

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

    logger.info("Got all chunks.")

    client.restore_checkpoint(job_id, checkpoint_id)

    logger.info("Restored checkpoint.")

    # Obtain post checkpoint chunks
    chunks_pc: list[list[ResultChunk]] = [[] for _ in range(max(num_workers, 1)) ]
    for i, worker_iterator in enumerate(worker_iterators):
        try:
            while True:
                chunks_pc[i].append(next(worker_iterator))
        except StopIteration:
            continue

    logger.info("Obtained all chunks again.")

    # Validate for chunk 0 that worker status has been transferred correctly
    for i in range(max(num_workers, 1)):
        assert chunks_pc[i][0]._samples_to_skip == i + 2, f"Worker {i} chunk 0: skipping { chunks_pc[i][0]._samples_to_skip} instead of {i+2} samples."

    logger.info("Validated worker status.")

    # Validate for all chunks that the content is equivalent
    for i in range(max(num_workers, 1)):
        assert len(chunks_pc[i]) == len(chunks[i]), f"Worker {i}: Pre Checkpoint: {len(chunks[i])} chunks, post checkpoint: {len(chunks_pc[i]) } chunks"
        for j,pre_chunk, post_chunk in enumerate(zip(chunks_pc[i], chunks[i])):
            if j == 0:
                post_chunk._samples_to_skip = 0 # to make objects equal

            TestCase().assertDictEqual(pre_chunk._mixture, post_chunk._mixture)
            TestCase().assertDictEqual(pre_chunk._result_index, post_chunk._result_index)
            TestCase().assertEqual(pre_chunk, post_chunk)


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