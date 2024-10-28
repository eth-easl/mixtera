import sys
import tempfile
from pathlib import Path
from time import sleep
from unittest import TestCase

from integrationtests.utils import TestMetadataParser, setup_test_dataset
from loguru import logger
from mixtera.core.client.mixtera_client import MixteraClient, QueryExecutionArgs, ResultStreamingArgs
from mixtera.core.client.server.server_stub import ServerStub
from mixtera.core.datacollection.datasets import JSONLDataset
from mixtera.core.query import ArbitraryMixture, Query, ResultChunk

TEST_SERVER_INSTANCE_COUNT = 1000
TEST_SERVER_FILE_COUNT = 5
TEST_SERVER_FRACTION_MULTIPLIER = 5


def parsing_func(sample):
    import json

    return json.loads(sample)["text"]


def run_test_arbitrarymixture_server(client: ServerStub, dp_groups, nodes_per_group, num_workers):
    job_id = f"server_checkpoint_test_dpg{dp_groups}_npg{nodes_per_group}_nw{num_workers}"
    query = Query.for_job(job_id).select(None)

    query_execution_args = QueryExecutionArgs(
        mixture=ArbitraryMixture(chunk_size=5),
        dp_groups=dp_groups,
        nodes_per_group=nodes_per_group,
        num_workers=num_workers,
    )
    client.execute_query(query, query_execution_args)
    logger.info(
        f"Executed query for job {job_id} with dp_groups={dp_groups}, nodes_per_group={nodes_per_group}, num_workers={num_workers}"
    )

    # Simulating workers per node per dp_group
    chunks = {}  # Stores chunks: {dp_group_id: {node_id: {worker_id: {'iterator': iterator, 'chunks': [chunks]}}}}
    worker_statuses = {}  # Stores worker statuses: {(dp_group_id, node_id): [worker_status_list]}

    # For each dp_group, node, and worker, simulate fetching at least one chunk
    for dp_group_id in range(dp_groups):
        chunks[dp_group_id] = {}
        for node_id in range(nodes_per_group):
            chunks[dp_group_id][node_id] = {}
            worker_status_list = []
            for worker_id in range(num_workers if num_workers > 0 else 1):
                # Create an iterator for this worker
                worker_iterator = client._stream_result_chunks(job_id, dp_group_id, node_id, worker_id)
                worker_chunks = []
                try:
                    # Get at least one chunk
                    chunk = next(worker_iterator)
                    worker_chunks.append(chunk)
                    # Simulate that the worker consumed samples_consumed samples in the current chunk
                    samples_consumed = worker_id + 2  # Each worker reports status as worker_id + 2 samples consumed
                    chunk._samples_to_skip = samples_consumed
                    worker_status_list.append(samples_consumed)
                except StopIteration:
                    logger.error(
                        f"Unexpected end of data while fetching initial chunk for dp_group {dp_group_id}, node {node_id}, worker {worker_id}."
                    )
                    return
                chunks[dp_group_id][node_id][worker_id] = {
                    "iterator": worker_iterator,
                    "chunks": worker_chunks,
                }
            worker_statuses[(dp_group_id, node_id)] = worker_status_list

    # Initiate checkpoint
    logger.info("Initiating checkpoint.")
    checkpoint_id = None
    for (dp_group_id, node_id), statuses in worker_statuses.items():
        cid = client.checkpoint(
            job_id=job_id,
            dp_group_id=dp_group_id,
            node_id=node_id,
            worker_status=statuses,
        )
        if checkpoint_id is None:
            checkpoint_id = cid
        else:
            assert checkpoint_id == cid, "Checkpoint IDs do not match across nodes."

    # Wait for checkpoint to complete
    while not client.checkpoint_completed(job_id, checkpoint_id, True):
        sleep(0.1)

    logger.info("Checkpoint completed.")

    # After checkpoint, consume all remaining chunks for each worker
    for dp_group_id in range(dp_groups):
        for node_id in range(nodes_per_group):
            for worker_id in range(num_workers if num_workers > 0 else 1):
                worker_info = chunks[dp_group_id][node_id][worker_id]
                worker_iterator = worker_info["iterator"]
                worker_chunks = worker_info["chunks"]
                try:
                    while True:
                        chunk = next(worker_iterator)
                        worker_chunks.append(chunk)
                except StopIteration:
                    pass

    logger.info("Fetched all chunks after checkpoint.")

    # Restore from checkpoint
    client.restore_checkpoint(job_id, checkpoint_id)
    logger.info("Restored from checkpoint.")

    # Obtain chunks after restoring from checkpoint
    chunks_pc = {}  # Stores restored chunks
    for dp_group_id in range(dp_groups):
        chunks_pc[dp_group_id] = {}
        for node_id in range(nodes_per_group):
            chunks_pc[dp_group_id][node_id] = {}
            for worker_id in range(num_workers if num_workers > 0 else 1):
                worker_iterator_pc = client._stream_result_chunks(job_id, dp_group_id, node_id, worker_id)
                worker_chunks_pc = []
                try:
                    while True:
                        chunk = next(worker_iterator_pc)
                        worker_chunks_pc.append(chunk)
                except StopIteration:
                    pass
                chunks_pc[dp_group_id][node_id][worker_id] = worker_chunks_pc

    logger.info("Fetched all chunks after restoring from checkpoint.")

    # Compare chunks after the checkpoint
    for dp_group_id in range(dp_groups):
        for node_id in range(nodes_per_group):
            for worker_id in range(num_workers if num_workers > 0 else 1):
                worker_chunks_original = chunks[dp_group_id][node_id][worker_id]["chunks"]
                worker_chunks_restored = chunks_pc[dp_group_id][node_id][worker_id]

                # Validate worker status: samples_to_skip should match the expected value
                expected_samples_to_skip = worker_id + 2
                restored_chunk = worker_chunks_restored[0]
                assert restored_chunk._samples_to_skip == expected_samples_to_skip, (
                    f"Worker {worker_id} on node {node_id} in dp_group {dp_group_id}: "
                    f"samples_to_skip mismatch after checkpoint: expected {expected_samples_to_skip}, got {restored_chunk._samples_to_skip}"
                )

                # Compare the rest of the chunks
                len_original = len(worker_chunks_original)
                len_restored = len(worker_chunks_restored)
                assert len_original == len_restored, (
                    f"Chunk count mismatch for worker {worker_id} on node {node_id} in dp_group {dp_group_id}: "
                    f"original {len_original}, restored {len_restored}"
                )

                for i in range(len_original):
                    pre_chunk = worker_chunks_original[i]
                    post_chunk = worker_chunks_restored[i]
                    TestCase().assertDictEqual(
                        pre_chunk._result_index,
                        post_chunk._result_index,
                        f"Mismatch in result index for worker {worker_id}, node {node_id}, dp_group {dp_group_id}, chunk {i}",
                    )
                    if pre_chunk._mixture is None:
                        assert post_chunk._mixture is None
                    else:
                        TestCase().assertDictEqual(
                            pre_chunk._mixture,
                            post_chunk._mixture,
                            f"Mismatch in mixture for worker {worker_id}, node {node_id}, dp_group {dp_group_id}, chunk {i}",
                        )

    logger.info("Validated chunks after checkpoint.")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python test_server_checkpointing.py <server_directory>")
        sys.exit(1)

    server_dir = Path(sys.argv[1])

    setup_test_dataset(server_dir, TEST_SERVER_INSTANCE_COUNT, TEST_SERVER_FILE_COUNT, TEST_SERVER_FRACTION_MULTIPLIER)
    client = MixteraClient.from_remote("127.0.0.1", 6666)

    assert client.register_metadata_parser("TEST_PARSER_SERVER_CHKPNT", TestMetadataParser)
    assert client.register_dataset(
        "server_integrationtest_checkpointing_dataset", server_dir, JSONLDataset, parsing_func, "TEST_PARSER_SERVER_CHKPNT"
    )
    logger.info("Registered dataset on server.")

    dp_groups_list = [1, 2, 4]
    nodes_per_group_list = [1, 2, 4]
    num_workers_list = [0, 1, 8]

    for dp_groups in dp_groups_list:
        for nodes_per_group in nodes_per_group_list:
            for num_workers in num_workers_list:
                try:
                    logger.info(
                        f"Testing with dp_groups={dp_groups}, nodes_per_group={nodes_per_group}, num_workers={num_workers}"
                    )
                    run_test_arbitrarymixture_server(client, dp_groups, nodes_per_group, num_workers)
                except Exception as e:
                    logger.error(
                        f"Error with configuration dp_groups={dp_groups}, nodes_per_group={nodes_per_group}, num_workers={num_workers}"
                    )
                    logger.exception(e)
                    raise e

    logger.info("Server checkpointing tests completed.")


if __name__ == "__main__":
    main()
