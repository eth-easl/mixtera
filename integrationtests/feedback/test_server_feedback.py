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
from mixtera.core.query import Query, ResultChunk
from mixtera.core.query.mixture import StaticMixture, MixtureSchedule, ScheduleEntry, MixtureKey
from mixtera.network.client.client_feedback import ClientFeedback

TEST_SERVER_INSTANCE_COUNT = 1000
TEST_SERVER_FILE_COUNT = 5
TEST_SERVER_FRACTION_MULTIPLIER = 5

def parsing_func(sample):
    import json

    return json.loads(sample)["text"]


def run_test_mixture_schedule_server(client: ServerStub, dp_groups, nodes_per_group, num_workers):
    job_id = f"server_checkpoint_test_dpg{dp_groups}_npg{nodes_per_group}_nw{num_workers}"
    query = Query.for_job(job_id).select(None)

    chunk_size = 10
    mixture_schedule = MixtureSchedule(
            chunk_size,
            [
                ScheduleEntry(0, StaticMixture(chunk_size, {MixtureKey({"language": ["JavaScript"]}): 1.0})),
                ScheduleEntry(100, StaticMixture(chunk_size, {MixtureKey({"language": ["HTML"]}): 1.0})),
            ],
        )
    
    query_execution_args = QueryExecutionArgs(
        mixture=mixture_schedule,
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

    # For each dp_group, node, and worker, simulate fetching at least one chunk from the initial mixture
    for dp_group_id in range(dp_groups):
        chunks[dp_group_id] = {}
        for node_id in range(nodes_per_group):
            chunks[dp_group_id][node_id] = {}
            for worker_id in range(num_workers if num_workers > 0 else 1):
                # Create an iterator for this worker
                worker_iterator = client._stream_result_chunks(job_id, dp_group_id, node_id, worker_id)
                worker_chunks = []
                try:
                    # Get at least one chunk
                    chunk = next(worker_iterator)
                    assert chunk._mixture[MixtureKey({"language": ["JavaScript"]})] == 10, "Non Javascript mixture is selected"
                    worker_chunks.append(chunk)
                    # Simulate that the worker consumed samples_consumed samples in the current chunk
                    samples_consumed = worker_id + 2  # Each worker reports status as worker_id + 2 samples consumed
                    chunk._samples_to_skip = samples_consumed
                except StopIteration:
                    logger.error(
                        f"Unexpected end of data while fetching initial chunk for dp_group {dp_group_id}, node {node_id}, worker {worker_id}."
                    )
                    return
                chunks[dp_group_id][node_id][worker_id] = {
                    "iterator": worker_iterator,
                    "chunks": worker_chunks,
                }

    feedback = ClientFeedback(100)
    client.process_feedback(job_id, feedback)

    # For each dp_group, node, and worker, simulate fetching at least one chunk from the next mixture
    for dp_group_id in range(dp_groups):
        chunks[dp_group_id] = {}
        for node_id in range(nodes_per_group):
            chunks[dp_group_id][node_id] = {}
            for worker_id in range(num_workers if num_workers > 0 else 1):
                # Create an iterator for this worker
                worker_iterator = client._stream_result_chunks(job_id, dp_group_id, node_id, worker_id)
                worker_chunks = []
                try:
                    # Get at least one chunk
                    chunk = next(worker_iterator)
                    assert chunk._mixture[MixtureKey({"language": ["HTML"]})] == 10, "Non Javascript mixture is selected"
                    worker_chunks.append(chunk)
                    # Simulate that the worker consumed samples_consumed samples in the current chunk
                    samples_consumed = worker_id + 2  # Each worker reports status as worker_id + 2 samples consumed
                    chunk._samples_to_skip = samples_consumed
                except StopIteration:
                    logger.error(
                        f"Unexpected end of data while fetching initial chunk for dp_group {dp_group_id}, node {node_id}, worker {worker_id}."
                    )
                    return
                chunks[dp_group_id][node_id][worker_id] = {
                    "iterator": worker_iterator,
                    "chunks": worker_chunks,
                }
                
    logger.info("Successfully trained with schedule.")

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python test_server_feedback.py <server_directory>")
        sys.exit(1)

    server_dir = Path(sys.argv[1])

    setup_test_dataset(server_dir, TEST_SERVER_INSTANCE_COUNT, TEST_SERVER_FILE_COUNT, TEST_SERVER_FRACTION_MULTIPLIER)
    client = MixteraClient.from_remote("127.0.0.1", 6666)

    assert client.register_metadata_parser("TEST_PARSER_SERVER_FEEDBACK", TestMetadataParser)
    assert client.register_dataset(
        "server_integrationtest_feedback_dataset",
        server_dir,
        JSONLDataset,
        parsing_func,
        "TEST_PARSER_SERVER_CHKPNT",
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
                    run_test_mixture_schedule_server(client, dp_groups, nodes_per_group, num_workers)
                except Exception as e:
                    logger.error(
                        f"Error with configuration dp_groups={dp_groups}, nodes_per_group={nodes_per_group}, num_workers={num_workers}"
                    )
                    logger.exception(e)
                    raise e

    logger.info("Server feedback tests completed.")


if __name__ == "__main__":
    main()