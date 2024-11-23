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
from mixtera.core.query import Query
from mixtera.core.query.mixture import MixtureSchedule, ScheduleEntry, MixtureKey, StaticMixture
from mixtera.network.client.client_feedback import ClientFeedback

TEST_LOCAL_INSTANCE_COUNT = 50
TEST_LOCAL_FILE_COUNT = 1
TEST_LOCAL_FRACTION_MULTIPLIER = 2


def parsing_func(sample):
    import json

    return json.loads(sample)["text"]

def run_mixture_schedule(client: MixteraClient):
    job_id = f"local_checkpoint_test_0_workers"
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
        dp_groups=1,
        nodes_per_group=1,
        num_workers=0,
    )

    client.execute_query(query, query_execution_args)
    result_streaming_args = ResultStreamingArgs(job_id=job_id)
    logger.info("Executed query.")
    # Retriving the first set of chunks.
    worker_iterator = client._stream_result_chunks(job_id, 0, 0, 0)
    try:
        while True:
            chunk = next(worker_iterator)
            assert chunk._mixture[MixtureKey({"language": ["JavaScript"]})] == 10, "Non Javascript mixture is selected"
    except StopIteration:
        pass

    # Sending the feedback and seeing the mixture update.
    feedback = ClientFeedback(100)
    client.process_feedback(job_id, feedback)
    worker_iterator = client._stream_result_chunks(job_id, 0, 0, 0)

    assert mixture_schedule.current_step == 100, "The training step information did not propagate."
    try:
        while True:
            chunk = next(worker_iterator)
            assert chunk._mixture[MixtureKey({"language": ["HTML"]})] == 10, "Non HTML mixture is selected"
    except StopIteration:
        pass


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

        run_mixture_schedule(client)

    print("Local feedback tests are done.")


if __name__ == "__main__":
    main()