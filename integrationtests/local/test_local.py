import multiprocessing as mp
import tempfile
from pathlib import Path

from integrationtests.utils import (
    REPRODUCIBILITY_ITERATIONS,
    TestMetadataParser,
    get_expected_js_and_html_samples,
    setup_test_dataset,
)
from loguru import logger
from mixtera.core.client import MixteraClient
from mixtera.core.client.mixtera_client import QueryExecutionArgs, ResultStreamingArgs
from mixtera.core.datacollection.datasets import JSONLDataset
from mixtera.core.query import Query
from mixtera.core.query.mixture import ArbitraryMixture, MixtureKey, MixtureSchedule, ScheduleEntry, StaticMixture
from mixtera.network.client.client_feedback import ClientFeedback

TEST_LOCAL_INSTANCE_COUNT = 1000
TEST_LOCAL_FILE_COUNT = 5
TEST_LOCAL_FRACTION_MULTIPLIER = 2

EXPECTED_JS_SAMPLES, EXPECTED_HTML_SAMPLES = get_expected_js_and_html_samples(
    TEST_LOCAL_INSTANCE_COUNT, TEST_LOCAL_FRACTION_MULTIPLIER
)


def parsing_func(sample):
    import json

    return json.loads(sample)["text"]


def test_filter_javascript(
    client: MixteraClient, query_exec_args: QueryExecutionArgs, result_streaming_args: ResultStreamingArgs
) -> None:
    result_streaming_args.job_id = (
        f"0_{query_exec_args.mixture.chunk_size}_{query_exec_args.dp_groups}"
        + f"_{query_exec_args.nodes_per_group}_{query_exec_args.num_workers}_{result_streaming_args.chunk_reading_degree_of_parallelism}"
        + f"_{result_streaming_args.chunk_reading_window_size}_{result_streaming_args.chunk_reading_per_window_mixture}"
    )
    query = Query.for_job(result_streaming_args.job_id).select(("language", "==", "JavaScript"))
    client.execute_query(query, query_exec_args)
    result_samples = []
    for sample in client.stream_results(result_streaming_args):
        result_samples.append(sample)

    assert (
        len(result_samples) == EXPECTED_JS_SAMPLES
    ), f"Got {len(result_samples)} samples instead of the expected {EXPECTED_JS_SAMPLES}!"
    for _, sample in result_samples:
        assert int(sample) % TEST_LOCAL_FRACTION_MULTIPLIER == 0, f"Sample {sample} should not appear for JavaScript"


def test_filter_html(
    client: MixteraClient, query_exec_args: QueryExecutionArgs, result_streaming_args: ResultStreamingArgs
):
    result_streaming_args.job_id = (
        f"1_{query_exec_args.mixture.chunk_size}_{query_exec_args.dp_groups}"
        + f"_{query_exec_args.nodes_per_group}_{query_exec_args.num_workers}_{result_streaming_args.chunk_reading_degree_of_parallelism}"
        + f"_{result_streaming_args.chunk_reading_window_size}_{result_streaming_args.chunk_reading_per_window_mixture}"
    )
    query = Query.for_job(result_streaming_args.job_id).select(("language", "==", "HTML"))
    client.execute_query(query, query_exec_args)
    result_samples = []

    for sample in client.stream_results(result_streaming_args):
        result_samples.append(sample)

    assert (
        len(result_samples) == EXPECTED_HTML_SAMPLES
    ), f"Got {len(result_samples)} samples instead of the expected {EXPECTED_HTML_SAMPLES}!"
    for _, sample in result_samples:
        assert int(sample) % TEST_LOCAL_FRACTION_MULTIPLIER == 1, f"Sample {sample} should not appear for HTML"


def test_filter_both(
    client: MixteraClient, query_exec_args: QueryExecutionArgs, result_streaming_args: ResultStreamingArgs
):
    result_streaming_args.job_id = (
        f"2_{query_exec_args.mixture.chunk_size}_{query_exec_args.dp_groups}"
        + f"_{query_exec_args.nodes_per_group}_{query_exec_args.num_workers}_{result_streaming_args.chunk_reading_degree_of_parallelism}"
        + f"_{result_streaming_args.chunk_reading_window_size}_{result_streaming_args.chunk_reading_per_window_mixture}"
    )
    query = (
        Query.for_job(result_streaming_args.job_id)
        .select(("language", "==", "HTML"))
        .select(("language", "==", "JavaScript"))
    )
    client.execute_query(query, query_exec_args)
    result_samples = []

    for sample in client.stream_results(result_streaming_args):
        result_samples.append(sample)

    assert (
        len(result_samples) == TEST_LOCAL_INSTANCE_COUNT
    ), f"Got {len(result_samples)} samples instead of the expected {TEST_LOCAL_INSTANCE_COUNT}!"
    for _, sample in result_samples:
        assert 0 <= int(sample) < TEST_LOCAL_INSTANCE_COUNT, f"Sample {sample} should not appear"


def test_filter_license(
    client: MixteraClient, query_exec_args: QueryExecutionArgs, result_streaming_args: ResultStreamingArgs
):
    result_streaming_args.job_id = (
        f"3_{query_exec_args.mixture.chunk_size}_{query_exec_args.dp_groups}"
        + f"_{query_exec_args.nodes_per_group}_{query_exec_args.num_workers}_{result_streaming_args.chunk_reading_degree_of_parallelism}"
        + f"_{result_streaming_args.chunk_reading_window_size}_{result_streaming_args.chunk_reading_per_window_mixture}"
    )
    query = Query.for_job(result_streaming_args.job_id).select(("license", "==", "CC"))
    client.execute_query(query, query_exec_args)
    result_samples = []

    for sample in client.stream_results(result_streaming_args):
        result_samples.append(sample)

    assert (
        len(result_samples) == TEST_LOCAL_INSTANCE_COUNT // 2
    ), f"Got {len(result_samples)} samples instead of the expected {TEST_LOCAL_INSTANCE_COUNT}!"
    for _, sample in result_samples:
        assert 0 <= int(sample) < TEST_LOCAL_INSTANCE_COUNT, f"Sample {sample} should not appear"


def test_filter_unknown_license(
    client: MixteraClient, query_exec_args: QueryExecutionArgs, result_streaming_args: ResultStreamingArgs
):
    result_streaming_args.job_id = (
        f"4_{query_exec_args.mixture.chunk_size}_{query_exec_args.dp_groups}"
        + f"_{query_exec_args.nodes_per_group}_{query_exec_args.num_workers}_{result_streaming_args.chunk_reading_degree_of_parallelism}"
        + f"_{result_streaming_args.chunk_reading_window_size}_{result_streaming_args.chunk_reading_per_window_mixture}"
    )
    query = Query.for_job(result_streaming_args.job_id).select(("license", "==", "All rights reserved."))
    client.execute_query(query, query_exec_args)
    assert len(list(client.stream_results(result_streaming_args))) == 0, "Got results back for expected empty results."


def test_filter_license_and_html(
    client: MixteraClient, query_exec_args: QueryExecutionArgs, result_streaming_args: ResultStreamingArgs
):
    result_streaming_args.job_id = (
        f"5_{query_exec_args.mixture.chunk_size}_{query_exec_args.dp_groups}"
        + f"_{query_exec_args.nodes_per_group}_{query_exec_args.num_workers}_{result_streaming_args.chunk_reading_degree_of_parallelism}"
        + f"_{result_streaming_args.chunk_reading_window_size}_{result_streaming_args.chunk_reading_per_window_mixture}"
    )
    query = (
        Query.for_job(result_streaming_args.job_id).select(("language", "==", "HTML")).select(("license", "==", "CC"))
    )
    client.execute_query(query, query_exec_args)
    result_samples = []

    for sample in client.stream_results(result_streaming_args):
        result_samples.append(sample)

    expected_samples = EXPECTED_HTML_SAMPLES + EXPECTED_JS_SAMPLES // 2
    assert (
        len(result_samples) == expected_samples
    ), f"Got {len(result_samples)} samples instead of the expected {expected_samples}!"
    for _, sample in result_samples:
        assert 0 <= int(sample) < expected_samples, f"Sample {sample} should not appear"


def test_reproducibility(
    client: MixteraClient, query_exec_args: QueryExecutionArgs, result_streaming_args: ResultStreamingArgs
):
    mixture = StaticMixture(
        query_exec_args.mixture.chunk_size,
        {MixtureKey({"language": ["JavaScript"]}): 0.6, MixtureKey({"language": ["HTML"]}): 0.4},
    )
    result_list = []

    for i in range(REPRODUCIBILITY_ITERATIONS):
        result_streaming_args.job_id = (
            f"6_{query_exec_args.mixture.chunk_size}_{query_exec_args.dp_groups}"
            + f"_{query_exec_args.nodes_per_group}_{query_exec_args.num_workers}_{result_streaming_args.chunk_reading_degree_of_parallelism}"
            + f"_{result_streaming_args.chunk_reading_window_size}_{result_streaming_args.chunk_reading_per_window_mixture}"
            + f"_reproducibility_{i}"
        )
        query = (
            Query.for_job(result_streaming_args.job_id)
            .select(("language", "==", "HTML"))
            .select(("language", "==", "JavaScript"))
        )
        query_exec_args.mixture = mixture
        client.execute_query(query, query_exec_args)
        result_samples = []

        for sample in client.stream_results(result_streaming_args):
            result_samples.append(sample)

        result_list.append(result_samples)

    for i in range(1, REPRODUCIBILITY_ITERATIONS):
        assert result_list[i] == result_list[i - 1], "Results are not reproducible"


def test_mixture_schedule(client: MixteraClient):
    job_id = f"local_feedback_test_0_workers"
    query = Query.for_job(job_id).select(None)

    chunk_size = 10
    mixture_schedule = MixtureSchedule(
        chunk_size,
        [
            ScheduleEntry(0, StaticMixture(chunk_size, {MixtureKey({"language": ["JavaScript"]}): 1.0})),
            ScheduleEntry(100, StaticMixture(chunk_size, {MixtureKey({"language": ["HTML"]}): 1.0})),
            ScheduleEntry(
                200,
                StaticMixture(
                    chunk_size, {MixtureKey({"language": ["JavaScript"]}): 0.5, MixtureKey({"language": ["HTML"]}): 0.5}
                ),
            ),
        ],
    )

    query_execution_args = QueryExecutionArgs(
        mixture=mixture_schedule,
        dp_groups=1,
        nodes_per_group=1,
        num_workers=0,
    )

    client.execute_query(query, query_execution_args)
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

    # Sending the feedback and seeing the mixture update.
    feedback = ClientFeedback(200)
    client.process_feedback(job_id, feedback)
    worker_iterator = client._stream_result_chunks(job_id, 0, 0, 0)

    assert mixture_schedule.current_step == 200, "The training step information did not propagate."
    try:
        while True:
            chunk = next(worker_iterator)
            assert chunk._mixture[MixtureKey({"language": ["HTML"]})] == 5, "Wrong amount of HTML data is selected."
            assert (
                chunk._mixture[MixtureKey({"language": ["Javascript"]})] == 5
            ), "Wrong amount of Javascript data is selected."
    except StopIteration:
        pass


def test_client_chunksize(
    client: MixteraClient, query_exec_args: QueryExecutionArgs, result_streaming_args: ResultStreamingArgs
):
    test_filter_javascript(client, query_exec_args, result_streaming_args)
    test_filter_html(client, query_exec_args, result_streaming_args)
    test_filter_both(client, query_exec_args, result_streaming_args)
    test_filter_license(client, query_exec_args, result_streaming_args)
    test_filter_unknown_license(client, query_exec_args, result_streaming_args)
    test_filter_license_and_html(client, query_exec_args, result_streaming_args)
    test_reproducibility(client, query_exec_args, result_streaming_args)
    test_mixture_schedule(client)


def test_chunk_readers(dir: Path) -> None:
    setup_test_dataset(dir, TEST_LOCAL_INSTANCE_COUNT, TEST_LOCAL_FILE_COUNT, TEST_LOCAL_FRACTION_MULTIPLIER)
    client = MixteraClient.from_directory(dir)
    client.register_metadata_parser("TEST_PARSER", TestMetadataParser)
    client.register_dataset(
        "client_integrationtest_chunk_reader_dataset", dir, JSONLDataset, parsing_func, "TEST_PARSER"
    )

    reader_degrees_of_parallelisms = [1, 4]
    per_window_mixtures = [False, True]
    window_sizes = [64, 128]

    for chunk_size in [100, 500]:
        for reader_degree_of_parallelism in reader_degrees_of_parallelisms:
            for per_window_mixture in per_window_mixtures:
                for window_size in window_sizes:
                    #  Only run per_window_mixture tests for one window size (subsequent are equivalent)
                    if not per_window_mixture and window_size > 64:
                        continue
                    query_exec_args = QueryExecutionArgs(mixture=ArbitraryMixture(chunk_size))
                    result_streaming_args = ResultStreamingArgs(
                        None,
                        chunk_reading_degree_of_parallelism=reader_degree_of_parallelism,
                        chunk_reading_per_window_mixture=per_window_mixture,
                        chunk_reading_window_size=window_size,
                    )
                    logger.debug(
                        f"Running chunk reader tests with chunk_size={chunk_size}, degree_of_parallelism={reader_degree_of_parallelism}, "
                        f"per_window_mixture={per_window_mixture}, window_size={window_size}"
                    )
                    test_client_chunksize(client, query_exec_args, result_streaming_args)

    print("Successfully ran chunk reader tests!")

    client.remove_dataset("client_integrationtest_chunk_reader_dataset")


def main() -> None:
    print(f"Running tests with {mp.get_start_method()} start method.")
    with tempfile.TemporaryDirectory() as directory:
        test_chunk_readers(Path(directory))

    print("Local tests done.")


if __name__ == "__main__":
    main()
