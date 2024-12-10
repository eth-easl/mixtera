import sys
from pathlib import Path
from typing import Any

import numpy as np
from integrationtests.utils import (
    REPRODUCIBILITY_ITERATIONS,
    TestMetadataParser,
    calc_func,
    get_expected_js_and_html_samples,
    setup_func,
    setup_test_dataset,
)
from loguru import logger
from mixtera.core.algo.loss_avg.loss_avg import SimpleAveragingAlgorithm
from mixtera.core.client import MixteraClient
from mixtera.core.client.mixtera_client import QueryExecutionArgs, ResultStreamingArgs
from mixtera.core.client.server import ServerStub
from mixtera.core.datacollection.datasets import JSONLDataset
from mixtera.core.datacollection.property_type import PropertyType
from mixtera.core.processing.execution_mode import ExecutionMode
from mixtera.core.query import Query
from mixtera.core.query.mixture import ArbitraryMixture, MixtureKey, MixtureSchedule, ScheduleEntry, StaticMixture
from mixtera.core.query.mixture.dynamic_mixture import DynamicMixture
from mixtera.network.client.client_feedback import ClientFeedback

TEST_SERVER_INSTANCE_COUNT = 1000
TEST_SERVER_FILE_COUNT = 5
TEST_SERVER_FRACTION_MULTIPLIER = 2

EXPECTED_JS_SAMPLES, EXPECTED_HTML_SAMPLES = get_expected_js_and_html_samples(
    TEST_SERVER_INSTANCE_COUNT, TEST_SERVER_FRACTION_MULTIPLIER
)


def parsing_func(sample):
    import json

    return json.loads(sample)["text"]


def test_filter_javascript(
    client: ServerStub, query_exec_args: QueryExecutionArgs, result_streaming_args: ResultStreamingArgs
):
    result_streaming_args.job_id = (
        f"0_{query_exec_args.mixture.chunk_size}_{query_exec_args.dp_groups}"
        + f"_{query_exec_args.nodes_per_group}_{query_exec_args.num_workers}_{result_streaming_args.chunk_reading_degree_of_parallelism}"
        + f"_{result_streaming_args.chunk_reading_window_size}_{result_streaming_args.chunk_reading_mixture_type}"
        + f"_{result_streaming_args.tunnel_via_server}"
    )
    query = Query.for_job(result_streaming_args.job_id).select(("language", "==", "JavaScript"))
    assert client.execute_query(query, query_exec_args)
    result_samples = []

    for sample in client.stream_results(result_streaming_args):
        result_samples.append(sample)

    assert (
        len(result_samples) == EXPECTED_JS_SAMPLES
    ), f"Got {len(result_samples)} samples instead of the expected {EXPECTED_JS_SAMPLES}!"
    for _, _, sample in result_samples:
        assert int(sample) % 2 == 0, f"Sample {sample} should not appear for JavaScript"


def test_filter_html(
    client: ServerStub, query_exec_args: QueryExecutionArgs, result_streaming_args: ResultStreamingArgs
):
    result_streaming_args.job_id = (
        f"1_{query_exec_args.mixture.chunk_size}_{query_exec_args.dp_groups}"
        + f"_{query_exec_args.nodes_per_group}_{query_exec_args.num_workers}_{result_streaming_args.chunk_reading_degree_of_parallelism}"
        + f"_{result_streaming_args.chunk_reading_window_size}_{result_streaming_args.chunk_reading_mixture_type}"
        + f"_{result_streaming_args.tunnel_via_server}"
    )
    query = Query.for_job(result_streaming_args.job_id).select(("language", "==", "HTML"))
    assert client.execute_query(query, query_exec_args)
    result_samples = []

    for sample in client.stream_results(result_streaming_args):
        result_samples.append(sample)

    assert (
        len(result_samples) == EXPECTED_HTML_SAMPLES
    ), f"Got {len(result_samples)} samples instead of the expected {EXPECTED_HTML_SAMPLES}!"
    for _, _, sample in result_samples:
        assert int(sample) % 2 == 1, f"Sample {sample} should not appear for HTML"


def test_filter_both(
    client: ServerStub, query_exec_args: QueryExecutionArgs, result_streaming_args: ResultStreamingArgs
):
    result_streaming_args.job_id = (
        f"2_{query_exec_args.mixture.chunk_size}_{query_exec_args.dp_groups}"
        + f"_{query_exec_args.nodes_per_group}_{query_exec_args.num_workers}_{result_streaming_args.chunk_reading_degree_of_parallelism}"
        + f"_{result_streaming_args.chunk_reading_window_size}_{result_streaming_args.chunk_reading_mixture_type}"
        + f"_{result_streaming_args.tunnel_via_server}"
    )
    query = (
        Query.for_job(result_streaming_args.job_id)
        .select(("language", "==", "HTML"))
        .select(("language", "==", "JavaScript"))
    )
    assert client.execute_query(query, query_exec_args)
    result_samples = []

    for sample in client.stream_results(result_streaming_args):
        result_samples.append(sample)

    assert (
        len(result_samples) == TEST_SERVER_INSTANCE_COUNT
    ), f"Got {len(result_samples)} samples instead of {TEST_SERVER_INSTANCE_COUNT}!"
    for _, _, sample in result_samples:
        assert 0 <= int(sample) < TEST_SERVER_INSTANCE_COUNT, f"Sample {sample} should not appear"


def test_filter_license(
    client: ServerStub, query_exec_args: QueryExecutionArgs, result_streaming_args: ResultStreamingArgs
):
    result_streaming_args.job_id = (
        f"3_{query_exec_args.mixture.chunk_size}_{query_exec_args.dp_groups}"
        + f"_{query_exec_args.nodes_per_group}_{query_exec_args.num_workers}_{result_streaming_args.chunk_reading_degree_of_parallelism}"
        + f"_{result_streaming_args.chunk_reading_window_size}_{result_streaming_args.chunk_reading_mixture_type}"
        + f"_{result_streaming_args.tunnel_via_server}"
    )
    query = Query.for_job(result_streaming_args.job_id).select(("license", "==", "CC"))
    assert client.execute_query(query, query_exec_args)
    result_samples = []

    for sample in client.stream_results(result_streaming_args):
        result_samples.append(sample)

    num_cc_samples = TEST_SERVER_INSTANCE_COUNT // 2

    assert (
        len(result_samples) == num_cc_samples
    ), f"Got {len(result_samples)} samples instead of the expected {num_cc_samples}!"
    for _, _, sample in result_samples:
        assert 0 <= int(sample) < TEST_SERVER_INSTANCE_COUNT, f"Sample {sample} should not appear"


def test_filter_unknown_license(
    client: ServerStub, query_exec_args: QueryExecutionArgs, result_streaming_args: ResultStreamingArgs
):
    result_streaming_args.job_id = (
        f"4_{query_exec_args.mixture.chunk_size}_{query_exec_args.dp_groups}"
        + f"_{query_exec_args.nodes_per_group}_{query_exec_args.num_workers}_{result_streaming_args.chunk_reading_degree_of_parallelism}"
        + f"_{result_streaming_args.chunk_reading_window_size}_{result_streaming_args.chunk_reading_mixture_type}"
        + f"_{result_streaming_args.tunnel_via_server}"
    )
    query = Query.for_job(result_streaming_args.job_id).select(("license", "==", "All rights reserved."))
    assert client.execute_query(query, query_exec_args)
    assert len(list(client.stream_results(result_streaming_args))) == 0, "Got results back for expected empty results."


def test_filter_license_and_html(
    client: ServerStub, query_exec_args: QueryExecutionArgs, result_streaming_args: ResultStreamingArgs
):
    result_streaming_args.job_id = (
        f"5_{query_exec_args.mixture.chunk_size}_{query_exec_args.dp_groups}"
        + f"_{query_exec_args.nodes_per_group}_{query_exec_args.num_workers}_{result_streaming_args.chunk_reading_degree_of_parallelism}"
        + f"_{result_streaming_args.chunk_reading_window_size}_{result_streaming_args.chunk_reading_mixture_type}"
        + f"_{result_streaming_args.tunnel_via_server}"
    )
    query = (
        Query.for_job(result_streaming_args.job_id).select(("language", "==", "HTML")).select(("license", "==", "CC"))
    )
    assert client.execute_query(query, query_exec_args)
    result_samples = []

    for sample in client.stream_results(result_streaming_args):
        result_samples.append(sample)
    expected_samples = EXPECTED_HTML_SAMPLES + EXPECTED_JS_SAMPLES // 2
    assert (
        len(result_samples) == expected_samples
    ), f"Got {len(result_samples)} samples instead of the expected {expected_samples}!"
    for _, _, sample in result_samples:
        assert 0 <= int(sample) < expected_samples, f"Sample {sample} should not appear"


def test_reproducibility(
    client: ServerStub, query_exec_args: QueryExecutionArgs, result_streaming_args: ResultStreamingArgs
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
            + f"_{result_streaming_args.chunk_reading_window_size}_{result_streaming_args.chunk_reading_mixture_type}"
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


def test_mixture_schedule(client: ServerStub):
    job_id = "server_feedback_test"
    query = Query.for_job(job_id).select(None)

    chunk_size = 10
    mixture_schedule = MixtureSchedule(
        chunk_size,
        [
            ScheduleEntry(
                0, StaticMixture(chunk_size, {MixtureKey({"language": ["JavaScript"], "license": ["CC"]}): 1.0})
            ),
            ScheduleEntry(100, StaticMixture(chunk_size, {MixtureKey({"language": ["HTML"]}): 1.0})),
            ScheduleEntry(200, StaticMixture(chunk_size, {MixtureKey({"language": ["JavaScript"]}): 1.0})),
        ],
    )

    query_execution_args = QueryExecutionArgs(mixture=mixture_schedule)
    result_streaming_args = ResultStreamingArgs(job_id)
    assert client.execute_query(query, query_execution_args)
    logger.info(f"Executed query for job {job_id} for mixture schedule.")

    result_samples = []

    for _, _, sample in client.stream_results(result_streaming_args):
        result_samples.append(sample)
        assert int(sample) % 2 == 0, f"Sample {sample} should not appear for JavaScript"

    assert len(result_samples) == EXPECTED_JS_SAMPLES // 2, f"got {len(result_samples)} != {EXPECTED_JS_SAMPLES // 2}"

    client.process_feedback(job_id, ClientFeedback(100))

    result_samples = []
    for _, _, sample in client.stream_results(result_streaming_args):
        result_samples.append(sample)
        assert int(sample) % 2 == 1, f"Sample {sample} should not appear for HTML"

    assert len(result_samples) == EXPECTED_HTML_SAMPLES, f"got {len(result_samples)} != {EXPECTED_HTML_SAMPLES}"

    client.process_feedback(job_id, ClientFeedback(200))

    result_samples = []
    for _, _, sample in client.stream_results(result_streaming_args):
        result_samples.append(sample)
        assert int(sample) % 2 == 0, f"Sample {sample} should not appear for JavaScript"

    assert len(result_samples) == EXPECTED_JS_SAMPLES // 2, f"got {len(result_samples)} != {EXPECTED_JS_SAMPLES // 2}"

    logger.info("Successfully trained with schedule.")


def test_dynamic_mixture(client: MixteraClient):
    job_id = "client_dynamic_mixture_test"
    query = Query.for_job(job_id).select(None)

    chunk_size = 12  # makes it easy to have a 2:1 ratio

    mixture = DynamicMixture(
        chunk_size=chunk_size,
        initial_mixture=StaticMixture(
            chunk_size=chunk_size,
            mixture={
                MixtureKey({"language": ["HTML"]}): 0.5,
                MixtureKey({"language": ["JavaScript"]}): 0.5,
            },
        ),
        mixing_alg=SimpleAveragingAlgorithm(),
    )

    query_execution_args = QueryExecutionArgs(mixture=mixture)
    result_streaming_args = ResultStreamingArgs(job_id)

    assert client.execute_query(query, query_execution_args)
    logger.info(f"Executed query for job {job_id} for dynamic mixture.")

    result_iter = client.stream_results(result_streaming_args)

    # Parse first chunk: 50:50 data
    num_js = 0
    num_html = 0
    key_html = None
    key_js = None
    for _ in range(chunk_size):
        idx_in_chunk, key_id, sample = next(result_iter)
        assert idx_in_chunk < chunk_size
        if int(sample) % 2 == 0:
            num_js += 1
            if key_js is None:
                key_js = key_id
            else:
                assert key_js == key_id, f"Inconsistent JS key ID: {key_js} != {key_id}"
        else:
            num_html += 1
            if key_html is None:
                key_html = key_id
            else:
                assert key_html == key_id, f"Inconsistent HTML key ID: {key_html} != {key_id}"

    assert num_js == num_html, f"First chunk has unequal distribution despite 50:50 mixture: {num_js} != {num_html}"

    # 2. Report losses such that we should get a 2:1 ratio of JS to HTML via the SimpleAveragingAlgorithm
    losses = np.zeros(max(key_js, key_html) + 1, dtype=np.float32)
    counts = np.zeros_like(losses, dtype=np.int64)
    losses[key_js] = 2
    losses[key_html] = 1
    counts[key_js] = 1
    counts[key_html] = 1
    client.process_feedback(job_id, ClientFeedback(training_steps=1, losses=losses, counts=counts, mixture_id=0))

    # 3. Check for two chunks whether they fulfill the new mixture
    num_js = 0
    num_html = 0
    for _ in range(chunk_size):
        idx_in_chunk, key_id, sample = next(result_iter)
        assert idx_in_chunk < chunk_size
        if int(sample) % 2 == 0:
            num_js += 1
            assert key_js == key_id, f"Inconsistent JS key ID: {key_js} != {key_id}"
        else:
            num_html += 1
            assert key_html == key_id, f"Inconsistent HTML key ID: {key_html} != {key_id}"

    assert (
        num_js == 2 * num_html
    ), f"Second chunk has does not have expected distribution: js =  {num_js}, html = {num_html}"

    num_js = 0
    num_html = 0
    for _ in range(chunk_size):
        idx_in_chunk, key_id, sample = next(result_iter)
        assert idx_in_chunk < chunk_size
        if int(sample) % 2 == 0:
            num_js += 1
            assert key_js == key_id, f"Inconsistent JS key ID: {key_js} != {key_id}"
        else:
            num_html += 1
            assert key_html == key_id, f"Inconsistent HTML key ID: {key_html} != {key_id}"

    assert (
        num_js == 2 * num_html
    ), f"Third chunk has does not have expected distribution: js =  {num_js}, html = {num_html}"

    # 3. Report losses such that we should get a 2:1 ratio of HTML to JS via the SimpleAveragingAlgorithm
    losses = np.zeros_like(losses, dtype=np.float32)
    counts = np.zeros_like(losses, dtype=np.int64)
    losses[key_js] = 0
    # Note that the SimpleAveragingAlgorithm does _not reset_ its state (it stored 2,1 before, after this it's 2,4)
    losses[key_html] = 3
    counts[key_js] = 1
    counts[key_html] = 1
    client.process_feedback(job_id, ClientFeedback(training_steps=3, losses=losses, counts=counts, mixture_id=1))

    # 3. Check for one chunk whether it fulfills the new mixture
    num_js = 0
    num_html = 0
    for _ in range(chunk_size):
        idx_in_chunk, key_id, sample = next(result_iter)
        assert idx_in_chunk < chunk_size
        if int(sample) % 2 == 0:
            num_js += 1
            assert key_js == key_id, f"Inconsistent JS key ID: {key_js} != {key_id}"
        else:
            num_html += 1
            assert key_html == key_id, f"Inconsistent HTML key ID: {key_html} != {key_id}"

    assert (
        2 * num_js == num_html
    ), f"Fourth chunk has does not have expected distribution: js =  {num_js}, html = {num_html}"

    logger.info("Successfully finished dynamic mixture test..")


def test_check_dataset_exists(client: ServerStub):
    assert client.check_dataset_exists("ldc_integrationtest_dataset"), "Dataset does not exist!"


def test_list_datasets(client: ServerStub):
    assert "ldc_integrationtest_dataset" in client.list_datasets(), "Dataset not in list!"


def test_add_property(client: ServerStub):
    # TOOO(#177): Adding new properties is currently broken.
    pass
    # assert client.add_property("test_property", setup_func, calc_func, ExecutionMode.LOCAL, PropertyType.CATEGORICAL)


def test_remove_dataset(client: ServerStub):
    assert client.remove_dataset("ldc_integrationtest_dataset"), "Could not remove dataset!"
    assert not client.check_dataset_exists("ldc_integrationtest_dataset"), "Dataset still exists!"


def test_server(server_dir: Path) -> None:
    client = MixteraClient.from_remote("127.0.0.1", 6666)

    assert client.register_metadata_parser("TEST_PARSER", TestMetadataParser)
    assert client.register_dataset(
        "ldc_integrationtest_dataset", server_dir, JSONLDataset, parsing_func, "TEST_PARSER"
    ), "Could not register dataset!"

    test_check_dataset_exists(client)

    reader_degrees_of_parallelisms = [1, 4]
    per_window_mixtures = [False, True]
    window_sizes = [64, 128]

    for chunk_size in [100, 500]:
        for reader_degree_of_parallelism in reader_degrees_of_parallelisms:
            for per_window_mixture in per_window_mixtures:
                for window_size in window_sizes:
                    for tunnel in [False, True]:
                        query_exec_args = QueryExecutionArgs(mixture=ArbitraryMixture(chunk_size))
                        result_streaming_args = ResultStreamingArgs(
                            None,
                            tunnel_via_server=tunnel,
                            chunk_reading_degree_of_parallelism=reader_degree_of_parallelism,
                            chunk_reading_mixture_type=per_window_mixture,
                            chunk_reading_window_size=window_size,
                        )
                        test_rdc_chunksize_tunnel(client, query_exec_args, result_streaming_args)

    test_mixture_schedule(client)
    test_dynamic_mixture(client)
    test_list_datasets(client)
    test_add_property(client)
    test_remove_dataset(client)

    print("Successfully ran server tests!")


def test_rdc_chunksize_tunnel(
    client: ServerStub, query_exec_args: QueryExecutionArgs, result_streaming_args: ResultStreamingArgs
):
    test_filter_javascript(client, query_exec_args, result_streaming_args)
    test_filter_html(client, query_exec_args, result_streaming_args)
    test_filter_both(client, query_exec_args, result_streaming_args)
    test_filter_license(client, query_exec_args, result_streaming_args)
    test_filter_unknown_license(client, query_exec_args, result_streaming_args)
    test_filter_license_and_html(client, query_exec_args, result_streaming_args)
    test_reproducibility(client, query_exec_args, result_streaming_args)


def main() -> None:
    server_dir = Path(sys.argv[1])

    setup_test_dataset(server_dir, TEST_SERVER_INSTANCE_COUNT, TEST_SERVER_FILE_COUNT, TEST_SERVER_FRACTION_MULTIPLIER)
    test_server(server_dir)


if __name__ == "__main__":
    main()
