import sys
from pathlib import Path
from typing import Any

from integrationtests.utils import (
    REPRODUCIBILITY_ITERATIONS,
    TestMetadataParser,
    calc_func,
    get_expected_js_and_html_samples,
    setup_func,
    setup_test_dataset,
)
from mixtera.core.client import MixteraClient
from mixtera.core.client.mixtera_client import QueryExecutionArgs, ResultStreamingArgs
from mixtera.core.client.server import ServerStub
from mixtera.core.datacollection.datasets import JSONLDataset
from mixtera.core.datacollection.property_type import PropertyType
from mixtera.core.processing.execution_mode import ExecutionMode
from mixtera.core.query import ArbitraryMixture, MixtureKey, Query, StaticMixture

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
        + f"_{result_streaming_args.chunk_reading_window_size}_{result_streaming_args.chunk_reading_per_window_mixture}"
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
    for sample in result_samples:
        assert int(sample) % 2 == 0, f"Sample {sample} should not appear for JavaScript"


def test_filter_html(
    client: ServerStub, query_exec_args: QueryExecutionArgs, result_streaming_args: ResultStreamingArgs
):
    result_streaming_args.job_id = (
        f"1_{query_exec_args.mixture.chunk_size}_{query_exec_args.dp_groups}"
        + f"_{query_exec_args.nodes_per_group}_{query_exec_args.num_workers}_{result_streaming_args.chunk_reading_degree_of_parallelism}"
        + f"_{result_streaming_args.chunk_reading_window_size}_{result_streaming_args.chunk_reading_per_window_mixture}"
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
    for sample in result_samples:
        assert int(sample) % 2 == 1, f"Sample {sample} should not appear for HTML"


def test_filter_both(
    client: ServerStub, query_exec_args: QueryExecutionArgs, result_streaming_args: ResultStreamingArgs
):
    result_streaming_args.job_id = (
        f"2_{query_exec_args.mixture.chunk_size}_{query_exec_args.dp_groups}"
        + f"_{query_exec_args.nodes_per_group}_{query_exec_args.num_workers}_{result_streaming_args.chunk_reading_degree_of_parallelism}"
        + f"_{result_streaming_args.chunk_reading_window_size}_{result_streaming_args.chunk_reading_per_window_mixture}"
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
    for sample in result_samples:
        assert 0 <= int(sample) < TEST_SERVER_INSTANCE_COUNT, f"Sample {sample} should not appear"


def test_filter_license(
    client: ServerStub, query_exec_args: QueryExecutionArgs, result_streaming_args: ResultStreamingArgs
):
    result_streaming_args.job_id = (
        f"3_{query_exec_args.mixture.chunk_size}_{query_exec_args.dp_groups}"
        + f"_{query_exec_args.nodes_per_group}_{query_exec_args.num_workers}_{result_streaming_args.chunk_reading_degree_of_parallelism}"
        + f"_{result_streaming_args.chunk_reading_window_size}_{result_streaming_args.chunk_reading_per_window_mixture}"
        + f"_{result_streaming_args.tunnel_via_server}"
    )
    query = Query.for_job(result_streaming_args.job_id).select(("license", "==", "CC"))
    assert client.execute_query(query, query_exec_args)
    result_samples = []

    for sample in client.stream_results(result_streaming_args):
        result_samples.append(sample)

    assert (
        len(result_samples) == TEST_SERVER_INSTANCE_COUNT
    ), f"Got {len(result_samples)} samples instead of the expected {TEST_SERVER_INSTANCE_COUNT}!"
    for sample in result_samples:
        assert 0 <= int(sample) < TEST_SERVER_INSTANCE_COUNT, f"Sample {sample} should not appear"


def test_filter_unknown_license(
    client: ServerStub, query_exec_args: QueryExecutionArgs, result_streaming_args: ResultStreamingArgs
):
    result_streaming_args.job_id = (
        f"4_{query_exec_args.mixture.chunk_size}_{query_exec_args.dp_groups}"
        + f"_{query_exec_args.nodes_per_group}_{query_exec_args.num_workers}_{result_streaming_args.chunk_reading_degree_of_parallelism}"
        + f"_{result_streaming_args.chunk_reading_window_size}_{result_streaming_args.chunk_reading_per_window_mixture}"
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
        + f"_{result_streaming_args.chunk_reading_window_size}_{result_streaming_args.chunk_reading_per_window_mixture}"
        + f"_{result_streaming_args.tunnel_via_server}"
    )
    # TODO(#41): This test currently tests unexpected behavior - we want to deduplicate!
    query = (
        Query.for_job(result_streaming_args.job_id).select(("language", "==", "HTML")).select(("license", "==", "CC"))
    )
    assert client.execute_query(query, query_exec_args)
    result_samples = []

    for sample in client.stream_results(result_streaming_args):
        result_samples.append(sample)

    assert (
        len(result_samples) == TEST_SERVER_INSTANCE_COUNT
    ), f"Got {len(result_samples)} samples instead of the expected {TEST_SERVER_INSTANCE_COUNT}!"
    for sample in result_samples:
        assert 0 <= int(sample) < TEST_SERVER_INSTANCE_COUNT, f"Sample {sample} should not appear"


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


def test_check_dataset_exists(client: ServerStub):
    assert client.check_dataset_exists("ldc_integrationtest_dataset"), "Dataset does not exist!"


def test_list_datasets(client: ServerStub):
    assert "ldc_integrationtest_dataset" in client.list_datasets(), "Dataset not in list!"


def test_add_property(client: ServerStub):
    assert client.add_property("test_property", setup_func, calc_func, ExecutionMode.LOCAL, PropertyType.CATEGORICAL)


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
                            chunk_reading_per_window_mixture=per_window_mixture,
                            chunk_reading_window_size=window_size,
                        )
                        test_rdc_chunksize_tunnel(client, query_exec_args, result_streaming_args)

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
