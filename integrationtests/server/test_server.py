import sys
import time
from pathlib import Path
from typing import Any

from integrationtests.utils import (
    TestMetadataParser,
    calc_func,
    get_expected_js_and_html_samples,
    get_job_id,
    setup_func,
    setup_test_dataset,
)
from mixtera.core.client import MixteraClient
from mixtera.core.client.server import ServerStub
from mixtera.core.datacollection.datasets import JSONLDataset
from mixtera.core.datacollection.property_type import PropertyType
from mixtera.core.processing.execution_mode import ExecutionMode
from mixtera.core.query import ArbitraryMixture, Mixture, Query

TEST_SERVER_INSTANCE_COUNT = 1000
TEST_SERVER_FILE_COUNT = 5
TEST_SERVER_FRACTION_MULTIPLIER = 2

EXPECTED_JS_SAMPLES, EXPECTED_HTML_SAMPLES = get_expected_js_and_html_samples(
    TEST_SERVER_INSTANCE_COUNT, TEST_SERVER_FRACTION_MULTIPLIER
)


def parsing_func(sample):
    import json

    return json.loads(sample)["text"]


def test_filter_javascript(client: ServerStub, mixture: Mixture, tunnel: bool):
    job_id = get_job_id()
    query = Query.for_job(job_id).select(("language", "==", "JavaScript"))
    assert client.execute_query(query, mixture)
    result_samples = []

    for sample in client.stream_results(job_id, tunnel_via_server=tunnel):
        result_samples.append(sample)

    assert (
        len(result_samples) == EXPECTED_JS_SAMPLES
    ), f"Got {len(result_samples)} samples instead of the expected {EXPECTED_JS_SAMPLES}!"
    for sample in result_samples:
        assert int(sample) % 2 == 0, f"Sample {sample} should not appear for JavaScript"


def test_filter_html(client: ServerStub, mixture: Mixture, tunnel: bool):
    job_id = get_job_id()
    query = Query.for_job(job_id).select(("language", "==", "HTML"))
    assert client.execute_query(query, mixture)
    result_samples = []

    for sample in client.stream_results(job_id, tunnel_via_server=tunnel):
        result_samples.append(sample)

    assert (
        len(result_samples) == EXPECTED_HTML_SAMPLES
    ), f"Got {len(result_samples)} samples instead of the expected {EXPECTED_HTML_SAMPLES}!"
    for sample in result_samples:
        assert int(sample) % 2 == 1, f"Sample {sample} should not appear for HTML"


def test_filter_both(client: ServerStub, mixture: Mixture, tunnel: bool):
    job_id = get_job_id()
    query = (
        Query.for_job(job_id)
        .select(("language", "==", "HTML"))
        .union(Query.for_job(job_id).select(("language", "==", "JavaScript")))
    )
    assert client.execute_query(query, mixture)
    result_samples = []

    for sample in client.stream_results(job_id, tunnel_via_server=tunnel):
        result_samples.append(sample)

    assert (
        len(result_samples) == TEST_SERVER_INSTANCE_COUNT
    ), f"Got {len(result_samples)} samples instead of {TEST_SERVER_INSTANCE_COUNT}!"
    for sample in result_samples:
        assert 0 <= int(sample) < TEST_SERVER_INSTANCE_COUNT, f"Sample {sample} should not appear"


def test_filter_license(client: ServerStub, mixture: Mixture, tunnel: bool):
    job_id = get_job_id()
    query = Query.for_job(job_id).select(("license", "==", "CC"))
    assert client.execute_query(query, mixture)
    result_samples = []

    for sample in client.stream_results(job_id, tunnel_via_server=tunnel):
        result_samples.append(sample)

    assert (
        len(result_samples) == TEST_SERVER_INSTANCE_COUNT
    ), f"Got {len(result_samples)} samples instead of the expected {TEST_SERVER_INSTANCE_COUNT}!"
    for sample in result_samples:
        assert 0 <= int(sample) < TEST_SERVER_INSTANCE_COUNT, f"Sample {sample} should not appear"


def test_filter_unknown_license(client: ServerStub, mixture: Mixture, tunnel: bool):
    job_id = get_job_id()
    query = Query.for_job(job_id).select(("license", "==", "All rights reserved."))
    assert client.execute_query(query, mixture)
    assert (
        len(list(client.stream_results(job_id, tunnel_via_server=tunnel))) == 0
    ), "Got results back for expected empty results."


def test_filter_license_and_html(client: ServerStub, mixture: Mixture, tunnel: bool):
    # TODO(#41): This test currently tests unexpected behavior - we want to deduplicate!
    job_id = get_job_id()
    query = (
        Query.for_job(job_id)
        .select(("language", "==", "HTML"))
        .union(Query.for_job(job_id).select(("license", "==", "CC")))
    )
    assert client.execute_query(query, mixture)
    result_samples = []

    for sample in client.stream_results(job_id, tunnel_via_server=tunnel):
        result_samples.append(sample)

    assert (
        len(result_samples) == TEST_SERVER_INSTANCE_COUNT
    ), f"Got {len(result_samples)} samples instead of the expected {TEST_SERVER_INSTANCE_COUNT}!"
    for sample in result_samples:
        assert 0 <= int(sample) < TEST_SERVER_INSTANCE_COUNT, f"Sample {sample} should not appear"


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

    for chunk_size in [1, 3, 250, 500, 750, 1000, 2000]:
        for tunnel in [False, True]:
            test_rdc_chunksize_tunnel(client, ArbitraryMixture(chunk_size), tunnel)

    test_list_datasets(client)
    test_add_property(client)
    test_remove_dataset(client)

    print("Successfully ran server tests!")


def test_rdc_chunksize_tunnel(client: ServerStub, mixture: Mixture, tunnel: bool):
    test_filter_javascript(client, mixture, tunnel)
    test_filter_html(client, mixture, tunnel)
    test_filter_both(client, mixture, tunnel)
    test_filter_license(client, mixture, tunnel)
    test_filter_unknown_license(client, mixture, tunnel)
    test_filter_license_and_html(client, mixture, tunnel)


def main() -> None:
    server_dir = Path(sys.argv[1])

    setup_test_dataset(server_dir, TEST_SERVER_INSTANCE_COUNT, TEST_SERVER_FILE_COUNT, TEST_SERVER_FRACTION_MULTIPLIER)
    test_server(server_dir)


if __name__ == "__main__":
    main()
