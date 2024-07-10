import sys
import time
from pathlib import Path
from typing import Any

from integrationtests.utils import TestMetadataParser, calc_func, setup_func, write_jsonl
from mixtera.core.client import MixteraClient
from mixtera.core.client.server import ServerStub
from mixtera.core.datacollection.datasets import JSONLDataset
from mixtera.core.datacollection.property_type import PropertyType
from mixtera.core.processing.execution_mode import ExecutionMode
from mixtera.core.query import ArbitraryMixture, Mixture, Query


def parsing_func(sample):
    import json

    return json.loads(sample)["text"]


def test_filter_javascript(client: ServerStub, mixture: Mixture, tunnel: bool):
    job_id = str(round(time.time() * 1000))
    query = Query.for_job(job_id).select(("language", "==", "JavaScript"))
    assert client.execute_query(query, mixture)
    result_samples = []

    for sample in client.stream_results(job_id, tunnel_via_server=tunnel):
        result_samples.append(sample)

    assert len(result_samples) == 500, f"Got {len(result_samples)} samples instead of the expected 500!"
    for sample in result_samples:
        assert int(sample) % 2 == 0, f"Sample {sample} should not appear for JavaScript"


def test_filter_html(client: ServerStub, mixture: Mixture, tunnel: bool):
    job_id = str(round(time.time() * 1000))
    query = Query.for_job(job_id).select(("language", "==", "HTML"))
    assert client.execute_query(query, mixture)
    result_samples = []

    for sample in client.stream_results(job_id, tunnel_via_server=tunnel):
        result_samples.append(sample)

    assert len(result_samples) == 500, f"Got {len(result_samples)} samples instead of the expected 500!"
    for sample in result_samples:
        assert int(sample) % 2 == 1, f"Sample {sample} should not appear for HTML"


def test_filter_both(client: ServerStub, mixture: Mixture, tunnel: bool):
    job_id = str(round(time.time() * 1000))
    query = (
        Query.for_job(job_id)
        .select(("language", "==", "HTML"))
        .union(Query.for_job(job_id).select(("language", "==", "JavaScript")))
    )
    assert client.execute_query(query, mixture)
    result_samples = []

    for sample in client.stream_results(job_id, tunnel_via_server=tunnel):
        result_samples.append(sample)

    assert len(result_samples) == 1000, f"Got {len(result_samples)} samples instead of 1000!"
    for sample in result_samples:
        assert 0 <= int(sample) < 1000, f"Sample {sample} should not appear"


def test_filter_license(client: ServerStub, mixture: Mixture, tunnel: bool):
    job_id = str(round(time.time() * 1000))
    query = Query.for_job(job_id).select(("license", "==", "CC"))
    assert client.execute_query(query, mixture)
    result_samples = []

    for sample in client.stream_results(job_id, tunnel_via_server=tunnel):
        result_samples.append(sample)

    assert len(result_samples) == 1000, f"Got {len(result_samples)} samples instead of the expected 1000!"
    for sample in result_samples:
        assert 0 <= int(sample) < 1000, f"Sample {sample} should not appear"


def test_filter_unknown_license(client: ServerStub, mixture: Mixture, tunnel: bool):
    job_id = str(round(time.time() * 1000))
    query = Query.for_job(job_id).select(("license", "==", "All rights reserved."))
    assert client.execute_query(query, mixture)
    assert (
        len(list(client.stream_results(job_id, tunnel_via_server=tunnel))) == 0
    ), "Got results back for expected empty results."


def test_filter_license_and_html(client: ServerStub, mixture: Mixture, tunnel: bool):
    # TODO(#41): This test currently tests unexpected behavior - we want to deduplicate!
    job_id = str(round(time.time() * 1000))
    query = (
        Query.for_job(job_id)
        .select(("language", "==", "HTML"))
        .union(Query.for_job(job_id).select(("license", "==", "CC")))
    )
    assert client.execute_query(query, mixture)
    result_samples = []

    for sample in client.stream_results(job_id, tunnel_via_server=tunnel):
        result_samples.append(sample)

    assert len(result_samples) == 1000, f"Got {len(result_samples)} samples instead of the expected 1000!"
    for sample in result_samples:
        assert 0 <= int(sample) < 1000, f"Sample {sample} should not appear"


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
        "ldc_integrationtest_dataset", server_dir / "testd.jsonl", JSONLDataset, parsing_func, "TEST_PARSER"
    ), "Could not register dataset!"

    test_check_dataset_exists(client)

    for chunk_size in [1, 3, 250, 500, 750, 1000, 2000]:
        for tunnel in [False, True]:
            test_rdc_chunksize_tunnel(client, ArbitraryMixture(chunk_size), tunnel)

    test_list_datasets(client)
    test_add_property(client)
    test_remove_dataset(client)

    print("Successfully ran server tests!")


def prep_server(server_dir: Path) -> None:
    print(f"Prepping directory {server_dir}.")
    write_jsonl(server_dir / "testd.jsonl")
    print("Directory prepped.")


def test_rdc_chunksize_tunnel(client: ServerStub, mixture: Mixture, tunnel: bool):
    test_filter_javascript(client, mixture, tunnel)
    test_filter_html(client, mixture, tunnel)
    test_filter_both(client, mixture, tunnel)
    test_filter_license(client, mixture, tunnel)
    test_filter_unknown_license(client, mixture, tunnel)
    test_filter_license_and_html(client, mixture, tunnel)


def main() -> None:
    server_dir = Path(sys.argv[1])

    prep_server(server_dir)
    test_server(server_dir)


if __name__ == "__main__":
    main()
