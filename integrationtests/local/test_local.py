import tempfile
import multiprocessing as mp
from pathlib import Path

from integrationtests.utils import TestMetadataParser, write_jsonl
from mixtera.core.client import MixteraClient
from mixtera.core.datacollection.datasets import JSONLDataset
from mixtera.core.query import ArbitraryMixture, Mixture, Query


def parsing_func(sample):
    import json

    return json.loads(sample)["text"]


def test_filter_javascript(client: MixteraClient, mixture: Mixture):
    job_id = str(int(1e4 + mixture.chunk_size))
    query = Query.for_job(job_id).select(("language", "==", "JavaScript"))
    client.execute_query(query, mixture, 1, 1, 1)
    result_samples = []
    for sample in client.stream_results(job_id, 0, 0, 0, False):
        result_samples.append(sample)

    assert len(result_samples) == 500, f"Got {len(result_samples)} samples instead of the expected 500!"
    for sample in result_samples:
        assert int(sample) % 2 == 0, f"Sample {sample} should not appear for JavaScript"


def test_filter_html(client: MixteraClient, mixture: Mixture):
    job_id = str(int(2e4 + mixture.chunk_size))
    query = Query.for_job(job_id).select(("language", "==", "HTML"))
    client.execute_query(query, mixture, 1, 1, 1)
    result_samples = []

    for sample in client.stream_results(job_id, 0, 0, 0, False):
        result_samples.append(sample)

    assert len(result_samples) == 500, f"Got {len(result_samples)} samples instead of the expected 500!"
    for sample in result_samples:
        assert int(sample) % 2 == 1, f"Sample {sample} should not appear for HTML"


def test_filter_both(client: MixteraClient, mixture: Mixture):
    job_id = str(int(3e4 + mixture.chunk_size))
    query = (
        Query.for_job(job_id)
        .select(("language", "==", "HTML"))
        .union(Query.for_job(job_id).select(("language", "==", "JavaScript")))
    )
    client.execute_query(query, mixture, 1, 1, 1)
    result_samples = []

    for sample in client.stream_results(job_id, 0, 0, 0, False):
        result_samples.append(sample)

    assert len(result_samples) == 1000, f"Got {len(result_samples)} samples instead of the expected 1000!"
    for sample in result_samples:
        assert 0 <= int(sample) < 1000, f"Sample {sample} should not appear"


def test_filter_license(client: MixteraClient, mixture: Mixture):
    job_id = str(int(4e4 + mixture.chunk_size))
    query = Query.for_job(job_id).select(("license", "==", "CC"))
    client.execute_query(query, mixture, 1, 1, 1)
    result_samples = []

    for sample in client.stream_results(job_id, 0, 0, 0, False):
        result_samples.append(sample)

    assert len(result_samples) == 1000, f"Got {len(result_samples)} samples instead of the expected 1000!"
    for sample in result_samples:
        assert 0 <= int(sample) < 1000, f"Sample {sample} should not appear"


def test_filter_unknown_license(client: MixteraClient, mixture: Mixture):
    job_id = str(int(5e4 + mixture.chunk_size))
    query = Query.for_job(job_id).select(("license", "==", "All rights reserved."))
    client.execute_query(query, mixture, 1, 1, 1)
    assert len(list(client.stream_results(job_id, 0, 0, 0, False))) == 0, "Got results back for expected empty results."


def test_filter_license_and_html(client: MixteraClient, mixture: Mixture):
    job_id = str(int(6e4 + mixture.chunk_size))
    query = (
        Query.for_job(job_id)
        .select(("language", "==", "HTML"))
        .union(Query.for_job(job_id).select(("license", "==", "CC")))
    )
    client.execute_query(query, mixture, 1, 1, 1)
    result_samples = []

    for sample in client.stream_results(job_id, 0, 0, 0, False):
        result_samples.append(sample)

    assert len(result_samples) == 1000, f"Got {len(result_samples)} samples instead of the expected 1000!"
    for sample in result_samples:
        assert 0 <= int(sample) < 1000, f"Sample {sample} should not appear"


def test_client_chunksize(client: MixteraClient, mixture: Mixture):
    test_filter_javascript(client, mixture)
    test_filter_html(client, mixture)
    test_filter_both(client, mixture)
    test_filter_license(client, mixture)
    test_filter_unknown_license(client, mixture)
    test_filter_license_and_html(client, mixture)


def test_client(dir: Path) -> None:
    write_jsonl(dir / "testd.jsonl")
    client = MixteraClient.from_directory(dir)
    client.register_metadata_parser("TEST_PARSER", TestMetadataParser)
    client.register_dataset(
        "client_integrationtest_dataset", dir / "testd.jsonl", JSONLDataset, parsing_func, "TEST_PARSER"
    )

    for chunk_size in [1, 3, 250, 500, 750, 1000, 2000]:
        test_client_chunksize(client, ArbitraryMixture(chunk_size))

    print("Successfully ran client test!")


def main() -> None:
    print(f"Running tests with {mp.get_start_method()} start method.")
    with tempfile.TemporaryDirectory() as directory:
        test_client(Path(directory))


if __name__ == "__main__":
    main()
