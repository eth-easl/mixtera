import tempfile
import time
from pathlib import Path
from typing import Any

from integrationtests.utils import TestMetadataParser, setup_test_dataset, get_expected_js_and_html_samples
from mixtera.core.client import ChunkReaderType, MixteraClient
from mixtera.core.datacollection.datasets import JSONLDataset
from mixtera.core.query import ArbitraryMixture, Mixture, Query


TEST_LOCAL_INSTANCE_COUNT = 1000
TEST_LOCAL_FILE_COUNT = 5
TEST_LOCAL_FRACTION_MULTIPLIER = 2

EXPECTED_JS_SAMPLES, EXPECTED_HTML_SAMPLES = get_expected_js_and_html_samples(TEST_LOCAL_INSTANCE_COUNT, TEST_LOCAL_FRACTION_MULTIPLIER)


def parsing_func(sample):
    import json

    return json.loads(sample)["text"]


def test_filter_javascript(
    client: MixteraClient,
    mixture: Mixture,
    chunk_reader_type: ChunkReaderType = ChunkReaderType.NON_PARALLEL,
    **chunk_reader_args: Any,
) -> None:
    job_id = str(int(1e4 + mixture.chunk_size))
    query = Query.for_job(job_id).select(("language", "==", "JavaScript"))
    client.execute_query(query, mixture)
    result_samples = []
    for sample in client.stream_results(job_id, False, reader_type=chunk_reader_type, **chunk_reader_args):
        result_samples.append(sample)

    assert len(result_samples) == EXPECTED_JS_SAMPLES, f"Got {len(result_samples)} samples instead of the expected {EXPECTED_JS_SAMPLES}!"
    for sample in result_samples:
        assert int(sample) % TEST_LOCAL_FRACTION_MULTIPLIER == 0, f"Sample {sample} should not appear for JavaScript"


def test_filter_html(client: MixteraClient, mixture: Mixture):
    job_id = str(int(2e4 + mixture.chunk_size))
    query = Query.for_job(job_id).select(("language", "==", "HTML"))
    client.execute_query(query, mixture)
    result_samples = []

    for sample in client.stream_results(job_id, False):
        result_samples.append(sample)

    assert len(result_samples) == EXPECTED_HTML_SAMPLES, f"Got {len(result_samples)} samples instead of the expected {EXPECTED_HTML_SAMPLES}!"
    for sample in result_samples:
        assert int(sample) % TEST_LOCAL_FRACTION_MULTIPLIER == 1, f"Sample {sample} should not appear for HTML"


def test_filter_both(client: MixteraClient, mixture: Mixture):
    job_id = str(int(3e4 + mixture.chunk_size))
    query = (
        Query.for_job(job_id)
        .select(("language", "==", "HTML"))
        .union(Query.for_job(job_id).select(("language", "==", "JavaScript")))
    )
    client.execute_query(query, mixture)
    result_samples = []

    for sample in client.stream_results(job_id, False):
        result_samples.append(sample)

    assert len(result_samples) == TEST_LOCAL_INSTANCE_COUNT, f"Got {len(result_samples)} samples instead of the expected {TEST_LOCAL_INSTANCE_COUNT}!"
    for sample in result_samples:
        assert 0 <= int(sample) < TEST_LOCAL_INSTANCE_COUNT, f"Sample {sample} should not appear"


def test_filter_license(client: MixteraClient, mixture: Mixture):
    job_id = str(int(4e4 + mixture.chunk_size))
    query = Query.for_job(job_id).select(("license", "==", "CC"))
    client.execute_query(query, mixture)
    result_samples = []

    for sample in client.stream_results(job_id, False):
        result_samples.append(sample)

    assert len(result_samples) == TEST_LOCAL_INSTANCE_COUNT, f"Got {len(result_samples)} samples instead of the expected {TEST_LOCAL_INSTANCE_COUNT}!"
    for sample in result_samples:
        assert 0 <= int(sample) < TEST_LOCAL_INSTANCE_COUNT, f"Sample {sample} should not appear"


def test_filter_unknown_license(client: MixteraClient, mixture: Mixture):
    job_id = str(int(5e4 + mixture.chunk_size))
    query = Query.for_job(job_id).select(("license", "==", "All rights reserved."))
    client.execute_query(query, mixture)
    assert len(list(client.stream_results(job_id, False))) == 0, "Got results back for expected empty results."


def test_filter_license_and_html(client: MixteraClient, mixture: Mixture):
    job_id = str(int(6e4 + mixture.chunk_size))
    query = (
        Query.for_job(job_id)
        .select(("language", "==", "HTML"))
        .union(Query.for_job(job_id).select(("license", "==", "CC")))
    )
    client.execute_query(query, mixture)
    result_samples = []

    for sample in client.stream_results(job_id, False):
        result_samples.append(sample)

    assert len(result_samples) == TEST_LOCAL_INSTANCE_COUNT, f"Got {len(result_samples)} samples instead of the expected {TEST_LOCAL_INSTANCE_COUNT}!"
    for sample in result_samples:
        assert 0 <= int(sample) < TEST_LOCAL_INSTANCE_COUNT, f"Sample {sample} should not appear"


def test_result_order(client: MixteraClient, mixture: Mixture, chunk_reader_type: ChunkReaderType = ChunkReaderType.NON_PARALLEL, **chunk_reader_args: Any
    ) -> None:
    job_id = str(int(1e4 + mixture.chunk_size))
    query = Query.for_job(job_id).select(("language", "==", "JavaScript"))
    client.execute_query(query, mixture)
    result_samples = []
    for sample in client.stream_results(job_id, False, reader_type=chunk_reader_type, **chunk_reader_args):
        result_samples.append(sample)

    job_id_2 = str(int(1e4 + mixture.chunk_size))
    query_2 = Query.for_job(job_id_2).select(("language", "==", "JavaScript"))
    client.execute_query(query_2, mixture)

    result_samples_2 = []
    for sample in client.stream_results(job_id_2, False, reader_type=chunk_reader_type, **chunk_reader_args):
        result_samples_2.append(sample)

    assert result_samples == result_samples_2, "Results are not the same!"


def test_client_chunksize(
    client: MixteraClient,
    mixture: Mixture,
    chunk_reader_type: ChunkReaderType = ChunkReaderType.NON_PARALLEL,
    **chunk_reader_args: Any,
):
    test_filter_javascript(client, mixture, chunk_reader_type=chunk_reader_type, **chunk_reader_args)
    test_filter_html(client, mixture)
    test_filter_both(client, mixture)
    test_filter_license(client, mixture)
    test_filter_unknown_license(client, mixture)
    test_filter_license_and_html(client, mixture)
    test_result_order(client, mixture, chunk_reader_type=chunk_reader_type, **chunk_reader_args)


def test_direct_client(dir: Path) -> None:
    setup_test_dataset(dir, TEST_LOCAL_INSTANCE_COUNT, TEST_LOCAL_FILE_COUNT, TEST_LOCAL_FRACTION_MULTIPLIER)
    client = MixteraClient.from_directory(dir)
    client.register_metadata_parser("TEST_PARSER", TestMetadataParser)
    client.register_dataset(
        "client_integrationtest_dataset", dir, JSONLDataset, parsing_func, "TEST_PARSER"
    )

    for chunk_size in [1, 3, 250, 500, 750, 1000, 2000]:
        test_client_chunksize(client, ArbitraryMixture(chunk_size))

    print("Successfully ran client test!")

    client.remove_dataset("client_integrationtest_dataset")


def test_chunk_readers(dir: Path) -> None:
    setup_test_dataset(dir, TEST_LOCAL_INSTANCE_COUNT, TEST_LOCAL_FILE_COUNT, TEST_LOCAL_FRACTION_MULTIPLIER)
    client = MixteraClient.from_directory(dir)
    client.register_metadata_parser("TEST_PARSER", TestMetadataParser)
    client.register_dataset("client_integrationtest_dataset", dir, JSONLDataset, parsing_func, "TEST_PARSER")

    reader_types = [ChunkReaderType.NON_PARALLEL] * 2 + [ChunkReaderType.PARALLEL] * 2
    base_parallel_params = {
        "reader_count": 4,
        "per_window_mixture": False,
    }
    special_params = [
        {"ensure_mixture": False},
        {"ensure_mixture": True},
        base_parallel_params,
        base_parallel_params | {"per_window_mixture": True},
    ]

    for chunk_size in [1, 3, 250, 500, 750, 1000, 2000]:
        for chunk_reader_type, chunk_reader_args in zip(reader_types, special_params):
            test_client_chunksize(client, ArbitraryMixture(chunk_size), chunk_reader_type=chunk_reader_type, **chunk_reader_args)
    
    print("Successfully ran chunk reader tests!")

    client.remove_dataset("client_integrationtest_dataset")


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        test_direct_client(Path(directory))
        test_chunk_readers(Path(directory))


if __name__ == "__main__":
    main()
