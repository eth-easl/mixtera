import tempfile
import time
from pathlib import Path
from typing import Any

from integrationtests.utils import TestMetadataParser, write_jsonl_ensemble, write_single_jsonl
from mixtera.core.client import ChunkReaderType, MixteraClient
from mixtera.core.datacollection.datasets import JSONLDataset
from mixtera.core.query import Query


def parsing_func(sample):
    import json

    return json.loads(sample)["text"]


def test_filter_javascript(
    client: MixteraClient,
    chunk_size: int,
    chunk_reader_type: ChunkReaderType = ChunkReaderType.NON_PARALLEL,
    **chunk_reader_args: Any,
) -> None:
    job_id = str(round(time.time() * 1000))
    query = Query.for_job(job_id).select(("language", "==", "JavaScript"))
    client.execute_query(query, chunk_size)
    result_samples = []
    for sample in client.stream_results(job_id, False, reader_type=chunk_reader_type, **chunk_reader_args):
        result_samples.append(sample)

    assert len(result_samples) == 2500, f"Got {len(result_samples)} samples instead of the expected 500!"
    for sample in result_samples:
        assert int(sample) % 2 == 0, f"Sample {sample} should not appear for JavaScript"


def test_filter_html(client: MixteraClient, chunk_size: int):
    job_id = str(round(time.time() * 1000))
    query = Query.for_job(job_id).select(("language", "==", "HTML"))
    client.execute_query(query, chunk_size)
    result_samples = []

    for sample in client.stream_results(job_id, False):
        result_samples.append(sample)

    assert len(result_samples) == 500, f"Got {len(result_samples)} samples instead of the expected 500!"
    for sample in result_samples:
        assert int(sample) % 2 == 1, f"Sample {sample} should not appear for HTML"


def test_filter_both(client: MixteraClient, chunk_size: int):
    job_id = str(round(time.time() * 1000))
    query = (
        Query.for_job(job_id)
        .select(("language", "==", "HTML"))
        .union(Query.for_job(job_id).select(("language", "==", "JavaScript")))
    )
    client.execute_query(query, chunk_size)
    result_samples = []

    for sample in client.stream_results(job_id, False):
        result_samples.append(sample)

    assert len(result_samples) == 1000, f"Got {len(result_samples)} samples instead of the expected 1000!"
    for sample in result_samples:
        assert 0 <= int(sample) < 1000, f"Sample {sample} should not appear"


def test_filter_license(client: MixteraClient, chunk_size: int):
    job_id = str(round(time.time() * 1000))
    query = Query.for_job(job_id).select(("license", "==", "CC"))
    client.execute_query(query, chunk_size)
    result_samples = []

    for sample in client.stream_results(job_id, False):
        result_samples.append(sample)

    assert len(result_samples) == 1000, f"Got {len(result_samples)} samples instead of the expected 1000!"
    for sample in result_samples:
        assert 0 <= int(sample) < 1000, f"Sample {sample} should not appear"


def test_filter_unknown_license(client: MixteraClient, chunk_size: int):
    job_id = str(round(time.time() * 1000))
    query = Query.for_job(job_id).select(("license", "==", "All rights reserved."))
    client.execute_query(query, chunk_size)
    assert len(list(client.stream_results(job_id, False))) == 0, "Got results back for expected empty results."


def test_filter_license_and_html(client: MixteraClient, chunk_size: int):
    # TODO(41): This test currently tests unexpected behavior - we want to deduplicate!
    job_id = str(round(time.time() * 1000))
    query = (
        Query.for_job(job_id)
        .select(("language", "==", "HTML"))
        .union(Query.for_job(job_id).select(("license", "==", "CC")))
    )
    client.execute_query(query, chunk_size)
    result_samples = []

    for sample in client.stream_results(job_id, False):
        result_samples.append(sample)

    assert len(result_samples) == 1000, f"Got {len(result_samples)} samples instead of the expected 1000!"
    for sample in result_samples:
        assert 0 <= int(sample) < 1000, f"Sample {sample} should not appear"


def test_client_chunksize(
    client: MixteraClient,
    chunk_size: int,
    chunk_reader_type: ChunkReaderType = ChunkReaderType.NON_PARALLEL,
    **chunk_reader_args: Any,
):
    test_filter_javascript(client, chunk_size, chunk_reader_type=chunk_reader_type, **chunk_reader_args)
    # test_filter_html(client, chunk_size)
    # test_filter_both(client, chunk_size)
    # test_filter_license(client, chunk_size)
    # test_filter_unknown_license(client, chunk_size)
    # test_filter_license_and_html(client, chunk_size)


def test_client(dir: Path) -> None:
    write_single_jsonl(dir / "testd.jsonl")
    client = MixteraClient.from_directory(dir)
    client.register_metadata_parser("TEST_PARSER", TestMetadataParser)
    client.register_dataset(
        "client_integrationtest_dataset", dir / "testd.jsonl", JSONLDataset, parsing_func, "TEST_PARSER"
    )

    for chunk_size in [1, 3, 250, 500, 750, 1000, 2000]:
        test_client_chunksize(client, chunk_size)

    print("Successfully ran client test!")


def test_chunk_readers(dir: Path) -> None:
    write_jsonl_ensemble(dir)
    client = MixteraClient.from_directory(dir)
    client.register_metadata_parser("TEST_PARSER", TestMetadataParser)
    client.register_dataset("client_integrationtest_dataset", dir, JSONLDataset, parsing_func, "TEST_PARSER")

    reader_types = [ChunkReaderType.NON_PARALLEL] * 2 + [ChunkReaderType.PARALLEL] * 2
    base_parallel_params = {
        "reader_count": 4,
        "per_window_mixture": False,
    }
    special_params = [{"ensure_mixture": False}, {"ensure_mixture": True}, base_parallel_params,
                      base_parallel_params | {"per_window_mixture": True},]

    for chunk_reader_type, chunk_reader_args in zip(reader_types, special_params):
        test_client_chunksize(
            client, chunk_size=250, chunk_reader_type=chunk_reader_type, **chunk_reader_args
        )


def main() -> None:
    # with tempfile.TemporaryDirectory() as directory:
    #     test_client(Path(directory))

    # Testing different types of readers
    with tempfile.TemporaryDirectory() as directory:
        test_chunk_readers(Path(directory))


if __name__ == "__main__":
    main()
