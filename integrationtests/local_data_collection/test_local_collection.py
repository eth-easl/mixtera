import tempfile
import time
from pathlib import Path

from integrationtests.utils import TestMetadataParser, write_jsonl
from mixtera.core.datacollection import MixteraClient
from mixtera.core.datacollection.datasets import JSONLDataset
from mixtera.core.client.local import MixteraDataCollection
from mixtera.core.query import Query


def parsing_func(sample):
    import json

    return json.loads(sample)["text"]


def test_filter_javascript(ldc: MixteraDataCollection, chunk_size: int):
    training_id = str(round(time.time() * 1000))
    query = Query.for_training(training_id, 1).select(("language", "==", "JavaScript"))
    query_result = query.execute(ldc, chunk_size=chunk_size)
    result_samples = []

    for sample in ldc.stream_query_results(query_result):
        result_samples.append(sample)

    assert len(result_samples) == 500, f"Got {len(result_samples)} samples instead of the expected 500!"
    for sample in result_samples:
        assert int(sample) % 2 == 0, f"Sample {sample} should not appear for JavaScript"


def test_filter_html(ldc: MixteraDataCollection, chunk_size: int):
    training_id = str(round(time.time() * 1000))
    query = Query.for_training(training_id, 1).select(("language", "==", "HTML"))
    query_result = query.execute(ldc, chunk_size=chunk_size)
    result_samples = []

    for sample in ldc.stream_query_results(query_result):
        result_samples.append(sample)

    assert len(result_samples) == 500, f"Got {len(result_samples)} samples instead of the expected 500!"
    for sample in result_samples:
        assert int(sample) % 2 == 1, f"Sample {sample} should not appear for HTML"


def test_filter_both(ldc: MixteraDataCollection, chunk_size: int):
    training_id = str(round(time.time() * 1000))
    query = (
        Query.for_training(training_id, 1)
        .select(("language", "==", "HTML"))
        .union(Query.for_training(training_id, 1).select(("language", "==", "JavaScript")))
    )
    query_result = query.execute(ldc, chunk_size=chunk_size)
    result_samples = []

    for sample in ldc.stream_query_results(query_result):
        result_samples.append(sample)

    assert len(result_samples) == 1000, f"Got {len(result_samples)} samples instead of the expected 1000!"
    for sample in result_samples:
        assert 0 <= int(sample) < 1000, f"Sample {sample} should not appear"


def test_filter_license(ldc: MixteraDataCollection, chunk_size: int):
    training_id = str(round(time.time() * 1000))
    query = Query.for_training(training_id, 1).select(("license", "==", "CC"))
    query_result = query.execute(ldc, chunk_size=chunk_size)
    result_samples = []

    for sample in ldc.stream_query_results(query_result):
        result_samples.append(sample)

    assert len(result_samples) == 1000, f"Got {len(result_samples)} samples instead of the expected 1000!"
    for sample in result_samples:
        assert 0 <= int(sample) < 1000, f"Sample {sample} should not appear"


def test_filter_unknown_license(ldc: MixteraDataCollection, chunk_size: int):
    training_id = str(round(time.time() * 1000))
    query = Query.for_training(training_id, 1).select(("license", "==", "All rights reserved."))
    query_result = query.execute(ldc, chunk_size=chunk_size)
    assert len(list(ldc.stream_query_results(query_result))) == 0, "Got results back for expected empty results."


def test_filter_license_and_html(ldc: MixteraDataCollection, chunk_size: int):
    # TODO(41): This test currently tests unexpected behavior - we want to deduplicate!
    training_id = str(round(time.time() * 1000))
    query = (
        Query.for_training(training_id, 1)
        .select(("language", "==", "HTML"))
        .union(Query.for_training(training_id, 1).select(("license", "==", "CC")))
    )
    query_result = query.execute(ldc, chunk_size=chunk_size)
    result_samples = []

    for sample in ldc.stream_query_results(query_result):
        result_samples.append(sample)

    assert len(result_samples) == 1500, f"Got {len(result_samples)} samples instead of the expected 1500!"
    for sample in result_samples:
        assert 0 <= int(sample) < 1000, f"Sample {sample} should not appear"


def test_ldc_chunksize(ldc: MixteraDataCollection, chunk_size: int):
    test_filter_javascript(ldc, chunk_size)
    test_filter_html(ldc, chunk_size)
    test_filter_both(ldc, chunk_size)
    test_filter_license(ldc, chunk_size)
    test_filter_unknown_license(ldc, chunk_size)
    test_filter_license_and_html(ldc, chunk_size)


def test_ldc(dir: Path) -> None:
    write_jsonl(dir / "testd.jsonl")
    ldc = MixteraClient.from_directory(dir)
    # TODO(create issue): We might want to offer this on the MDC?
    ldc._metadata_factory.add_parser("TEST_PARSER", TestMetadataParser)
    ldc.register_dataset("ldc_integrationtest_dataset", dir / "testd.jsonl", JSONLDataset, parsing_func, "TEST_PARSER")

    for chunk_size in [1, 3, 250, 500, 750, 1000, 2000]:
        test_ldc_chunksize(ldc, chunk_size)

    print("Successfully ran LDC test!")


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        test_ldc(Path(directory))


if __name__ == "__main__":
    main()
