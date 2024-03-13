import tempfile
import time
from pathlib import Path
from typing import Any, Optional

from mixtera.core.datacollection import MixteraDataCollection
from mixtera.core.datacollection.datasets import JSONLDataset
from mixtera.core.datacollection.index.parser import MetadataParser
from mixtera.core.datacollection.local import LocalDataCollection
from mixtera.core.query import Query


def parsing_func(sample):
    import json

    return json.loads(sample)["text"]


def write_jsonl(path: Path) -> None:
    data = ""
    for i in range(1000):
        data = (
            data
            + '{ "text": "'
            + str(i)
            + '", "meta": { "language": "'
            + ("JavaScript" if i % 2 == 0 else "HTML")
            + '", "license": "CC"}}\n'
        )

    with open(path, "w") as text_file:
        text_file.write(data)


class TestMetadataParser(MetadataParser):
    def parse(self, line_number: int, payload: Any, **kwargs: Optional[dict[Any, Any]]) -> None:
        metadata = payload["meta"]
        self._index.append_entry("language", metadata["language"], self.dataset_id, self.file_id, line_number)
        self._index.append_entry("license", metadata["license"], self.dataset_id, self.file_id, line_number)


def test_filter_javascript(ldc: LocalDataCollection, chunk_size: int):
    training_id = str(round(time.time() * 1000))
    query = Query.for_training(training_id, 1).select(("language", "==", "JavaScript"))
    query_result = query.execute(ldc, chunk_size=chunk_size)
    result_samples = []

    for sample in ldc.stream_query_results(query_result):
        result_samples.append(sample)

    assert len(result_samples) == 500, f"Got {len(result_samples)} samples instead of the expected 500!"
    for sample in result_samples:
        assert int(sample) % 2 == 0, f"Sample {sample} should not appear for JavaScript"


def test_filter_html(ldc: LocalDataCollection, chunk_size: int):
    training_id = str(round(time.time() * 1000))
    query = Query.for_training(training_id, 1).select(("language", "==", "HTML"))
    query_result = query.execute(ldc, chunk_size=chunk_size)
    result_samples = []

    for sample in ldc.stream_query_results(query_result):
        result_samples.append(sample)

    assert len(result_samples) == 500, f"Got {len(result_samples)} samples instead of the expected 500!"
    for sample in result_samples:
        assert int(sample) % 2 == 1, f"Sample {sample} should not appear for HTML"


def test_filter_both(ldc: LocalDataCollection, chunk_size: int):
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

    assert len(result_samples) == 1000, f"Got {len(result_samples)} samples instead of the expected 500!"
    for sample in result_samples:
        assert 0 <= int(sample) < 1000, f"Sample {sample} should not appear"


def test_filter_license(ldc: LocalDataCollection, chunk_size: int):
    pass


def test_filter_license_and_html(ldc: LocalDataCollection, chunk_size: int):
    pass


def test_filter_unknown_license(ldc: LocalDataCollection, chunk_size: int):
    pass


def test_ldc_chunksize(ldc: LocalDataCollection, chunk_size: int):
    test_filter_javascript(ldc, chunk_size)
    test_filter_html(ldc, chunk_size)
    test_filter_both(ldc, chunk_size)


def test_ldc(dir: Path) -> None:
    write_jsonl(dir / "testd.jsonl")
    ldc = MixteraDataCollection.from_directory(dir)
    # TODO(create issue): We might want to offer this on the MDC?
    ldc._metadata_factory.add_parser("TEST_PARSER", TestMetadataParser)
    ldc.register_dataset("ldc_integrationtest_dataset", dir / "testd.jsonl", JSONLDataset, parsing_func, "TEST_PARSER")

    for chunk_size in [1, 250, 500, 750, 1000, 2000]:
        test_ldc_chunksize(ldc, chunk_size)


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        test_ldc(Path(directory))


if __name__ == "__main__":
    main()
