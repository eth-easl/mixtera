from pathlib import Path
from typing import Any, Optional

from mixtera.core.datacollection.index.parser import MetadataParser


def write_single_jsonl(path: Path) -> None:
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


def write_jsonl_ensemble(
    path: Path, file_count: int = 100, instance_count: int = 100, fraction_multiplier: int = 4
) -> None:
    for file_number in range(file_count):
        data = ""
        for i in range(instance_count):
            data = (
                data
                + '{ "text": "'
                + str(i)
                + '", "meta": { "language": "'
                + ("JavaScript" if i % fraction_multiplier == 0 else "HTML")
                + '", "license": "CC"}}\n'
            )

        with open(path / f"data_{file_number}.jsonl", "w") as text_file:
            text_file.write(data)


class TestMetadataParser(MetadataParser):
    def parse(self, line_number: int, payload: Any, **kwargs: Optional[dict[Any, Any]]) -> None:
        metadata = payload["meta"]
        self._index.append_entry("language", metadata["language"], self.dataset_id, self.file_id, line_number)
        self._index.append_entry("license", metadata["license"], self.dataset_id, self.file_id, line_number)
