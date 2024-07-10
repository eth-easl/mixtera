from pathlib import Path
from typing import Any, List, Optional

import numpy as np
from mixtera.core.datacollection.index.parser import MetadataParser


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


def prep_dir(dir: Path) -> str:
    print(f"Prepping directory {dir}.")
    write_jsonl(dir / "testd.jsonl")
    print("Directory prepped.")
    return dir / "testd.jsonl"


def setup_func(some_class: Any):
    pass


def calc_func(executor: Any, batch: dict[str, np.ndarray]) -> List[Any]:
    return ["test_category"]


class TestMetadataParser(MetadataParser):
    def parse(self, line_number: int, payload: Any, **kwargs: Optional[dict[Any, Any]]) -> None:
        metadata = payload["meta"]
        self._index.append_entry("language", metadata["language"], self.dataset_id, self.file_id, line_number)
        self._index.append_entry("license", metadata["license"], self.dataset_id, self.file_id, line_number)
