from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

from mixtera.core.datacollection.index.parser import MetadataParser
from mixtera.core.datacollection.index.parser.metadata_parser import MetadataProperty

REPRODUCIBILITY_ITERATIONS = 4


def write_jsonl(path: Path, file_count: int, instance_count_per_file: int, fraction_multiplier: int) -> None:
    # We alternate within each language between the two licenses.
    counters = {"JavaScript": 0, "HTML": 0}

    for file_number in range(file_count):
        data = ""
        for i in range(instance_count_per_file):
            # Determine the language and increment the counter
            language = "JavaScript" if i % fraction_multiplier == 0 else "HTML"
            counters[language] += 1

            # Determine the license based on the counter for the current language
            license = "MIT" if counters[language] % 2 == 0 else "CC"

            # Append the data string with the new entry
            data += f'{{ "text": "{i}", "meta": {{ "language": "{language}", "license": "{license}"}}}}\n'

        # Write the data to a file
        with open(path / f"data_{file_number}.jsonl", "w") as text_file:
            text_file.write(data)


def get_expected_js_and_html_samples(
    total_instance_count: int, fraction_multiplier: int
) -> tuple[List[int], List[int]]:
    return (
        total_instance_count // fraction_multiplier,
        total_instance_count - total_instance_count // fraction_multiplier,
    )


def setup_test_dataset(
    dir: Path, total_instance_count: int = 1000, file_count: int = 10, fraction_multiplier: int = 2
) -> None:
    print(f"Prepping directory {dir}.")
    write_jsonl(dir, file_count, total_instance_count // file_count, fraction_multiplier)
    print("Directory prepped.")


def setup_func(some_class: Any):
    pass


def calc_func(executor: Any, batch: dict[str, np.ndarray]) -> List[Any]:
    return ["test_category"]


class TestMetadataParser(MetadataParser):
    @classmethod
    def get_properties(cls) -> list[MetadataProperty]:
        return [
            MetadataProperty(
                name="language", dtype="ENUM", multiple=False, nullable=False, enum_options={"JavaScript", "HTML"}
            ),
            MetadataProperty(
                name="license", dtype="STRING", multiple=False, nullable=False, enum_options={"CC", "MIT"}
            ),  # Could be ENUM but we are using string to test
            MetadataProperty(
                name="doublelanguage", dtype="ENUM", multiple=True, nullable=False, enum_options={"JavaScript", "HTML"}
            ),
        ]

    def parse(self, line_number: int, payload: Any, **kwargs: Optional[dict[Any, Any]]) -> None:
        metadata = payload["meta"]
        self.add_metadata(
            sample_id=line_number,
            language=metadata["language"],
            license=metadata["license"],
            doublelanguage=[metadata["language"], metadata["language"]],
        )
