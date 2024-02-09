from typing import Any, Callable
from pathlib import Path
from mixtera.core.processing.property_calculation import PropertyCalculationExecutor
from mixtera.utils import flatten


class LocalPropertyCalculationExecutor(PropertyCalculationExecutor):
    def __init__(
        self,
        dop: int,
        setup_func: Callable,
        calc_func: Callable,
    ):
        self._dop = dop
        self._setup_func = setup_func
        self._calc_func = calc_func
        self._ds_sample_cache: dict[str, list[str]] = {}

    def load_data(self, datasets_and_files: list[tuple[str, Any]], data_only_on_primary: bool) -> None:
        del data_only_on_primary # Only primary loads data for local processng anyways

        file: str
        for dataset, file in datasets_and_files:
            self._sample_cache.extend(flatten([LocalPropertyCalculationExecutor._read_samples_from_file()]))


    def run(self) -> dict[str, dict[str, Any]]:
        raise NotImplementedError("wip")
    
    @classmethod
    def _read_samples_from_file(file: str) -> str:
        # TODO(create issue): This currently assumes everything is jsonl file
        if not file.endswith(".jsonl"): # hacky check for extension to have some kind of check
            raise NotImplementedError("The current implementation assumes jsonl files.")
        
        file_path = Path(file)

        if not file_path.exists():
            raise RuntimeError(f"File {file_path} does not exist.")

        with open(file_path, encoding="utf-8") as fp:
            return [line.rstrip() for line in fp]

