from collections import defaultdict
from itertools import batched
from pathlib import Path
from typing import Any, Callable

from loguru import logger
from mixtera.core.processing.property_calculation import PropertyCalculationExecutor
from mixtera.utils import dict_into_dict
from mixtera.utils.utils import defaultdict_to_dict


class LocalPropertyCalculationExecutor(PropertyCalculationExecutor):
    def __init__(
        self,
        dop: int,
        batch_size: int,
        setup_func: Callable,
        calc_func: Callable,
    ):
        self._dop = dop
        self._setup_func = setup_func
        self._calc_func = calc_func
        self._batch_size = batch_size

        self._sample_cache: list[list[str]] = []
        self._setup_func(self)  # We need to explicitly pass self here

    def load_data(self, files: list[str], data_only_on_primary: bool) -> None:
        if not data_only_on_primary:
            logger.warning("Set data_only_on_primary = False, but LocalExecutor is running only on primary anyways.")

        # TODO(create issue): Maybe add option to chunk to maximum number of samples
        for file_batch in batched(files, self._batch_size):
            self._sample_cache.extend(
                [LocalPropertyCalculationExecutor._read_samples_from_file(file) for file in file_batch]
            )

    def run(self) -> dict[str, list[Any]]:
        result: defaultdict[str, defaultdict[str, list]] = defaultdict(lambda: defaultdict(list))
        for batch in self._sample_cache:
            dict_into_dict(result, self._calc_func(self, batch))

        return defaultdict_to_dict(result)

    @staticmethod
    def _read_samples_from_file(file: str) -> list[str]:
        # TODO(create issue): This currently assumes everything is jsonl file

        if not file.endswith(".jsonl"):  # hacky check for extension to have some kind of check
            raise NotImplementedError("The current implementation assumes jsonl files.")

        file_path = Path(file)

        if not file_path.exists():
            raise RuntimeError(f"File {file_path} does not exist.")

        with open(file_path, encoding="utf-8") as fp:
            return [line.rstrip() for line in fp]
