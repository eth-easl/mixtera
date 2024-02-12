from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Generator

import numpy as np
from loguru import logger
from mixtera.core.processing.property_calculation import PropertyCalculationExecutor
from mixtera.utils import ranges
from tqdm import tqdm


class LocalPropertyCalculationExecutor(PropertyCalculationExecutor):
    def __init__(
        self,
        property_name: str,
        dop: int,
        batch_size: int,
        setup_func: Callable[[Any], None],
        calc_func: Callable[[Any, dict[str, np.ndarray]], list[Any]],
    ):
        self._dop = dop  # TODO (create issue): support dop using multiprocessing or so
        self._setup_func = setup_func
        self._calc_func = calc_func
        self._batch_size = batch_size
        self._property_name = property_name

        self._batches: list[dict[str, np.ndarray]] = []
        self._setup_func(self)  # We need to explicitly pass self here

    def load_data(self, files: list[tuple[int, str]], data_only_on_primary: bool) -> None:
        if not data_only_on_primary:
            logger.warning("Set data_only_on_primary = False, but LocalExecutor is running only on primary anyways.")

        data = []
        file_ids = []
        line_ids = []
        count = 0
        for file_id, path in files:
            for file_id, line_id, line in self._read_samples_from_file(file_id, path):
                data.append(line)
                file_ids.append(file_id)
                line_ids.append(line_id)
                count += 1
                if count == self._batch_size:
                    self._batches.append(self._create_batch(data, file_ids, line_ids))
                    data = []
                    file_ids = []
                    line_ids = []
                    count = 0

        if count > 0:
            self._batches.append(self._create_batch(data, file_ids, line_ids))

    def run(self) -> dict[str, list[tuple[int, int, int]]]:
        # We use self._property_name here as first index to rely on dict_into_dict
        inference_result_per_file: defaultdict[str, defaultdict[int, list[tuple[int, int]]]] = defaultdict(
            defaultdict(list)
        )

        for batch in tqdm(self._batches, desc="Processing batches", total=len(self._batches)):
            batch_predictions = self._calc_func(self, batch)

            if len(batch_predictions) != len(batch["file_id"]) or len(batch_predictions) != len(batch["line_id"]):
                raise RuntimeError(f"Length mismatch: {batch_predictions} vs {batch}.")

            # Build up index { "bucket_name": [(file_id, line_id) tuples]}
            for prediction, file_id, line_id in zip(batch_predictions, batch["file_id"], batch["line_id"]):
                # TODO(#11): This currently assumes we are in a categorical bucket and do not need to discretize
                # (prediction directly gives bucket)
                inference_result_per_file[prediction][file_id].append(line_id)

        return {
            bucket: [(file_id,) + ranges(sorted(lines)) for file_id, lines in file_dict.items()]
            for bucket, file_dict in inference_result_per_file.items()
        }

    @staticmethod
    def _read_samples_from_file(file_id: int, file: str) -> Generator[tuple[int, int, str]]:
        # TODO(create issue): This currently assumes everything is jsonl file

        if not file.endswith(".jsonl"):  # hacky check for extension to have some kind of check
            raise NotImplementedError("The current implementation assumes jsonl files.")

        file_path = Path(file)

        if not file_path.exists():
            raise RuntimeError(f"File {file_path} does not exist.")

        with open(file_path, encoding="utf-8") as fp:
            for line_id, line in enumerate(fp):
                yield file_id, line_id, line.rstrip()

    def _create_batch(self, data: list[str], file_ids: list[int], line_ids: list[int]) -> dict[str, np.ndarray]:
        return {"data": np.array(data), "file_id": np.array(file_ids), "line_id": np.array(line_ids)}
