from pathlib import Path
from typing import Any, Callable, Generator, Type

import numpy as np
from loguru import logger
from mixtera.core.datacollection.datasets import Dataset, JSONLDataset
from mixtera.core.processing.property_calculation import PropertyCalculationExecutor
from tqdm import tqdm


class LocalPropertyCalculationExecutor(PropertyCalculationExecutor):
    def __init__(
        self,
        degree_of_parallelism: int,
        batch_size: int,
        setup_func: Callable[[Any], None],
        calc_func: Callable[[Any, dict[str, np.ndarray]], list[Any]],
    ):
        # TODO(#24): support degree_of_parallelism using multiprocessing
        self._degree_of_parallelism = degree_of_parallelism
        self._setup_func = setup_func
        self._calc_func = calc_func
        self._batch_size = batch_size

        self._batches: list[dict[str, np.ndarray]] = []
        self._setup_func(self)  # We need to explicitly pass self here

        if self._degree_of_parallelism > 1:
            raise NotImplementedError("The LocalPropertyCalculationExecutor currently does not support parallelism.")

    def load_data(self, files: list[tuple[int, int, Type[Dataset], str]], data_only_on_primary: bool) -> None:
        if not data_only_on_primary:
            logger.warning("Set data_only_on_primary = False, but LocalExecutor is running only on primary anyways.")

        data = []
        file_ids = []
        dataset_ids = []
        line_ids = []
        count = 0
        for file_id, dataset_id, dtype, path in files:
            for line_id, line in self._read_samples_from_file(path, dtype):
                data.append(line)
                file_ids.append(file_id)
                line_ids.append(line_id)
                dataset_ids.append(dataset_id)
                count += 1
                if count == self._batch_size:
                    self._batches.append(self._create_batch(data, file_ids, dataset_ids, line_ids))
                    data = []
                    file_ids = []
                    line_ids = []
                    dataset_ids = []
                    count = 0

        if count > 0:
            self._batches.append(self._create_batch(data, file_ids, dataset_ids, line_ids))

    def run(self) -> list[dict[str, Any]]:
        inference_results = []

        for batch in tqdm(self._batches, desc="Processing batches", total=len(self._batches)):
            batch_predictions = self._calc_func(self, batch)

            if (
                len(batch_predictions) != len(batch["file_id"])
                or len(batch_predictions) != len(batch["line_id"])
                or len(batch_predictions) != len(batch["dataset_id"])
            ):
                raise RuntimeError(f"Length mismatch: {batch_predictions} vs {batch}.")

            for prediction, file_id, dataset_id, line_id in zip(
                batch_predictions, batch["file_id"], batch["dataset_id"], batch["line_id"]
            ):
                # prediction can be a value or a list of values (for properties that are lists)
                if not isinstance(prediction, (str, int, float, list)):
                    raise NotImplementedError("Predictions must be string, int, float, or list of these types.")

                result = {
                    "dataset_id": dataset_id,
                    "file_id": file_id,
                    "sample_id": line_id,
                    "property_value": prediction,
                }
                inference_results.append(result)

        return inference_results

    @staticmethod
    def _read_samples_from_file(file: str, dtype: Type[Dataset]) -> Generator[tuple[int, str], None, None]:
        if dtype is JSONLDataset:
            return LocalPropertyCalculationExecutor._read_samples_from_jsonl_file(file)

        raise NotImplementedError(f"LocalExecutor currently does not support dataset type {dtype}")

    @staticmethod
    def _read_samples_from_jsonl_file(file: str) -> Generator[tuple[int, str], None, None]:
        file_path = Path(file)

        if not file_path.exists():
            raise RuntimeError(f"File {file_path} does not exist.")

        with open(file_path, encoding="utf-8") as fp:
            for line_id, line in enumerate(fp):
                yield line_id, line.rstrip()

    def _create_batch(
        self, data: list[str], file_ids: list[int], dataset_ids: list[int], line_ids: list[int]
    ) -> dict[str, np.ndarray]:
        return {
            "data": np.array(data),
            "file_id": np.array(file_ids),
            "dataset_id": np.array(dataset_ids),
            "line_id": np.array(line_ids),
        }
