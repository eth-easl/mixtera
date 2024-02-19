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
        dop: int,
        batch_size: int,
        setup_func: Callable[[Any], None],
        calc_func: Callable[[Any, dict[str, np.ndarray]], list[Any]],
    ):
        self._dop = dop  # TODO(#24): support dop using multiprocessing or so
        self._setup_func = setup_func
        self._calc_func = calc_func
        self._batch_size = batch_size

        self._batches: list[dict[str, np.ndarray]] = []
        self._setup_func(self)  # We need to explicitly pass self here

        if self._dop > 1:
            raise NotImplementedError("The LocalPropertyCalculationExecutor currently does not support parallelism.")

    def load_data(self, files: list[tuple[int, int, str]], data_only_on_primary: bool) -> None:
        if not data_only_on_primary:
            logger.warning("Set data_only_on_primary = False, but LocalExecutor is running only on primary anyways.")

        data = []
        file_ids = []
        dataset_ids = []
        line_ids = []
        count = 0
        for file_id, dataset_id, path in files:
            for line_id, line in self._read_samples_from_file(path):
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

    def run(self) -> defaultdict[str, defaultdict[int, defaultdict[int, list[tuple[int, int]]]]]:
        inference_result_per_file: defaultdict[str, defaultdict[int, defaultdict[int, list[tuple[int]]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )

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
                # TODO(#11): This currently assumes we are in a categorical bucket and do not need to discretize
                # (prediction directly gives bucket)
                if not isinstance(prediction, str):
                    raise NotImplementedError(
                        "Right now we assume prediction to be a category, numerical values not yet supported."
                    )

                inference_result_per_file[prediction][dataset_id][file_id].append(line_id)

        rangified: defaultdict[str, defaultdict[int, defaultdict[int, list[tuple[int, int]]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )

        for bucket_val, datasets in inference_result_per_file.items():
            for dataset_id, files in datasets.items():
                for file_id, list_of_lines in files.items():
                    rangified[bucket_val][dataset_id][file_id] = ranges(list_of_lines)

        return rangified

    @staticmethod
    def _read_samples_from_file(file: str) -> Generator[tuple[int, str], None, None]:
        # TODO(#22): This currently assumes everything is jsonl file
        if not file.endswith(".jsonl"):  # hacky check for extension to have some kind of check
            raise NotImplementedError("The current implementation assumes jsonl files.")

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
