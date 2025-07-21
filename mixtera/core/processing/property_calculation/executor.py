from abc import ABC, abstractmethod
from typing import Any, Callable, Type

import numpy as np

from mixtera.core.datacollection.datasets import Dataset
from mixtera.core.processing import ExecutionMode


class PropertyCalculationExecutor(ABC):
    @staticmethod
    def from_mode(
        mode: ExecutionMode,
        degree_of_parallelism: int,
        batch_size: int,
        setup_func: Callable[[Any], None],
        calc_func: Callable[[Any, dict[str, np.ndarray]], list[Any]],
    ) -> "PropertyCalculationExecutor":
        """
        This function instantiates a new PropertyCalculationExecutor based on the mode.

        Args:
            mode (ExecutionMode): The execution mode to use
            degree_of_parallelism (int): Degree of parallelism. How many processing units should be used in parallel.
                       Meaning depends on execution_mode
            setup_func (Callable): Function that performs setup (e.g., load model).
                                   It is passed an instance of a class (type "Any") to put attributes on.
                                   This class will be available in the calc_func.
            calc_func (Callable): Given a batch of data in form of a dict[str, np.ndarray]]
                                  { "data": [...], "file_id": [...], "line_id": [...] },
                                  i.e., batched along the properties in numpy arrays (depicted above as a list),
                                  this returns one prediction (class or score) per item in the batch.
                                  The batching is taken care of by the concrete executor.
                                  It has access to the class that was prepared by the setup_func.

        Returns:
            An instance of a PropertyCalculationExecutor subclass.
        """
        if degree_of_parallelism < 1:
            raise RuntimeError(f"Degree of parallelism = {degree_of_parallelism} < 1")

        if batch_size < 1:
            raise RuntimeError(f"Batch size = {batch_size} < 1")

        if mode == ExecutionMode.LOCAL:
            # pylint: disable-next=import-outside-toplevel
            from mixtera.core.processing.property_calculation import LocalPropertyCalculationExecutor

            return LocalPropertyCalculationExecutor(degree_of_parallelism, batch_size, setup_func, calc_func)

        raise NotImplementedError(f"Mode {mode} not yet implemented.")

    @abstractmethod
    def load_data(self, files: list[tuple[int, int, Type[Dataset], str]], data_only_on_primary: bool) -> None:
        """
        Loads the data, i.e., all files, into the executor. Needs to be called before calling `run`.

        Args:
            files (list[tuple[int, int, Type[Dataset], str]]): A list of file_ids, dataset_ids,
                dataset type of this id, and file paths to load
            data_only_on_primary (bool): If False, the processing units (may be remote machines)
                                         have access to the same paths as the primary.
                                         Allows for non-centralized reading of files.
        """

        raise NotImplementedError()

    @abstractmethod
    def run(self) -> dict[str, list[tuple[int, int, int]]]:
        """
        Actually runs calculation of the new property and returns the new property for the index.

        Returns:
            A dictionary to be merged into the main index under the appropriate key for the property.
        """

        raise NotImplementedError()
