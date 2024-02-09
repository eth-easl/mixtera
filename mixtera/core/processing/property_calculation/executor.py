from abc import ABC, abstractmethod
from typing import Any, Callable

from mixtera.core.processing import ExecutionMode


class PropertyCalculationExecutor(ABC):

    @staticmethod
    def from_mode(
        mode: ExecutionMode,
        dop: int,
        setup_func: Callable,
        calc_func: Callable,
    ) -> "PropertyCalculationExecutor":
        """
        This function instantiates a new PropertyCalculationExecutor based on the mode.


        Args:
            mode (ExecutionMode): The execution mode to use
            dop (int): Degree of parallelism. How many processing units should be used in parallel. Meaning depends on execution_mode
            setup_func (Callable): Function that performs setup (e.g., load model).
                                   It is passed an instance of a class to put attributes on.
            calc_func (Callable): The function that given a batch of data calculates a numerical or categorical value.
                                  It has access to the class that was prepared by the setup_func.

        Returns:
            An instance of a PropertyCalculationExecutor subclass.
        """
        if dop < 1:
            raise RuntimeError(f"dop = {dop} < 1")
        
        if mode == ExecutionMode.LOCAL:
            from mixtera.core.processing.property_calculation import LocalPropertyCalculationExecutor
            return LocalPropertyCalculationExecutor(dop, setup_func, calc_func)

        raise NotImplementedError(f"Mode {mode} not yet implemented.")

    @abstractmethod
    def load_data(self, datasets_and_files: list[tuple[str, Any]], data_only_on_primary: bool) -> None:
        """
        Loads the data, i.e., all files, into the executor. Needs to be done before calling run.

        Args:
            datasets_and_files (list[tuple[str, Any]): A list of tuples (dataset identifier, file path)
            data_only_on_primary (bool): If False, the processing units (may be remote machines) have access to the same paths as the primary.
                                         Allows for non-centralized reading of files.
        """

        raise NotImplementedError()

    @abstractmethod
    def run(
        self,
    ) -> dict[str, dict[str, Any]]:
        """
        Actually runs calculation of the new property and returns the new property for the index.

        Returns:
            A dictionary to be merged into the main index. Currently partitioned by dataset on the first level (probably we will get rid of that)
        """

        raise NotImplementedError()
