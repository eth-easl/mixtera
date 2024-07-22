import multiprocessing as mp
import random
from queue import Empty
from typing import Callable, Iterator, Optional, Type

import dill
from loguru import logger
from mixtera.core.datacollection.datasets import Dataset
from mixtera.core.datacollection.index import ChunkerIndex, IndexRowRangeType
from mixtera.core.query.mixture import StaticMixture
from mixtera.network.connection import ServerConnection
from mixtera.utils import generate_hash_string_from_list, seed_everything

Workload = tuple[int, int, IndexRowRangeType]
Workloads = list[Workload]


class ResultChunk:
    def __init__(
        self,
        result_index: ChunkerIndex,
        dataset_type_dict: dict[int, Type[Dataset]],
        file_path_dict: dict[int, str],
        parsing_func_dict: dict[int, Callable[[str], str]],
        chunk_size: int,
        mixture: Optional[dict[str, int]] = None,
    ) -> None:
        self._result_index = result_index
        self._dataset_type_dict = dataset_type_dict
        self._file_path_dict = file_path_dict
        self._parsing_func_dict = parsing_func_dict
        self._chunk_size = chunk_size
        self._mixture = mixture

        self._server_connection: Optional[ServerConnection] = None
        self._degree_of_parallelism: int = 1
        self._per_window_mixture: bool = False
        self._window_size: int = 128

        self._iterator: Optional[Iterator[str]] = None

    def configure_result_streaming(
        self,
        server_connection: ServerConnection,
        degree_of_parallelism: int = 1,
        per_window_mixture: bool = False,
        window_size: int = 128,
    ) -> None:
        self._server_connection = server_connection
        self._degree_of_parallelism = degree_of_parallelism
        self._per_window_mixture = per_window_mixture
        self._window_size = window_size

        if self._degree_of_parallelism < 1:
            logger.warning(
                f"Degree of parallelism is set to {self._degree_of_parallelism} which is invalid. "
                "Setting degree of parallelism to 1."
            )
            self._degree_of_parallelism = 1

        if self._per_window_mixture and self._window_size > self._chunk_size:
            logger.warning(
                f"Window size is set to {self._window_size} which is > the chunk size of {self._chunk_size}. "
                "Setting window size to the chunk size."
            )
            self._window_size = self._chunk_size

        if self._per_window_mixture and self._window_size < 1:
            logger.warning(
                f"Window size is set to {self._window_size} which is invalid. " "Setting window size to 128."
            )
            self._window_size = 128

        # To determine the number of processes per property combination, we need the mixture
        #  for parallel reading. If the mixture is not defined, we infer it from the result index.
        if (self._per_window_mixture or self._degree_of_parallelism > 1) and (
            self._mixture is None or len(self._mixture) == 0
        ):
            logger.debug("Mixture is not defined or empty but required. Infer mixture from the result index.")
            self._mixture = self._infer_mixture()

    def _infer_mixture(self) -> dict[str, int]:
        """
        Infer the mixture from the result index. This is done by calculating the mass of each partition
        and normalizing it to the total mass.

        Returns:
            The inferred mixture
        """
        total_count = 0
        partition_masses: dict[str, int | float] = {}

        def calculate_partition_mass(partition: dict[int, dict[int, list[tuple[int, int]]]]) -> int:
            mass = sum(
                end - start
                for file_entry in partition.values()
                for ranges in file_entry.values()
                for start, end in ranges
            )
            return mass

        for property_combination, partition_entry in self._result_index.items():
            partition_mass = calculate_partition_mass(partition_entry)
            partition_masses[property_combination] = partition_mass
            total_count += partition_mass

        for key in partition_masses:
            partition_masses[key] /= total_count

        return StaticMixture(total_count, partition_masses).mixture_in_rows()

    def _iterate_samples(self) -> Iterator[str]:
        if self._degree_of_parallelism == 1:
            yield_source = self._iterate_single_threaded()
        else:
            yield_source = self._iterate_multi_threaded()
        yield from yield_source

    def _iterate_single_threaded(self) -> Iterator[str]:
        if self._per_window_mixture:
            yield_source = self._iterate_single_threaded_window_mixture()
        else:
            yield_source = self._iterate_single_threaded_overall_mixture()
        yield from yield_source

    def _iterate_single_threaded_window_mixture(self) -> Iterator[str]:
        """
        Iterate over the result index with a windowed mixture. This means that the window size is fixed
        (should be <= chunk size) and the mixture is applied per window. This means that the window size
        is filled with the correct mixture of property combinations.

        In this single threaded version, we iterate over the workloads for each property combination and yield instances
        until the window is full. Then we start the next window.

        Returns:
            An iterator over the instances in the result index
        """
        element_counts = self._get_element_counts()
        workloads: dict[str, Workloads] = self._prepare_workloads()

        #  Create iterators for each property combination
        current_iterators = {
            property_name: iter(self._get_iterator_for_workload(property_workload))
            for property_name, property_workload in workloads.items()
        }

        processed_items = {property_name: 0 for property_name in workloads}

        #  Shuffle the results to ensure that the order of the property combinations is (reproducibly) random
        seed_everything(generate_hash_string_from_list([x[0] for x in element_counts]))
        random.shuffle(element_counts)

        # Continue until all workloads are processed
        while current_iterators:
            items_yielded = 0
            #  This inner while loop represents one window with the correct mixture
            #  We continue until the window is full or there are no more workloads to yield
            while current_iterators and items_yielded < self._window_size:
                nothing_yielded_window = True
                for property_name, property_count in element_counts:
                    if property_name not in current_iterators or processed_items[property_name] >= property_count:
                        #  If no more workloads for this property this window, skip
                        continue
                    try:
                        # Yield the next instance from the iterator
                        yield next(current_iterators[property_name])
                        nothing_yielded_window = False
                        processed_items[property_name] += 1
                        items_yielded += 1
                        if items_yielded >= self._window_size:
                            #  If the window is full, break the inner loop, will also break the outer loop
                            #  since the items_yielded >= self._window_size, start the next window
                            processed_items = {property_name: 0 for property_name in workloads}
                            break
                    except StopIteration:
                        # If no more workloads, this property is done
                        del current_iterators[property_name]

                if nothing_yielded_window:
                    break

    def _get_iterator_for_workload(self, workload: list[tuple[int, int, list]]) -> Iterator[str]:
        """
        Get an iterator for a workload. This iterator reads the instances from the files in the workload.
        For each dataset_id, file_id and ranges tuple in the workload, the instances are read from the file
        and yielded.

        Args:
            workload: a list of tuples with dataset_id, file_id and ranges

        Returns:
            An iterator over the instances in the workload
        """
        for dataset_id, file_id, ranges in workload:
            filename_dict = {self._file_path_dict[file_id]: ranges}
            yield from self._dataset_type_dict[dataset_id].read_ranges_from_files(
                filename_dict, self._parsing_func_dict[dataset_id], self._server_connection
            )

    def _iterate_single_threaded_overall_mixture(self) -> Iterator[str]:
        """
        Iterate over the result index with an overall mixture. This means that the mixture is applied to the entire
        result chunk. This means that the instances are yielded in a random, reproducible, order, but the mixture
        is respected.

        In this single threaded version, we iterate over the workloads for each property combination and yield instances
        until all workloads are processed.

        Returns:
            An iterator over the instances in the result index
        """
        workloads: dict[str, Workloads] = self._prepare_workloads()

        active_iterators = [
            (property_name, self._get_iterator_for_workload(workload)) for property_name, workload in workloads.items()
        ]

        #  Shuffle the results to ensure that the order of the property combinations is (reproducibly) random
        seed_everything(generate_hash_string_from_list([x[0] for x in active_iterators]))
        random.shuffle(active_iterators)

        while active_iterators:
            for property_name, iterator in active_iterators:
                try:
                    yield next(iterator)
                except StopIteration:
                    active_iterators.remove((property_name, iterator))

    def _iterate_multi_threaded(self) -> Iterator[str]:
        assert isinstance(self._mixture, dict), "Mixture must be defined for parallel reading, this should not happen."

        # Collect the workloads (i.e. did+fid+ranges) and group them by the property combination they belong to
        workloads: dict[str, Workloads] = self._prepare_workloads()

        # Determine the number of readers to use s.t. readers are not overprovisioned
        reader_count = min(
            sum(len(x) for x in workloads.values()),
            self._degree_of_parallelism if self._degree_of_parallelism is not None else mp.cpu_count(),
        )

        # Determine how many processes should be assigned per property combination
        process_counts = {key: int(val * reader_count) for key, val in self._mixture.items()}

        process_counts[list(process_counts.keys())[0]] += reader_count - sum(process_counts.values())

        processes: dict[str, list[tuple[mp.Queue, mp.Process]]] = self._spin_up_readers(workloads, process_counts)

        if self._per_window_mixture:
            yield_source = self._iterate_multi_threaded_window_mixture(processes)
        else:
            yield_source = self._iterate_multi_threaded_overall_mixture(processes)
        yield from yield_source

    def _get_element_counts(self) -> list[tuple[str, int]]:
        """
        Get the element counts for each property combination. This is used to determine how many instances
        of each property combination should be yielded in a window.

        Returns:
            A list of tuples with the property combination and the number of instances to yield
        """
        assert isinstance(self._mixture, dict), "Mixture must be defined for windowed reading, this should not happen."

        # Determine the per-property combination batch counts
        initial_counts = [
            (key, int(self._window_size * (value / self._chunk_size))) for key, value in self._mixture.items()
        ]
        total_counts = sum(count for _, count in initial_counts)
        remainder = self._window_size - total_counts

        #  Adjust the counts to ensure that the window size is met
        adjusted_counts = [
            (key, count + remainder if i == 0 else count) for i, (key, count) in enumerate(initial_counts)
        ]

        return adjusted_counts

    def _iterate_multi_threaded_window_mixture(
        self,
        processes: dict[str, list[tuple[mp.Queue, mp.Process]]],
    ) -> Iterator[str]:
        """
        Iterate over the result index with a windowed mixture. This means that the window size is fixed
        (should be <= chunk size) and the mixture is applied per window. This means that the window size
        is filled with the correct mixture of property combinations.

        In this multi-threaded version, we iterate over the workloads for each property combination and yield instances
        until the window is full. Then we start the next window.

        Returns:
            An iterator over the instances in the result index
        """
        element_counts = self._get_element_counts()

        processed_items = {property_tuple[0]: 0 for property_tuple in element_counts}

        #  Shuffle the results to ensure that the order of the property combinations is (reproducibly) random
        seed_everything(generate_hash_string_from_list([x[0] for x in element_counts]))
        random.shuffle(element_counts)

        while len(processes) > 0:  # pylint: disable=too-many-nested-blocks
            items_yielded = 0

            #  This inner while loop represents one window with the correct mixture
            #  We continue until the window is full or there are no more workloads to yield
            while len(processes) > 0 and items_yielded < self._window_size:
                nothing_yielded_window = True
                for property_name, property_count in element_counts:
                    if (
                        property_name not in processes
                        or processed_items[property_name] > property_count
                        or len(processes[property_name]) == 0
                    ):
                        #  If no more workloads for this property this window, skip
                        continue

                    #  Get the first queue and process for this property combination
                    # If this combination has no more processes, it will be removed from the list
                    # and the next iteration a new "first" combination will be retrieved
                    q, proc = processes[property_name][0]

                    try:
                        instance = q.get_nowait()
                    except Empty:
                        if not proc.is_alive():
                            #  If the process is dead, remove it from the list
                            processes[property_name].remove((q, proc))
                            if len(processes[property_name]) == 0:
                                del processes[property_name]
                        continue

                    yield instance
                    processed_items[property_name] += 1
                    items_yielded += 1
                    logger.debug(f"Yielded instance {items_yielded} in window for {property_name}")
                    nothing_yielded_window = False

                    if items_yielded >= self._window_size:
                        #  If the window is full, break the inner loop, will also break the outer loop
                        #  since the items_yielded >= self._window_size, start the next window
                        processed_items = {property_tuple[0]: 0 for property_tuple in element_counts}
                        break

            if nothing_yielded_window:
                break

    def _iterate_multi_threaded_overall_mixture(
        self,
        processes: dict[str, list[tuple[mp.Queue, mp.Process]]],
    ) -> Iterator[str]:
        """
        Iterate over the result index with an overall mixture. This means that the mixture is applied to the entire
        result chunk. This means that the instances are yielded in a random, reproducible, order,
        but the mixture is respected.

        In this multi-threaded version, we iterate over the workloads for each property combination and yield instances
        until all workloads are processed.

        Returns:
            An iterator over the instances in the result index
        """
        property_order = list(processes.keys())

        seed_everything(generate_hash_string_from_list(property_order))
        random.shuffle(property_order)

        while len(processes) > 0:
            yielded_in_round = False

            for property_name in property_order:
                if not processes[property_name] or len(processes[property_name]) == 0:
                    continue

                #  Get the first queue and process for this property combination
                # If this combination has no more processes, it will be removed from the list
                # and the next iteration a new "first" combination will be retrieved
                q, proc = processes[property_name][0]

                try:
                    instance = q.get_nowait()
                except Empty:
                    #  If the queue is empty, check if the process is still alive
                    if not proc.is_alive():
                        #  If the process is dead, remove it from the list
                        processes[property_name].remove((q, proc))
                        if len(processes[property_name]) == 0:
                            del processes[property_name]
                    continue

                yield instance
                yielded_in_round = True

            if not yielded_in_round:
                break

    def _spin_up_readers(
        self,
        workloads: dict[str, list[tuple[int, int, list]]],
        process_counts: dict[str, int],
    ) -> dict[str, list[tuple[mp.Queue, mp.Process]]]:
        """
        Spin up the reader processes for the workloads. This function creates the processes and queues
        for the workloads and starts the processes.

        Args:
            workloads: a dictionary with the workloads per property combination
            process_counts: a dictionary with the number of processes per property combination
        """
        processes: dict[str, list[tuple[mp.Queue, mp.Process]]] = {}
        for key, process_count in process_counts.items():
            processes[key] = []

            # Calculate per-process partition sizes
            partition_size = max(1, len(workloads[key]) // process_count)
            partition_ranges = list(range(0, len(workloads[key]), partition_size)) + [len(workloads[key])]

            # Create and start the processes
            for i in range(1, len(partition_ranges)):
                queue: mp.Queue = mp.Queue()
                processes[key].append(
                    (
                        queue,
                        mp.Process(
                            target=self._reader_process,
                            args=(
                                queue,
                                self._dataset_type_dict,
                                self._file_path_dict,
                                dill.dumps(self._parsing_func_dict),
                                self._server_connection,
                                workloads[key][partition_ranges[i - 1] : partition_ranges[i]],
                            ),
                        ),
                    )
                )

                # Start the process
                processes[key][-1][1].start()

        return processes

    @staticmethod
    def _reader_process(
        queue: mp.Queue,
        dataset_type_dict: dict[int, Type[Dataset]],
        file_path_dict: dict[int, str],
        pickled_parsing_func_dict: bytes,
        server_connection: ServerConnection,
        workloads: Workloads,
    ) -> None:
        """
        The reader process reads the instances from the files in the workloads and puts them into the queue.

        Args:
            queue: the queue to put the instances into
            dataset_type_dict: a dictionary with the dataset types
            file_path_dict: a dictionary with the file paths
            pickled_parsing_func_dict: a pickled dictionary with the parsing functions
            server_connection: the server connection to use
            workloads: the workloads to process
        """
        parsing_func_dict: dict[int, Callable[[str], str]] = dill.loads(pickled_parsing_func_dict)
        for dataset_id, file_id, ranges in workloads:
            filename_dict = {file_path_dict[file_id]: ranges}
            instance_iterator = dataset_type_dict[dataset_id].read_ranges_from_files(
                filename_dict, parsing_func_dict[dataset_id], server_connection
            )
            for instance in instance_iterator:
                queue.put(instance)

        queue.close()

    def _prepare_workloads(self) -> dict[str, Workloads]:
        """
        Prepare the workloads for the result index. This function creates a dictionary with the workloads
        per property combination.

        Returns:
            A dictionary with the workloads per property combination
        """
        workloads: dict[str, Workloads] = {}
        for property_combination, dataset_entries in self._result_index.items():
            if property_combination not in workloads:
                workloads[property_combination] = []
            for dataset_id, file_entries in dataset_entries.items():
                for file_id, ranges in file_entries.items():
                    workloads[property_combination].append((dataset_id, file_id, ranges))
        return workloads

    def __iter__(self) -> "ResultChunk":
        self._iterator = self._iterate_samples()
        return self

    def __next__(self) -> Iterator[str]:
        if self._iterator is None:
            raise StopIteration
        try:
            return next(self._iterator)  # type: ignore  # MyPy seems to expect SupportsNext[Iterator[str]] here
        except StopIteration:
            self._iterator = None
            raise
