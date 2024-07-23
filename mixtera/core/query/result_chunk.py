import multiprocessing as mp
import random
from queue import Empty
from typing import TYPE_CHECKING, Callable, Iterator, Optional, Type

import dill
from loguru import logger
from mixtera.core.datacollection.datasets import Dataset
from mixtera.core.datacollection.index import ChunkerIndex, IndexRowRangeType
from mixtera.core.query.mixture import StaticMixture
from mixtera.network.connection import ServerConnection
from mixtera.utils import generate_hash_string_from_list, seed_everything

if TYPE_CHECKING:
    from mixtera.core.client.mixtera_client import MixteraClient, ResultStreamingArgs

Workload = tuple[int, int, IndexRowRangeType]
Workloads = list[Workload]

MULTIPROCESSING_TIMEOUT = 5


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

    def configure_result_streaming(self, client: "MixteraClient", args: "ResultStreamingArgs") -> None:
        """
        Configure the result streaming for the ResultChunk. This function sets the degree of parallelism,
        the window size, and the mixture based on the arguments.

        Args:
            client: The MixteraClient instance
            args: The ResultStreamingArgs instance
        """
        self._degree_of_parallelism = args.chunk_reading_degree_of_parallelism
        self._per_window_mixture = args.chunk_reading_per_window_mixture
        self._window_size = args.chunk_reading_window_size

        from mixtera.core.client.server import ServerStub  # pylint: disable=import-outside-toplevel

        if args.tunnel_via_server:
            if isinstance(client, ServerStub):
                self._server_connection = client.get_server_connection()
            else:
                raise RuntimeError(
                    "Currently, tunneling samples via the server is only supported when using a ServerStub."
                )

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
        # for parallel reading. If the mixture is not defined, we infer it from the result index.
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
        """
        Iterate over the samples in the result index. This function yields the samples in the correct mixture
        and window size.

        Returns:
            An iterator over the samples
        """
        active_iterators: dict[str, Iterator[str]] = self._get_active_iterators()
        if self._per_window_mixture:
            yield_source = self._iterate_window_mixture(active_iterators)
        else:
            yield_source = self._iterate_overall_mixture(active_iterators)
        yield from yield_source

    def _get_active_iterators(self) -> dict[str, Iterator[str]]:
        """
        Get the active iterators for the result index. This function prepares the workloads and spins up the
        reader processes if required by the degree of parallelism.
        """
        workloads: dict[str, Workloads] = self._prepare_workloads()

        active_iterators: dict[str, Iterator[str]] = {}
        if self._degree_of_parallelism == 1:
            active_iterators = {
                property_name: self._get_iterator_for_workload_st(workload)
                for property_name, workload in workloads.items()
            }
        elif self._degree_of_parallelism > 1:
            process_counts = self._get_process_counts()

            processes: dict[str, list[tuple[mp.Queue, mp.Process]]] = self._spin_up_readers(workloads, process_counts)

            active_iterators = {
                property_name: self._get_iterator_for_workload_mt(process)
                for property_name, process in processes.items()
            }

        return active_iterators

    def _get_process_counts(self) -> dict[str, int]:
        """
        Get the number of processes per property combination. This function determines the number of processes
        to use based on the degree of parallelism and the mixture.

        Each property combination is assigned a number of processes based on the mass of the property combination
        in the mixture.
        """
        assert isinstance(
            self._mixture, dict
        ), "Mixture must be defined for parallel reading when getting the process counts, this should not happen."

        #  Determine the number of processes to use
        reader_count = min(
            self._degree_of_parallelism if self._degree_of_parallelism is not None else mp.cpu_count(),
            mp.cpu_count(),
        )

        # Determine how many processes should be assigned per property combination
        process_counts = {key: int((val / self._chunk_size) * reader_count) for key, val in self._mixture.items()}

        process_counts[list(process_counts.keys())[0]] += reader_count - sum(process_counts.values())

        return process_counts

    def _get_iterator_for_workload_st(self, workload: list[tuple[int, int, list]]) -> Iterator[str]:
        """
        Get the iterator for the workload in single-threaded mode. This function reads the instances from the
        files in the workload and yields them.
        """
        for dataset_id, file_id, ranges in workload:
            filename_dict = {self._file_path_dict[file_id]: ranges}
            yield from self._dataset_type_dict[dataset_id].read_ranges_from_files(
                filename_dict, self._parsing_func_dict[dataset_id], self._server_connection
            )

    def _get_iterator_for_workload_mt(self, processes: list[tuple[mp.Queue, mp.Process]]) -> Iterator[str]:
        """
        Get the iterator for the workload in multi-threaded mode. This function yields the instances from the
        queues of the processes.
        """
        for queue, proc in processes:
            while proc.is_alive() or not queue.empty():
                try:
                    instance = queue.get(timeout=MULTIPROCESSING_TIMEOUT)
                except Empty:
                    continue
                yield instance

    def _iterate_window_mixture(self, active_iterators: dict[str, Iterator[str]]) -> Iterator[str]:
        """
        Iterate over the samples in the result index with a windowed mixture. This function yields the samples
        in the correct mixture withing a window.
        """
        element_counts = self._get_element_counts()

        #  Shuffle the results to ensure that the order of the property combinations is (reproducibly) random
        seed_everything(generate_hash_string_from_list([x[0] for x in element_counts]))
        random.shuffle(element_counts)

        # Continue until all workloads are processed
        while active_iterators:
            items_yielded = 0
            processed_items = {property_tuple[0]: 0 for property_tuple in element_counts}
            #  This inner while loop represents one window with the correct mixture
            #  We continue until the window is full or there are no more workloads to yield
            while active_iterators and items_yielded < self._window_size:
                nothing_yielded_window = True
                for property_name, property_count in element_counts:
                    if property_name not in active_iterators or processed_items[property_name] >= property_count:
                        #  If no more workloads for this property this window, skip
                        continue
                    try:
                        # Yield the next instance from the iterator
                        yield next(active_iterators[property_name])
                        nothing_yielded_window = False
                        processed_items[property_name] += 1
                        items_yielded += 1
                        if items_yielded >= self._window_size:
                            #  If the window is full, break the inner loop, will also break the outer loop
                            #  since the items_yielded >= self._window_size, start the next window
                            break
                    except StopIteration:
                        # If no more workloads, this property is done
                        del active_iterators[property_name]

                if nothing_yielded_window:
                    break

    def _iterate_overall_mixture(self, active_iterators: dict[str, Iterator[str]]) -> Iterator[str]:
        """
        Iterate over the samples in the result index with an overall mixture. This function yields the samples
        in the overall correct mixture.
        """
        #  Shuffle the results to ensure that the order of the property combinations is (reproducibly) random
        property_names = list(active_iterators.keys())
        seed_everything(generate_hash_string_from_list(property_names))
        random.shuffle(property_names)

        while active_iterators:
            for property_name in property_names:
                if property_name in active_iterators:
                    try:
                        yield next(active_iterators[property_name])
                    except StopIteration:
                        del active_iterators[property_name]

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
