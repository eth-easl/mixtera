import multiprocessing as mp
import random
import time
from queue import Empty
from typing import Callable, Iterator, Optional, Type

from loguru import logger
from mixtera.core.datacollection.datasets import Dataset
from mixtera.core.datacollection.index import ChunkerIndex
from mixtera.core.query.mixture import Mixture, StaticMixture
from mixtera.network.connection import ServerConnection
from mixtera.utils import from_pickled_dict, generate_hash_string_from_list, to_pickled_dict

# These parameters are relevant to the parallel reader in order to avoid timeouts
RETRY_COUNT = 5
READ_TIMEOUT_TIME = 0.1  # 100ms


class ResultChunk:
    def __init__(
        self,
        result_index: ChunkerIndex,
        dataset_type_dict: dict[int, Type[Dataset]],
        file_path_dict: dict[int, str],
        parsing_func_dict: dict[int, Callable[[str], str]],
        mixture: Optional[Mixture] = None,
    ) -> None:
        self._result_index = result_index
        self._dataset_type_dict = dataset_type_dict
        self._file_path_dict = file_path_dict
        self._parsing_func_dict = parsing_func_dict
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

        if self._per_window_mixture and self._window_size < 1:
            logger.warning(
                f"Window size is set to {self._window_size} which is invalid. " "Setting window size to 128."
            )
            self._window_size = 128

        if (self._per_window_mixture or self._degree_of_parallelism > 1) and (
            self._mixture is None or len(self._mixture) == 0
        ):
            logger.info("Mixture is not defined or empty but required. Infer mixture from the result index.")
            self._mixture = self._infer_mixture()

    def _infer_mixture(self) -> Mixture:
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

        return StaticMixture(total_count, partition_masses)

    def _iterate_result_chunks(self) -> Iterator[str]:
        if self._degree_of_parallelism < 1:
            logger.warning(
                f"Degree of parallelism is set to {self._degree_of_parallelism} which is invalid. "
                "Setting degree of parallelism to 1."
            )
            self._degree_of_parallelism = 1
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
        element_counts = self._get_element_counts()
        workloads: dict[str, list[tuple[int, int, list[tuple[int, int]]]]] = self._prepare_workloads()

        current_iterators = {
            property_name: iter(self._get_iterator_for_workload(property_workload))
            for property_name, property_workload in workloads.items()
        }

        processed_items = {property_name: 0 for property_name in workloads}

        while current_iterators:
            for property_name, property_count in element_counts.items():
                for _ in range(property_count):
                    if property_name not in current_iterators:
                        break
                    try:
                        instance = next(current_iterators[property_name])
                        yield instance
                        processed_items[property_name] += 1
                    except StopIteration:
                        # If no more workloads, this property is done
                        del current_iterators[property_name]
                        continue

    def _get_iterator_for_workload(self, workload: list[tuple[int, int, list]]) -> Iterator[str]:
        for document_id, file_id, ranges in workload:
            filename_dict = {self._file_path_dict[file_id]: ranges}
            yield from self._dataset_type_dict[document_id].read_ranges_from_files(
                filename_dict, self._parsing_func_dict[document_id], self._server_connection
            )

    def _iterate_single_threaded_overall_mixture(self) -> Iterator[str]:
        workloads: dict[str, list[tuple[int, int, list[tuple[int, int]]]]] = self._prepare_workloads()

        iterators = {
            property_name: self._get_iterator_for_workload(workload) for property_name, workload in workloads.items()
        }
        active_iterators = list(iterators.items())
        active_iterators.sort(key=lambda x: x[0])

        random.seed(generate_hash_string_from_list([x[0] for x in active_iterators]))
        random.shuffle(active_iterators)

        while active_iterators:
            for property_name, iterator in active_iterators:
                try:
                    yield next(iterator)
                except StopIteration:
                    active_iterators.remove((property_name, iterator))

    def _iterate_multi_threaded(self) -> Iterator[str]:
        if not isinstance(self._mixture, Mixture):
            raise ValueError("Mixture must be defined for parallel reading, this should not happen.")

        # Collect the workloads (i.e. did+fid+ranges) and group them by the property combination they belong to
        workloads: dict[str, list[tuple[int, int, list[tuple[int, int]]]]] = self._prepare_workloads()

        # Determine the number of readers to use s.t. readers are not overprovisioned
        reader_count = min(
            sum(len(x) for x in workloads.values()),
            self._degree_of_parallelism if self._degree_of_parallelism is not None else mp.cpu_count(),
        )

        # Determine how many processes should be assigned per property combination
        process_counts = {key: int(val * reader_count) for key, val in self._mixture.mixture_in_rows().items()}

        process_counts[list(process_counts.keys())[0]] += reader_count - sum(process_counts.values())

        processes: dict[str, list[tuple[mp.Queue, mp.Process]]] = self._spin_up_readers(workloads, process_counts)

        if self._per_window_mixture:
            yield_source = self._iterate_multi_threaded_window_mixture(processes)
        else:
            yield_source = self._iterate_multi_threaded_overall_mixture(processes)
        yield from yield_source

    def _get_element_counts(self) -> dict[str, int]:
        if not isinstance(self._mixture, Mixture):
            raise ValueError("Mixture must be defined for parallel reading, this should not happen.")

        # Determine the per-property combination batch counts
        element_counts = {key: int(self._window_size * value) for key, value in self._mixture.mixture_in_rows().items()}
        element_counts[list(element_counts.keys())[0]] += self._window_size - sum(element_counts.values())
        return element_counts

    def _iterate_multi_threaded_window_mixture(
        self,
        processes: dict[str, list[tuple[mp.Queue, mp.Process]]],
    ) -> Iterator[str]:
        element_counts = self._get_element_counts()

        continue_iterating = True
        while continue_iterating:  # pylint: disable=too-many-nested-blocks
            continue_iterating = False
            for property_name, property_count in element_counts.items():
                for _ in range(property_count):
                    yielded = False
                    retries = RETRY_COUNT
                    while not yielded and len(processes[property_name]) > 0 and retries > 0:
                        to_remove = []
                        for i, q_proc in enumerate(processes[property_name]):
                            q, proc = q_proc

                            # Mark processes that are dead and have empty queues; else try to fetch an instance
                            if not proc.is_alive() and q.empty():
                                to_remove.append(i)
                            else:
                                try:
                                    instance = q.get_nowait()
                                except Empty:
                                    pass
                                else:
                                    yield instance
                                    yielded = True
                                    break

                        # Remove dead processes with empty queues; do it in reverse to not affect other indexes
                        for i in reversed(to_remove):
                            del processes[property_name][i]

                        if not yielded and len(processes[property_name]) > 0:
                            time.sleep(READ_TIMEOUT_TIME)
                            retries -= 1

                    # If at least one instance could be read we should continue
                    continue_iterating = continue_iterating or yielded

    def _iterate_multi_threaded_overall_mixture(
        self,
        processes: dict[str, list[tuple[mp.Queue, mp.Process]]],
    ) -> Iterator[str]:
        paired_results = []

        for key, proc_list in processes.items():
            for q, proc in proc_list:
                proc.join()
                while not q.empty():
                    paired_results.append((key, q.get_nowait()))

        #  Sort the results by the key to ensure reproducibility
        paired_results.sort(key=lambda x: x[0])
        results = [x[1] for x in paired_results]

        random.seed(generate_hash_string_from_list([x[0] for x in paired_results]))
        random.shuffle(results)

        yield from results

    def _spin_up_readers(
        self,
        workloads: dict[str, list[tuple[int, int, list]]],
        process_counts: dict[str, int],
    ) -> dict[str, list[tuple[mp.Queue, mp.Process]]]:
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
                                to_pickled_dict(self._parsing_func_dict),
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
        pickled_parsing_func_dict: dict[int, bytes],
        server_connection: ServerConnection,
        workloads: list[tuple[int, int, list]],
    ) -> None:
        parsing_func_dict: dict[int, Callable[[str], str]] = from_pickled_dict(pickled_parsing_func_dict)
        for document_id, file_id, ranges in workloads:
            filename_dict = {file_path_dict[file_id]: ranges}
            instance_iterator = dataset_type_dict[document_id].read_ranges_from_files(
                filename_dict, parsing_func_dict[document_id], server_connection
            )
            for instance in instance_iterator:
                queue.put_nowait(instance)

        queue.close()

    def _prepare_workloads(self) -> dict[str, list[tuple[int, int, list[tuple[int, int]]]]]:
        workloads: dict[str, list[tuple[int, int, list[tuple[int, int]]]]] = {}
        for property_combination, document_entries in self._result_index.items():
            if property_combination not in workloads:
                workloads[property_combination] = []
            for document_id, file_entries in document_entries.items():
                for file_id, ranges in file_entries.items():
                    workloads[property_combination].append((document_id, file_id, ranges))
        return workloads

    def __iter__(self) -> "ResultChunk":
        self._iterator = self._iterate_result_chunks()
        return self

    def __next__(self) -> Iterator[str]:
        if self._iterator is None:
            raise StopIteration
        try:
            return next(self._iterator)  # type: ignore  # MyPy seems to expect SupportsNext[Iterator[str]] here
        except StopIteration:
            self._iterator = None
            raise
