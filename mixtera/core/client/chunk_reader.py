import multiprocessing as mp
import random
import time
from abc import ABC, abstractmethod
from enum import Enum
from queue import Empty
from typing import Any, Callable, Iterator, Optional, Type

from loguru import logger
from mixtera.core.datacollection.datasets import Dataset
from mixtera.core.datacollection.index import ChunkerIndex
from mixtera.core.query import Mixture, StaticMixture
from mixtera.network.connection import ServerConnection
from mixtera.utils import from_pickled_dict, generate_hash_string_from_list, to_pickled_dict

# These parameters are relevant to the parallel reader in order to avoid timeouts
RETRY_COUNT = 5
READ_TIMEOUT_TIME = 0.1  # 100ms


class ChunkReaderType(Enum):
    """Specifies the types of chunk readers"""

    PARALLEL = 1
    NON_PARALLEL = 2


class ChunkReader(ABC):
    def __init__(
        self,
        chunker_index: ChunkerIndex,
        dataset_type_dict: dict[int, Type[Dataset]],
        file_path_dict: dict[int, str],
        parsing_func_dict: dict[int, Callable[[str], str]],
        server_connection: ServerConnection,
        degree_of_parallelism: int = 1,
        mixture: Optional[Mixture] = None,
        per_window_mixture: bool = False,
        window_size: int = 128,
    ) -> None:
        """
        Initializer for a ChunkReader.

        Args:
            chunker_index: the ChunkerIndex object
            dataset_type_dict: A mapping from dataset ID to dataset type.
            file_path_dict: A mapping from file ID to file path.
            parsing_func_dict: A mapping from dataset ID to parsing function.
            server_connection: The server connection
        """
        self._chunker_index = chunker_index

        self._dataset_type_dict = dataset_type_dict
        self._file_path_dict = file_path_dict
        self._parsing_func_dict = parsing_func_dict
        self._server_connection = server_connection

        self._mixture = mixture
        self._degree_of_parallelism = degree_of_parallelism
        self._per_window_mixture = per_window_mixture
        self._window_size = window_size

    @abstractmethod
    def iterate_result_chunk(self) -> Iterator[Any]:
        """
        Orchestrates the reading of component chunks and yields instances such that if a mixture exists,
        each batch has the concentration indicated by the mixture. If no mixture is specified, then each batch should
        have a mixture proportional to the original components of the chunk (i.e. if the chunk has 300 instances from
        dataset A and 700 from B, then each batch will have a 3:7 ratio of A to B instances).

        Returns:
            Yields instances
        """
        raise NotImplementedError("This method must be implemented by the subclass!")


class StandardChunkReader(ChunkReader):
    """Implementation of a single-process chunk reader that yields instances on a FCFS basis."""

    def __init__(
        self,
        chunker_index: ChunkerIndex,
        dataset_type_dict: dict[int, Type[Dataset]],
        file_path_dict: dict[int, str],
        parsing_func_dict: dict[int, Callable[[str], str]],
        server_connection: ServerConnection,
        degree_of_parallelism: int = 1,
        mixture: Optional[Mixture] = None,
        per_window_mixture: bool = False,
        window_size: int = 128,
    ) -> None:
        """
        Initializer for a StandardChunkReader.

        Args:
            chunker_index: the ChunkerIndex object
            dataset_type_dict: A mapping from dataset ID to dataset type.
            file_path_dict: A mapping from file ID to file path.
            parsing_func_dict: A mapping from dataset ID to parsing function.
            server_connection: The server connection
            ensure_mixture: if True, ensures the underlying chunk Mixture by first reading all the data then shuffling
                it at random and yielding it
        """
        if degree_of_parallelism != 1:
            logger.warning("StandardChunkReader does not support parallelism; ignoring degree_of_parallelism!")

        super().__init__(
            chunker_index,
            dataset_type_dict=dataset_type_dict,
            file_path_dict=file_path_dict,
            parsing_func_dict=parsing_func_dict,
            server_connection=server_connection,
            degree_of_parallelism=degree_of_parallelism,
            mixture=mixture,
            per_window_mixture=per_window_mixture,
            window_size=window_size,
        )

    def _iterate_result_chunk_without_mixture(self) -> Iterator[Any]:
        """
        Iterate over the data, yielding instances one by one from the chunk in a FCFS order.
        """
        # for dataset_entries in self._chunker_index.values():
        #     for did, file_entries in dataset_entries.items():
        #         filename_dict = {
        #             self._file_path_dict[file_id]: file_ranges for file_id, file_ranges in file_entries.items()
        #         }
        #         yield from self._dataset_type_dict[did].read_ranges_from_files(
        #             filename_dict, self._parsing_func_dict[did], self._server_connection
        #         )

        entry_combinations = []
        for dataset_entries in self._chunker_index.values():
            for did, file_entries in dataset_entries.items():
                for file_id, file_ranges in file_entries.items():
                    filename = self._file_path_dict[file_id]
                    entry_combinations.append((did, {filename: file_ranges}))

        random.shuffle(entry_combinations)

        for did, filename_dict in entry_combinations:
            yield from self._dataset_type_dict[did].read_ranges_from_files(
                filename_dict, self._parsing_func_dict[did], self._server_connection
            )

    def _iterate_result_chunk_with_mixture(self) -> Iterator[Any]:
        """
        Reads the data once fully, the shuffles it to finally yield it. This happens in order to ensure the mixture.
        """
        read_instances: list[str] = []
        for dataset_entries in self._chunker_index.values():
            for did, file_entries in dataset_entries.items():
                filename_dict = {
                    self._file_path_dict[file_id]: file_ranges for file_id, file_ranges in file_entries.items()
                }
                read_instances.extend(
                    self._dataset_type_dict[did].read_ranges_from_files(
                        filename_dict, self._parsing_func_dict[did], self._server_connection
                    )
                )

        random.seed(generate_hash_string_from_list(read_instances))
        random.shuffle(read_instances)
        yield from read_instances

    def iterate_result_chunk(self) -> Iterator[Any]:
        if self._per_window_mixture:
            yield_source = self._iterate_result_chunk_with_mixture
        else:
            yield_source = self._iterate_result_chunk_without_mixture
        yield from yield_source()


class ParallelChunkReader(ChunkReader):
    """Implementation of a multiprocess based chunk reader that fulfills the requirements of a mixture."""

    def __init__(
        self,
        chunker_index: ChunkerIndex,
        dataset_type_dict: dict[int, Type[Dataset]],
        file_path_dict: dict[int, str],
        parsing_func_dict: dict[int, Callable[[str], str]],
        server_connection: ServerConnection,
        degree_of_parallelism: int = 1,
        mixture: Optional[Mixture] = None,
        per_window_mixture: bool = False,
        window_size: int = 128,
    ) -> None:
        """
        Initializer for a ChunkReader.

        Args:
            chunker_index: the ChunkerIndex object
            dataset_type_dict: A mapping from dataset ID to dataset type.
            file_path_dict: A mapping from file ID to file path.
            parsing_func_dict: A mapping from dataset ID to parsing function.
            server_connection: The server connection
            window_size: the size of a window within which the mixture should be maintained
            mixture: an optional parameter that specifies the mixture
            reader_count: the number of parallel readers. If None, it is tuned to the number of CPUs.
            per_window_mixture: if True, the mixture will be fulfilled within each window_size; this might make
                the reader slower due to the additional synchronization and ordering requirements
        """
        if degree_of_parallelism <= 1:
            logger.warning("ParallelChunkReader requires degree_of_parallelism > 1; setting it to 1!")
            degree_of_parallelism = 2

        super().__init__(
            chunker_index,
            dataset_type_dict=dataset_type_dict,
            file_path_dict=file_path_dict,
            parsing_func_dict=parsing_func_dict,
            server_connection=server_connection,
            degree_of_parallelism=degree_of_parallelism,
            mixture=mixture,
            per_window_mixture=per_window_mixture,
            window_size=window_size,
        )

        # If no mixture is provided, it needs to be inferred
        if self._mixture is None:
            total_count = 0
            partition_masses: dict[str, int | float] = {}
            for property_combination, partition_entry in self._chunker_index.items():
                count = 0
                for document_entry in partition_entry.values():
                    for ranges in document_entry.values():
                        for base_range in ranges:
                            count += base_range[1] - base_range[0]
                partition_masses[property_combination] = count
                total_count += count

            for key, _ in partition_masses.items():
                partition_masses[key] = partition_masses[key] / total_count

            self._mixture = StaticMixture(total_count, partition_masses)

        # Collect the workloads (i.e. did+fid+ranges) and group them by the property combination they belong to
        self._workloads: dict[str, list[tuple[int, int | str, list]]] = {}
        for property_combination, document_entries in self._chunker_index.items():
            if property_combination not in self._workloads:
                self._workloads[property_combination] = []
            for document_id, file_entries in document_entries.items():
                for file_id, ranges in file_entries.items():
                    self._workloads[property_combination].append((document_id, file_id, ranges))

        # Determine the per-property combination batch counts
        self._element_counts = {key: int(window_size * value) for key, value in self._mixture.mixture_in_rows().items()}
        self._element_counts[list(self._element_counts.keys())[0]] += window_size - sum(self._element_counts.values())

        # Determine the number of readers to use s.t. readers are not overprovisioned
        self.reader_count = min(
            sum(len(x) for x in self._workloads.values()),
            reader_count if reader_count is not None else mp.cpu_count(),
        )

        # Determine how many processes should be assigned per property combination
        self._process_counts = {
            key: int(val * self.reader_count) for key, val in self._mixture.mixture_in_rows().items()
        }
        self._process_counts[list(self._process_counts.keys())[0]] += self.reader_count - sum(
            self._process_counts.values()
        )

        # Spin up the processes
        self._processes: dict[str, list[tuple[mp.Queue, mp.Process]]] = {}
        for key, process_count in self._process_counts.items():
            self._processes[key] = []

            # Calculate per-process partition sizes
            partition_size = max(1, len(self._workloads[key]) // process_count)
            partition_ranges = list(range(0, len(self._workloads[key]), partition_size)) + [len(self._workloads[key])]

            # Create and start the processes
            for i in range(1, len(partition_ranges)):
                queue: mp.Queue = mp.Queue()
                self._processes[key].append(
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
                                self._workloads[key][partition_ranges[i - 1] : partition_ranges[i]],
                            ),
                        ),
                    )
                )

                # Start the process
                self._processes[key][-1][1].start()

    @staticmethod
    def _reader_process(
        queue: mp.Queue,
        dataset_type_dict: dict[int, Type[Dataset]],
        file_path_dict: dict[int, str],
        pickled_parsing_func_dict: dict[int, bytes],
        server_connection: ServerConnection,
        workloads: list[tuple[int, int | str, list]],
    ) -> None:
        parsing_func_dict = from_pickled_dict(pickled_parsing_func_dict)
        for document_id, file_id, ranges in workloads:
            filename_dict = {file_path_dict[file_id]: ranges}
            instance_iterator = dataset_type_dict[document_id].read_ranges_from_files(
                filename_dict, parsing_func_dict[document_id], server_connection
            )
            for instance in instance_iterator:
                queue.put_nowait(instance)

        queue.close()

    def _iterate_result_chunk_window_level_mixture(self) -> Iterator[Any]:
        """
        Iterates over the data produced by the parallel readers and tries to build a mixture-correct dataset on the fly.
        """
        continue_iterating = True
        while continue_iterating:  # pylint: disable=too-many-nested-blocks
            continue_iterating = False
            for property_name, property_count in self._element_counts.items():
                for _ in range(property_count):
                    yielded = False
                    retries = RETRY_COUNT
                    while not yielded and len(self._processes[property_name]) > 0 and retries > 0:
                        to_remove = []
                        for i, q_proc in enumerate(self._processes[property_name]):
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
                            del self._processes[property_name][i]

                        if not yielded and len(self._processes[property_name]) > 0:
                            time.sleep(READ_TIMEOUT_TIME)
                            retries -= 1

                    # If at least one instance could be read we should continue
                    continue_iterating = continue_iterating or yielded

    def _iterate_result_chunk_no_window_level(self) -> Iterator[Any]:
        """
        This method first waits for all the data to be read by the parallel readers before collecting it and yielding it
        """
        paired_results = []

        for key, proc_list in self._processes.items():
            for q, proc in proc_list:
                proc.join()
                while not q.empty():
                    paired_results.append((key, q.get_nowait()))

        # Sort the results by the key to ensure reproducibility
        paired_results.sort(key=lambda x: x[0])
        results = [x[1] for x in paired_results]

        random.seed(generate_hash_string_from_list(results))
        random.shuffle(results)
        yield from results

    def iterate_result_chunk(self) -> Iterator[Any]:
        if self._per_window_mixture:
            yield_source = self._iterate_result_chunk_window_level_mixture
        else:
            yield_source = self._iterate_result_chunk_no_window_level
        yield from yield_source()


class ChunkReaderFactory:
    @staticmethod
    def create_chunk_reader(
        reader_type: ChunkReaderType,
        chunker_index: ChunkerIndex,
        dataset_type_dict: dict[int, Type[Dataset]],
        file_path_dict: dict[int, str],
        parsing_func_dict: dict[int, Callable[[str], str]],
        server_connection: ServerConnection,
        **kwargs: Any,
    ) -> ChunkReader:
        """
        Creates a chunk reader of a given type.

        Args:
            reader_type: The type of the instantiated chunk reader
            chunker_index: the ChunkerIndex object
            dataset_type_dict: A mapping from dataset ID to dataset type.
            file_path_dict: A mapping from file ID to file path.
            parsing_func_dict: A mapping from dataset ID to parsing function.
            server_connection: The server connection
            **kwargs: Any additional keyword arguments for the chunk reader

        Returns:
            A chunk reader instance with the specified type
        """
        if reader_type == ChunkReaderType.PARALLEL:
            return ParallelChunkReader(
                chunker_index, dataset_type_dict, file_path_dict, parsing_func_dict, server_connection, **kwargs
            )
        if reader_type == ChunkReaderType.NON_PARALLEL:
            return StandardChunkReader(
                chunker_index, dataset_type_dict, file_path_dict, parsing_func_dict, server_connection, **kwargs
            )
        logger.error(f"Mixtera does not support chunk reader type {reader_type}!")
        raise NotImplementedError()
