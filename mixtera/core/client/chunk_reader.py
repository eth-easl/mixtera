import multiprocessing as mp
import time
from abc import ABC, abstractmethod
from queue import Empty
from typing import Any, Callable, Iterator, Optional, Type

from mixtera.core.datacollection.datasets import Dataset
from mixtera.core.datacollection.index import ChunkerIndex
from mixtera.core.query import Mixture, NoopMixture
from mixtera.network.connection import ServerConnection

# These parameters are relevant to the parallel reader in order to avoid timeouts
RETRY_COUNT = 5
READ_TIMEOUT_TIME = 0.1  # 100ms


class ChunkReader(ABC):
    def __init__(
        self,
        chunker_index: ChunkerIndex,
        dataset_type_dict: dict[int, Type[Dataset]],
        file_path_dict: dict[int, str],
        parsing_func_dict: dict[int, Callable[[str], str]],
        server_connection: ServerConnection,
        batch_size: int = 32,
        mixture: Optional[Mixture] = None,
    ) -> None:
        """
        Initializer for a ChunkReader.

        Args:
            chunker_index: the ChunkerIndex object
            dataset_type_dict: A mapping from dataset ID to dataset type.
            file_path_dict: A mapping from file ID to file path.
            parsing_func_dict: A mapping from dataset ID to parsing function.
            server_connection: The server connection
            batch_size: the size of a batch
            mixture: an optional parameter that specifies the mixture
        """
        self.batch_size = batch_size
        self._chunker_index = chunker_index
        self._mixture = mixture

        self._dataset_type_dict = dataset_type_dict
        self._file_path_dict = file_path_dict
        self._parsing_func_dict = parsing_func_dict
        self._server_connection = server_connection

        # If no mixture is provided, it needs to be inferred
        if self._mixture is None:
            total_count = 0
            partition_masses = {}
            for property_combination, partition_entry in self._chunker_index.items():
                count = 0
                for _0, document_entry in partition_entry.items():
                    for _1, ranges in document_entry.items():
                        for base_range in ranges:
                            count += base_range[1] - base_range[0]
                partition_masses[property_combination] = count
                total_count += count

            for key in partition_masses.keys():
                partition_masses[key] = partition_masses[key] / total_count

            self._mixture = NoopMixture(total_count, partition_masses)

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


class ParallelChunkReader(ChunkReader):

    def __init__(
        self,
        chunker_index: ChunkerIndex,
        dataset_type_dict: dict[int, Type[Dataset]],
        file_path_dict: dict[int, str],
        parsing_func_dict: dict[int, Callable[[str], str]],
        server_connection: ServerConnection,
        batch_size: int = 32,
        mixture: Optional[Mixture] = None,
        reader_count: Optional[int] = None,
    ):
        """
        Initializer for a ChunkReader.

        Args:
            chunker_index: the ChunkerIndex object
            dataset_type_dict: A mapping from dataset ID to dataset type.
            file_path_dict: A mapping from file ID to file path.
            parsing_func_dict: A mapping from dataset ID to parsing function.
            server_connection: The server connection
            batch_size: the size of a batch
            mixture: an optional parameter that specifies the mixture
            reader_count: the number of parallel readers. If None, it is tuned to the number of CPUs.
        """
        super().__init__(
            chunker_index,
            dataset_type_dict=dataset_type_dict,
            file_path_dict=file_path_dict,
            parsing_func_dict=parsing_func_dict,
            server_connection=server_connection,
            batch_size=batch_size,
            mixture=mixture,
        )

        # Collect the workloads (i.e. did+fid+ranges) and group them by the property combination they belong to
        self._workloads = {}
        for property_combination, document_entries in self._chunker_index.items():
            if property_combination not in self._workloads:
                self._workloads[property_combination] = []
            for document_id, file_entries in document_entries.items():
                for file_id, ranges in file_entries.items():
                    self._workloads[property_combination].append((document_id, file_id, ranges))

        # Determine the per-property combination batch counts
        self._batch_counts = {key: int(batch_size * value) for key, value in self._mixture.get_raw_mixture().items()}
        self._batch_counts[list(self._batch_counts.keys())[0]] += batch_size - sum(self._batch_counts.values())

        # Determine the number of readers to use s.t. readers are not overprovisioned
        self.reader_count = min(
            sum([len(x) for x in self._workloads.values()]),
            reader_count if reader_count is not None else mp.cpu_count(),
        )

        # Determine how many processes should be assigned per property combination
        self._process_counts = {
            key: int(val * self.reader_count) for key, val in self._mixture.get_raw_mixture().items()
        }
        self._process_counts[list(self._process_counts.keys())[0]] += self.reader_count - sum(
            self._process_counts.values()
        )

        # Spin up the processes
        self._processes = {}
        for key, process_count in self._process_counts.items():
            self._processes[key] = []

            # Calculate per-process partition sizes
            partition_size = max(1, len(self._workloads[key]) // process_count)
            partition_ranges = list(range(0, len(self._workloads[key]), partition_size)) + [len(self._workloads[key])]
            assert process_count + 1 == len(
                partition_ranges
            ), f"Number of partitions: expected: {process_count + 1}; received: {len(partition_ranges)}"

            # Create and start the processes
            for i in range(1, len(partition_ranges)):
                queue = mp.Queue()
                self._processes[key].append(
                    (
                        queue,
                        mp.Process(
                            target=self._reader_process,
                            args=(
                                queue,
                                self._dataset_type_dict,
                                self._file_path_dict,
                                self._parsing_func_dict,
                                self._server_connection,
                                self._workloads[key][partition_ranges[i - 1] : partition_ranges[i]],
                            ),
                        ),
                    )
                )

                # Start the process
                self._processes[key][-1][1].start()

    @staticmethod
    def _reader_process(*payload) -> None:
        queue, dataset_type_dict, file_path_dict, parsing_func_dict, server_connection, workloads = payload

        for document_id, file_id, ranges in workloads:
            filename_dict = {file_path_dict[file_id]: ranges}
            instance_iterator = dataset_type_dict[document_id].read_ranges_from_files(
                filename_dict, parsing_func_dict[document_id], server_connection
            )
            for instance in instance_iterator:
                queue.put_nowait(instance)

    def iterate_result_chunk(self) -> Iterator[Any]:
        continue_iterating = True
        while continue_iterating:
            continue_iterating = False
            for property_name, property_count in self._batch_counts.items():
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
                    continue_iterating = continue_iterating and yielded
