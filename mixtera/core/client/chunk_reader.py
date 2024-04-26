import multiprocessing as mp
from abc import ABC, abstractmethod
from typing import Any, Iterator, Optional

from mixtera.core.datacollection.index import ChunkerIndex
from mixtera.core.query import Mixture, NoopMixture


class ChunkReader(ABC):
    def __init__(self, chunker_index: ChunkerIndex, batch_size: int = 32, mixture: Optional[Mixture] = None) -> None:
        """
        Initializer for a ChunkReader.

        Args:
            chunker_index: the ChunkerIndex object
            batch_size: the size of a batch
            mixture: an optional parameter that specifies the mixture
        """
        self.batch_size = batch_size
        self._chunker_index = chunker_index
        self._mixture = mixture

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
        This methods orchestrates the reading of component chunks and yields instances such that if a mixture exists,
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
        batch_size: int = 32,
        mixture: Optional[Mixture] = None,
        reader_count: Optional[int] = None,
    ):
        """
        Initializer for a ChunkReader.

        Args:
            chunker_index: the ChunkerIndex object
            batch_size: the size of a batch
            mixture: an optional parameter that specifies the mixture
            reader_count: the number of parallel readers. If None, it is tuned to the number of CPUs.
        """
        super().__init__(chunker_index, batch_size=batch_size, mixture=mixture)

        workloads = []
        for property_combination, document_entries in self._chunker_index.items():
            for document_id, file_entries in document_entries.items():
                for file_id, ranges in file_entries.items():
                    workloads.append((property_combination, document_id, file_id, ranges))

        self.reader_count = min(len(workloads), reader_count if reader_count is not None else mp.cpu_count())

    def iterate_result_chunk(self) -> Iterator[Any]:
        pass
