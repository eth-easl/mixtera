import multiprocessing as mp
from abc import ABC, abstractmethod
from typing import Any, Callable, Generator, Optional, Type

from loguru import logger
from mixtera.core.datacollection.datasets import Dataset
from mixtera.core.datacollection.index import IndexType
from mixtera.core.datacollection.index.index_collection import IndexFactory, IndexTypes, raw_index_dict_instantiator
from mixtera.core.datacollection.local import LocalDataCollection
from mixtera.network.connection import ServerConnection
from mixtera.utils import defaultdict_to_dict


class QueryResult(ABC):
    """QueryResult is a class that represents the results of a query.
    The QueryResult object is iterable and yields the results in chunks of size `chunk_size`.

    The QueryResult object also has three meta properties: `dataset_type`,
    `file_path` and `parsing_func`, each of which is a dictionary that maps
    dataset/file ids to their respective types, paths and parsing functions.
    """

    @abstractmethod
    def __next__(self) -> IndexType:
        raise NotImplementedError()

    def __iter__(self) -> "QueryResult":
        return self

    @property
    @abstractmethod
    def dataset_type(self) -> dict[int, Type[Dataset]]:
        raise NotImplementedError()

    @property
    @abstractmethod
    def file_path(self) -> dict[int, str]:
        raise NotImplementedError()

    @property
    @abstractmethod
    def parsing_func(self) -> dict[int, Callable[[str], str]]:
        raise NotImplementedError()


class LocalQueryResult(QueryResult):
    """LocalQueryResult is a class that represents the results of a query
    at the local machine.
    When constructing, it takes a list of indices (from the root of
    the query plan), a chunk size and a LocalDataCollection object.
    The LocalQueryResult object is iterable and yields the results in chunks of size `chunk_size`.
    The LocalQueryResult object also has three meta properties: `dataset_type`,
    `file_path` and `parsing_func`, each of which is a dictionary that maps
    dataset/file ids to their respective types, paths and parsing functions.
    """

    def __init__(self, ldc: LocalDataCollection, results: IndexType, chunk_size: int = 1) -> None:
        """
        Args:
            mdc (LocalDataCollection): The LocalDataCollection object.
            results (IndexType): The results of the query.
            chunk_size (int): The chunk size of the results.
        """
        self.chunk_size = chunk_size
        self.results = results
        self._meta = self._parse_meta(ldc)
        self._chunks: list[IndexType] = []
        self.chunking()
        # A process holding a LocalQueryResult might fork (e.g., for dataloaders).
        # Hence, we need to store the locks etc in shared memory.
        self._manager = mp.Manager()
        self._lock = self._manager.Lock()
        self._index = self._manager.Value("i", 0)
        # TODO(create issue): This is actually a big problem in case of multiple dataloaders.
        # Since we create new processes, the memory will get copied.
        # I tried to use a manager.List() but run into pickling errors.
        # However, this only affects the setting where we train without a MixteraServer.
        # It could be that we need defaultdict_to_dict here but I stopped exploring this for now.
        logger.debug(f"Instantiated LocalQueryResult with {len(self._chunks)} chunks.")
        # logger.debug([chunk._index for chunk in self._chunks])
        # logger.debug(self.results._index)

    def _parse_meta(self, ldc: LocalDataCollection) -> dict:
        dataset_ids = set()
        file_ids = set()

        total_length = 0
        for prop_name in self.results.get_all_features():
            for prop_val in self.results.get_dict_index_by_feature(prop_name).values():
                dataset_ids.update(prop_val.keys())
                for files in prop_val.values():
                    file_ids.update(files)
                    for file_ranges in files.values():
                        total_length += sum(end - start for start, end in file_ranges)

        return {
            "dataset_type": {did: ldc._get_dataset_type_by_id(did) for did in dataset_ids},
            "parsing_func": {did: ldc._get_dataset_func_by_id(did) for did in dataset_ids},
            "file_path": {fid: ldc._get_file_path_by_id(fid) for fid in file_ids},
            "total_length": total_length,
        }

    def chunking(self) -> None:
        # structure of self.chunks: [index_type, index_type, ...]
        # each index_type is an index, but contains exactly
        # chunk_size items, except the last one.
        chunks: list[dict] = []
        # here we chunk the self.results from a large index_type
        # into smaller index_types.
        current_chunk: IndexType = raw_index_dict_instantiator()
        current_chunk_length = 0
        # this is likely not a good impl, optimize it later.
        # now just ensure correct chunk_size and each chunk is an index.
        # pylint: disable-next=too-many-nested-blocks
        for property_name in self.results.get_all_features():
            for property_value, property_dict in self.results.get_dict_index_by_feature(property_name).items():
                for did, files in property_dict.items():
                    for fid, file_ranges in files.items():
                        for start, end in file_ranges:
                            # TODO(create issue): we may want to optimize this later together with mixture.
                            # for now this is a simple implementaion (for testing)
                            # case 1: the entire range fits into the current chunk, add it to the current chunk
                            if end - start < self.chunk_size - current_chunk_length:
                                current_chunk[property_name][property_value][did][fid].append((start, end))
                                current_chunk_length += end - start
                                # chunk is not full yet
                            # case 2: the entire range does not fit into the current chunk
                            else:
                                # # step 1: add the first part of the range to the current chunk, to fill it up
                                current_chunk[property_name][property_value][did][fid].append(
                                    (start, start + self.chunk_size - current_chunk_length)
                                )
                                # this time the chunk is full
                                chunks.append(current_chunk)
                                # step 2: calculate how many chunks are needed for the rest of the range
                                remaining_length = end - (start + self.chunk_size - current_chunk_length)
                                remaining_chunks = remaining_length // self.chunk_size
                                if remaining_chunks > 0:
                                    # step 3: add the remaining chunks
                                    for i in range(remaining_chunks):
                                        new_chunk = raw_index_dict_instantiator()
                                        new_chunk[property_name] = {
                                            property_value: {
                                                did: {
                                                    fid: [
                                                        (
                                                            start
                                                            + self.chunk_size
                                                            - current_chunk_length
                                                            + i * self.chunk_size,
                                                            start
                                                            + self.chunk_size
                                                            - current_chunk_length
                                                            + (i + 1) * self.chunk_size,
                                                        )
                                                    ]
                                                }
                                            }
                                        }
                                        chunks.append(new_chunk)
                                remaining_length = remaining_length % self.chunk_size
                                current_chunk = raw_index_dict_instantiator()
                                current_chunk_length = 0
                                start_new = start + remaining_chunks * self.chunk_size
                                if remaining_length > 0:
                                    current_chunk[property_name] = {
                                        property_value: {did: {fid: [(start_new, start_new + remaining_length)]}}
                                    }
                                    current_chunk_length += remaining_length
                        # reset current chunk after this file - we anyway need a new index for the next file
        if current_chunk_length > 0:
            chunks.append(current_chunk)
        for chunk in chunks:
            chunk_index = IndexFactory.create_index(IndexTypes.IN_MEMORY_DICT_RANGE)
            chunk_index._index = defaultdict_to_dict(chunk)
            self._chunks.append(chunk_index)

    @property
    def dataset_type(self) -> dict[int, Type[Dataset]]:
        return self._meta["dataset_type"]

    @property
    def file_path(self) -> dict[int, str]:
        return self._meta["file_path"]

    @property
    def parsing_func(self) -> dict[int, Callable[[str], str]]:
        return self._meta["parsing_func"]

    def __next__(self) -> IndexType:
        """Iterate over the results of the query with a chunk size thread-safe.
        This method is very dummy right now without ensuring the correct mixture.
        """
        local_index: Optional[int] = None
        with self._lock:
            if self._index.value < len(self._chunks):
                local_index = self._index.value
                self._index.value += 1
        # We exit the scope of the lock as early as possible.
        # For now the actual slicing does not need to be locked.

        if local_index is not None:
            return self._chunks[local_index]

        raise StopIteration


class RemoteQueryResult(QueryResult):
    def __init__(self, server_connection: ServerConnection, query_id: int):
        self._server_connection = server_connection
        self._query_id = query_id
        self._meta: dict[str, Any] = {}
        self._result_generator: Optional[Generator[IndexType, None, None]] = None

    def _fetch_meta_if_empty(self) -> None:
        if not self._meta:
            if (meta := self._server_connection.get_query_result_meta(self._query_id)) is None:
                raise RuntimeError("Error while fetching meta results")

            self._meta = meta

    @property
    def dataset_type(self) -> dict[int, Type[Dataset]]:
        self._fetch_meta_if_empty()
        return self._meta["dataset_type"]

    @property
    def file_path(self) -> dict[int, str]:
        self._fetch_meta_if_empty()
        return self._meta["file_path"]

    @property
    def parsing_func(self) -> dict[int, Callable[[str], str]]:
        self._fetch_meta_if_empty()
        return self._meta["parsing_func"]

    def _fetch_results_if_none(self) -> None:
        if self._result_generator is None:
            self._result_generator = self._server_connection.get_query_results(self._query_id)

    def __iter__(self) -> "RemoteQueryResult":
        self._fetch_results_if_none()
        return self

    def __next__(self) -> IndexType:
        if self._result_generator is None:
            raise StopIteration
        # This is thread safe in the sense that the server
        # handles each incoming request in a thread safe way
        # While worker 1 could send its request before worker 2,
        # worker 2 reads None, closes its generator, and then worker
        # 1 gets its data back, this is not a problem since we need
        # to consume all workers when in a multi data loader worker setting.

        try:
            return next(self._result_generator)
        except StopIteration:
            self._result_generator = None
            raise
