import multiprocessing as mp
from abc import ABC, abstractmethod
from typing import Any, Callable, Generator, Optional, Type

from loguru import logger
from mixtera.core.datacollection import IndexType
from mixtera.core.datacollection.datasets import Dataset
from mixtera.core.datacollection.local import LocalDataCollection
from mixtera.core.filesystem import AbstractFilesystem
from mixtera.network.connection import ServerConnection


class QueryResult(ABC):
    @abstractmethod
    def __next__(self) -> list[IndexType]:
        raise NotImplementedError()

    def __iter__(self) -> "QueryResult":
        return self

    @property
    @abstractmethod
    def dataset_fs(self) -> dict[int, Type[AbstractFilesystem]]:
        raise NotImplementedError()

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
    """QueryResult is a class that represents the results of a query.
    When constructing, it takes a list of indices (from the root of
    the query plan), a chunk size and a MixteraDataCollection object.

    The QueryResult object is iterable and yields the results in chunks of size `chunk_size`.

    The QueryResult object also has three meta properties: `dataset_type`,
    `file_path` and `parsing_func`, each of which is a dictionary that maps
    dataset/file ids to their respective types, paths and parsing functions.
    """

    def __init__(self, ldc: LocalDataCollection, results: list[IndexType], chunk_size: int = 1) -> None:
        """
        Args:
            mdc (MixteraDataCollection): The MixteraDataCollection object.
            results (list): The list of results of the query.
            chunk_size (int): The chunk size of the results.
        """
        self.chunk_size = chunk_size
        self._meta = self._parse_meta(ldc, results)
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
        self.results = results

        logger.debug(f"Instantiated LocalQueryResult with {len(self.results)} IndexTypes.")

    def _parse_meta(self, ldc: LocalDataCollection, indices: list[IndexType]) -> dict:
        dataset_ids = set()
        file_ids = set()

        for idx in indices:
            dataset_ids.update(idx.keys())
            for val in idx.values():
                file_ids.update(val)

        return {
            "dataset_type": {did: ldc._get_dataset_type_by_id(did) for did in dataset_ids},
            "parsing_func": {did: ldc._get_dataset_func_by_id(did) for did in dataset_ids},
            "file_path": {fid: ldc._get_file_path_by_id(fid) for fid in file_ids},
            "dataset_fs": {did: ldc._get_dataset_filesys_by_id(did) for did in dataset_ids},
        }

    @property
    def dataset_type(self) -> dict[int, Type[Dataset]]:
        return self._meta["dataset_type"]

    @property
    def file_path(self) -> dict[int, str]:
        return self._meta["file_path"]

    @property
    def parsing_func(self) -> dict[int, Callable[[str], str]]:
        return self._meta["parsing_func"]

    @property
    def dataset_fs(self) -> dict[int, Type[AbstractFilesystem]]:
        return self._meta["dataset_fs"]

    def __next__(self) -> list[IndexType]:
        """Iterate over the results of the query with a chunk size thread-safe.

        This method is very dummy right now without ensuring the correct mixture.
        """
        local_index: Optional[int] = None
        with self._lock:
            if self._index.value < len(self.results):
                local_index = self._index.value
                self._index.value += self.chunk_size
        # We exit the scope of the lock as early as possible.
        # For now the actual slicing does not need to be locked.

        if local_index is not None:
            return self.results[local_index : local_index + self.chunk_size]

        raise StopIteration


class RemoteQueryResult(QueryResult):
    def __init__(self, server_connection: ServerConnection, query_id: int):
        self._server_connection = server_connection
        self._query_id = query_id
        self._meta: dict[str, Any] = {}
        self._result_generator: Optional[Generator[list[IndexType], None, None]] = None

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

    @property
    def dataset_fs(self) -> dict[int, Type[AbstractFilesystem]]:
        self._fetch_meta_if_empty()
        return self._meta["dataset_fs"]

    def _fetch_results_if_none(self) -> None:
        if self._result_generator is None:
            self._result_generator = self._server_connection.get_query_results(self._query_id)

    def __iter__(self) -> "RemoteQueryResult":
        self._fetch_results_if_none()
        return self

    def __next__(self) -> list[IndexType]:
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
