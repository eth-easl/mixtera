from typing import Callable, Generator, Optional, Type
from mixtera.core.datacollection import IndexType
from mixtera.core.datacollection.datasets import Dataset
from mixtera.core.datacollection.local import LocalDataCollection
from mixtera.core.filesystem import AbstractFilesystem
from mixtera.network.connection import ServerConnection
from abc import ABC, abstractmethod

class QueryResult(ABC):
    @abstractmethod
    def __iter__(self) -> Generator[list[IndexType], None, None]:
        raise NotImplementedError()
    
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
        self.ldc = ldc
        self.chunk_size = chunk_size
        self._meta = self._parse_meta(results)
        self.results = results
        self._index = 0

    def _parse_meta(self, indices: list[IndexType]) -> dict:
        dataset_ids = set()
        file_ids = set()

        for idx in indices:
            dataset_ids.update(idx.keys())
            for val in idx.values():
                file_ids.update(val)

        return {
            "dataset_type": {did: self.ldc._get_dataset_type_by_id(did) for did in dataset_ids},
            "parsing_func": {did: self.ldc._get_dataset_func_by_id(did) for did in dataset_ids},
            "file_path": {fid: self.ldc._get_file_path_by_id(fid) for fid in file_ids},
            "dataset_fs": {did: self.ldc._get_dataset_filesys_by_id(did) for did in dataset_ids},
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
    
    def __iter__(self):
        return self
    
    def __next__(self):
        """Iterate over the results of the query with a chunk size.

        This method is very dummy right now without ensuring the correct mixture.
        """
        if self._index < len(self.results):
            next_chunk = self.results[self._index:self._index + self.chunk_size]
            self._index += self.chunk_size
            return next_chunk
        else:
            raise StopIteration


class RemoteQueryResult(QueryResult):
    def __init__(self, server_connection: ServerConnection, query_id: int):
        self._server_connection = server_connection
        self._query_id = query_id
        self._meta: Optional[dict] = None

    def _fetch_meta_if_none(self) -> None:
        if self._meta is None:
            if (meta := self._server_connection.get_query_result_meta(self._query_id)) is None:
                raise RuntimeError("Error while fetching meta results")
            
            self._meta = meta

    
    @property
    def dataset_type(self) -> dict[int, Type[Dataset]]:
        self._fetch_meta_if_none()
        return self._meta["dataset_type"]

    @property
    def file_path(self) -> dict[int, str]:
        self._fetch_meta_if_none()
        return self._meta["file_path"]

    @property
    def parsing_func(self) -> dict[int, Callable[[str], str]]:
        self._fetch_meta_if_none()
        return self._meta["parsing_func"]
    
    @property
    def dataset_fs(self) -> dict[int, Type[AbstractFilesystem]]:
        self._fetch_meta_if_none()
        return self._meta["dataset_fs"]
    
    def __iter__(self) -> Generator[list[IndexType], None, None]:
        yield from self._server_connection.get_query_results(self._query_id)