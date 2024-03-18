import multiprocessing as mp
from pathlib import Path
from typing import Callable, Generator, Type

from loguru import logger
from mixtera.core.client import MixteraClient
from mixtera.core.datacollection import MixteraDataCollection, PropertyType
from mixtera.core.datacollection.datasets import Dataset
from mixtera.core.datacollection.index.index import IndexType
from mixtera.core.processing import ExecutionMode
from mixtera.core.query import Query, QueryResult
from mixtera.utils import wait_for_key_in_dict


class LocalStub(MixteraClient):
    def __init__(self, directory: Path | str) -> None:
        if isinstance(directory, str):
            self.directory = Path(directory)
        else:
            self.directory = directory

        if not self.directory.exists():
            raise RuntimeError(f"Directory {self.directory} does not exist.")

        self._mdc = MixteraDataCollection(self.directory)
        self._training_query_map_lock = mp.Lock()
        self._training_query_map: dict[str, tuple[Query, int]] = {}  # (query, chunk_size)

    def register_dataset(
        self,
        identifier: str,
        loc: str,
        dtype: Type[Dataset],
        parsing_func: Callable[[str], str],
        metadata_parser_type: str,
    ) -> bool:
        return self._mdc.register_dataset(identifier, loc, dtype, parsing_func, metadata_parser_type)

    def check_dataset_exists(self, identifier: str) -> bool:
        return self._mdc.check_dataset_exists(identifier)

    def list_datasets(self) -> list[str]:
        return self._mdc.list_datasets()

    def remove_dataset(self, identifier: str) -> bool:
        return self._mdc.remove_dataset(identifier)

    def execute_query(self, query: Query, chunk_size: int) -> bool:
        query.execute(self._mdc, chunk_size=chunk_size)
        return self._register_query(query, chunk_size)

    def is_remote(self) -> bool:
        return False

    def add_property(
        self,
        property_name: str,
        setup_func: Callable,
        calc_func: Callable,
        execution_mode: ExecutionMode,
        property_type: "PropertyType",
        min_val: float = 0.0,
        max_val: float = 1,
        num_buckets: int = 10,
        batch_size: int = 1,
        dop: int = 1,
        data_only_on_primary: bool = True,
    ) -> None:
        return self._mdc.add_property(
            property_name,
            setup_func,
            calc_func,
            execution_mode,
            property_type,
            min_val=min_val,
            max_val=max_val,
            num_buckets=num_buckets,
            batch_size=batch_size,
            dop=dop,
            data_only_on_primary=data_only_on_primary,
        )

    def _stream_result_chunks(self, training_id: str) -> Generator[IndexType, None, None]:
        query_result = self._get_query_result(training_id)
        yield from query_result

    def _get_result_metadata(
        self, training_id: str
    ) -> tuple[dict[int, Type[Dataset]], dict[int, Callable[[str], str]], dict[int, str]]:
        query_result = self._get_query_result(training_id)
        return query_result.dataset_type, query_result.parsing_func, query_result.file_path

    def _register_query(self, query: "Query", chunk_size: int) -> bool:
        if query.training_id in self._training_query_map:
            logger.warning(f"We already have a query for training {query.training_id}!")
            return False

        with self._training_query_map_lock:
            self._training_query_map[query.training_id] = (query, chunk_size)

        logger.info(
            f"Registered query {str(query)} with chunk_size {chunk_size}" + f" for training {query.training_id}."
        )

        return True

    def _get_query_result(self, training_id: str) -> QueryResult:
        if not wait_for_key_in_dict(self._training_query_map, training_id, 60.0):
            raise RuntimeError(f"Unknown training {training_id}")
        # Since queries are only registered after they are executed, results is guaranteed to not be None
        return self._training_query_map[training_id][0].results
