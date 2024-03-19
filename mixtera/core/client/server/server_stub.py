from typing import Callable, Generator, Type

from loguru import logger
from mixtera.core.client import MixteraClient
from mixtera.core.datacollection import PropertyType
from mixtera.core.datacollection.datasets import Dataset
from mixtera.core.datacollection.index import IndexType
from mixtera.core.processing.execution_mode import ExecutionMode
from mixtera.core.query import Query
from mixtera.network.connection import ServerConnection


class ServerStub(MixteraClient):

    def __init__(self, host: str, port: int) -> None:
        self._server_connection = ServerConnection(host, port)
        self._host = host
        self._port = port

    def register_dataset(
        self,
        identifier: str,
        loc: str,
        dtype: Type[Dataset],
        parsing_func: Callable[[str], str],
        metadata_parser_type: str,
    ) -> bool:
        raise NotImplementedError("This functionality is not implemented on the ServerStub yet.")

    def check_dataset_exists(self, identifier: str) -> bool:
        raise NotImplementedError("This functionality is not implemented on the ServerStub yet.")

    def list_datasets(self) -> list[str]:
        raise NotImplementedError("This functionality is not implemented on the ServerStub yet.")

    def remove_dataset(self, identifier: str) -> bool:
        raise NotImplementedError("This functionality is not implemented on the ServerStub yet.")

    def execute_query(self, query: Query, chunk_size: int) -> bool:
        if not self._server_connection.execute_query(query, chunk_size):
            logger.error("Could not register query at server!")
            return False

        logger.info(f"Registered query for job {query.job_id} at server!")

        return True

    def _stream_result_chunks(self, job_id: str) -> Generator[IndexType, None, None]:
        yield from self._server_connection._stream_result_chunks(job_id)

    def _get_result_metadata(
        self, job_id: str
    ) -> tuple[dict[int, Type[Dataset]], dict[int, Callable[[str], str]], dict[int, str]]:
        return self._server_connection._get_result_metadata(job_id)

    def is_remote(self) -> bool:
        return True

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
        raise NotImplementedError("This functionality is not implemented on the ServerStub yet.")
