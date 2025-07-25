import time
from pathlib import Path
from typing import Any, Callable, Generator, Type

from loguru import logger

from mixtera.core.client import MixteraClient
from mixtera.core.client.mixtera_client import QueryExecutionArgs
from mixtera.core.datacollection import PropertyType
from mixtera.core.datacollection.datasets import Dataset
from mixtera.core.datacollection.index.parser import MetadataParser
from mixtera.core.processing.execution_mode import ExecutionMode
from mixtera.core.query import Query, ResultChunk
from mixtera.network.client.client_feedback import ClientFeedback
from mixtera.network.connection import ServerConnection


class ServerStub(MixteraClient):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        if len(args) == 1 and isinstance(args[0], tuple):
            host, port = args[0]
        elif len(args) == 2 and isinstance(args[0], str) and isinstance(args[1], int):
            host, port = args
        elif "host" in kwargs and "port" in kwargs:
            host = kwargs["host"]
            port = kwargs["port"]
        else:
            raise ValueError(
                "Invalid arguments. Please provide a tuple (host, port), separate host and port arguments,"
                " or keyword arguments 'host' and 'port'."
            )

        self.server_connection = ServerConnection(host, port)
        self._host: str = host
        self._port: int = port

    def register_dataset(
        self,
        identifier: str,
        loc: str | Path,
        dtype: Type[Dataset],
        parsing_func: Callable[[str], str],
        metadata_parser_identifier: str,
    ) -> bool:
        if isinstance(loc, Path):
            loc = str(loc)

        return self.server_connection.register_dataset(
            identifier, loc, dtype.type, parsing_func, metadata_parser_identifier
        )

    def register_metadata_parser(
        self,
        identifier: str,
        parser: Type[MetadataParser],
    ) -> bool:
        return self.server_connection.register_metadata_parser(identifier, parser)

    def check_dataset_exists(self, identifier: str) -> bool:
        return self.server_connection.check_dataset_exists(identifier)

    def list_datasets(self) -> list[str]:
        return self.server_connection.list_datasets()

    def remove_dataset(self, identifier: str) -> bool:
        return self.server_connection.remove_dataset(identifier)

    def execute_query(self, query: Query, args: QueryExecutionArgs) -> bool:
        if not self.server_connection.execute_query(query, args):
            logger.error("Could not register query at server!")
            return False

        logger.info(f"Started query registration for job {query.job_id} at server!")

        return True

    def process_feedback(self, job_id: str, feedback: ClientFeedback) -> bool:
        if not self.server_connection.receive_feedback(job_id, feedback):
            logger.error("Could not send the message to the server!")
            return False

        logger.info("Sent the feedback to the server!")
        return True

    def _stream_result_chunks(
        self, job_id: str, dp_group_id: int, node_id: int, worker_id: int
    ) -> Generator[ResultChunk, None, None]:
        yield from self.server_connection._stream_result_chunks(job_id, dp_group_id, node_id, worker_id)

    def _get_result_metadata(
        self, job_id: str
    ) -> tuple[dict[int, Type[Dataset]], dict[int, Callable[[str], str]], dict[int, str]]:
        return self.server_connection.get_result_metadata(job_id)

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
        max_val: float = 1.0,
        num_buckets: int = 10,
        batch_size: int = 1,
        degree_of_parallelism: int = 1,
        data_only_on_primary: bool = True,
    ) -> bool:
        return self.server_connection.add_property(
            property_name,
            setup_func,
            calc_func,
            execution_mode,
            property_type,
            min_val=min_val,
            max_val=max_val,
            num_buckets=num_buckets,
            batch_size=batch_size,
            degree_of_parallelism=degree_of_parallelism,
            data_only_on_primary=data_only_on_primary,
        )

    def checkpoint(
        self, job_id: str, dp_group_id: int, node_id: int, worker_status: list[int], server: bool = False
    ) -> str:
        return self.server_connection.checkpoint(job_id, dp_group_id, node_id, worker_status)

    def checkpoint_completed(self, job_id: str, chkpnt_id: str, on_disk: bool) -> bool:
        return self.server_connection.checkpoint_completed(job_id, chkpnt_id, on_disk)

    def restore_checkpoint(self, job_id: str, chkpnt_id: str) -> None:
        return self.server_connection.restore_checkpoint(job_id, chkpnt_id)

    def wait_for_execution(self, job_id: str) -> bool:
        logger.info("Waiting for query execution at server to finish.")
        status = self.server_connection.check_query_exec_status(job_id)

        timeout_minutes = 30
        curr_time = 0
        while status == 0 and curr_time <= timeout_minutes * 60:
            time.sleep(1)
            status = self.server_connection.check_query_exec_status(job_id)
            curr_time += 1

        if status != 1:
            logger.error(f"Query execution failed with status {status}.")
            return False

        logger.info("Query execution finished.")
        return True
