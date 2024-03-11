import asyncio
from typing import TYPE_CHECKING, Generator, Iterable, Optional

from loguru import logger
from mixtera.core.datacollection.index import IndexType
from mixtera.network import ID_BYTES, SAMPLE_SIZE_BYTES
from mixtera.network.network_utils import (
    read_int,
    read_pickeled_object,
    read_utf8_string,
    write_int,
    write_pickeled_object,
    write_utf8_string,
)
from mixtera.network.server_task import ServerTask
from mixtera.utils import run_async_until_complete

if TYPE_CHECKING:
    from mixtera.core.query import Query


class ServerConnection:
    def __init__(self, host: str, port: int) -> None:
        self._host = host
        self._port = port

    async def _fetch_file(self, file_path: str) -> Optional[str]:
        reader, writer = await self._connect_to_server()

        if reader is None or writer is None:
            return None

        await write_int(int(ServerTask.READ_FILE), ID_BYTES, writer)
        await write_utf8_string(file_path, ID_BYTES, writer)

        return await read_utf8_string(SAMPLE_SIZE_BYTES, reader)

    def get_file_iterable(self, file_path: str) -> Iterable[str]:
        if (lines := run_async_until_complete(self._fetch_file(file_path))) is None:
            return

        yield from lines.split("\n")

    async def _connect_to_server(self) -> tuple[Optional[asyncio.StreamReader], Optional[asyncio.StreamWriter]]:
        try:
            logger.debug("Connecting to server.")
            reader, writer = await asyncio.wait_for(asyncio.open_connection(self._host, self._port), timeout=5.0)
        except asyncio.TimeoutError:
            logger.error(f"Connection to {self._host}:{self._port} timed out.")
            return None, None
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error(f"Failed to connect to {self._host}:{self._port}. Is the server running?: {e}")
            return None, None

        logger.debug("Connected!")

        return reader, writer

    async def _execute_query(self, query: "Query", chunk_size: int) -> int:
        reader, writer = await self._connect_to_server()

        if reader is None or writer is None:
            return -1

        # Announce we want to register a query
        await write_int(int(ServerTask.REGISTER_QUERY), ID_BYTES, writer)

        # Announce metadata
        await write_int(chunk_size, ID_BYTES, writer)

        # Announce query
        await write_pickeled_object(query, SAMPLE_SIZE_BYTES, writer)

        query_id = await read_int(ID_BYTES, reader)
        logger.debug(f"Got query id {query_id} from server.")
        return query_id

    def execute_query(self, query: "Query", chunk_size: int) -> int:
        query_id = run_async_until_complete(self._execute_query(query, chunk_size))
        return query_id

    async def _get_query_id(self, training_id: str) -> int:
        reader, writer = await self._connect_to_server()

        if reader is None or writer is None:
            return -1

        # Announce we want to get the query id
        await write_int(int(ServerTask.GET_QUERY_ID), ID_BYTES, writer)

        # Announce training ID
        await write_utf8_string(training_id, SAMPLE_SIZE_BYTES, writer)
        query_id = await read_int(ID_BYTES, reader, timeout=500)
        logger.debug(f"Got query id {query_id} from server.")
        return query_id

    def get_query_id(self, training_id: str) -> int:
        return run_async_until_complete(self._get_query_id(training_id))

    async def _get_query_result_meta(self, query_id: int) -> Optional[dict]:
        reader, writer = await self._connect_to_server()

        if reader is None or writer is None:
            return None

        # Announce we want to get the query meta result
        await write_int(int(ServerTask.GET_META_RESULT), ID_BYTES, writer)

        # Announce query ID
        await write_int(query_id, ID_BYTES, writer)

        # Get meta object
        return await read_pickeled_object(SAMPLE_SIZE_BYTES, reader)

    def get_query_result_meta(self, query_id: int) -> Optional[dict]:
        return run_async_until_complete(self._get_query_result_meta(query_id))

    # TODO(create issue): Use some ResultChunk type
    async def _get_next_result(self, query_id: int) -> Optional[list[IndexType]]:
        logger.debug(f"Obtaining next result chunk for query_id {query_id}")
        reader, writer = await self._connect_to_server()

        if reader is None or writer is None:
            return None

        # Announce we want to get a result chunk
        await write_int(int(ServerTask.GET_NEXT_RESULT_CHUNK), ID_BYTES, writer)

        # Announce query ID
        await write_int(query_id, ID_BYTES, writer)

        # Get meta object
        return await read_pickeled_object(SAMPLE_SIZE_BYTES, reader)

    def get_query_results(self, query_id: int) -> Generator[list[IndexType], None, None]:
        # TODO(create issue): We might want to prefetch here
        while (next_result := run_async_until_complete(self._get_next_result(query_id))) is not None:
            yield next_result

        logger.debug("End of query results stream.")
