import asyncio
from typing import TYPE_CHECKING, Any, Callable, Generator, Iterable, Optional

from loguru import logger
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
    from mixtera.core.datacollection.index import IndexType
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

    async def _connect_to_server(
        self, max_retries: int = 5, retry_delay: int = 1
    ) -> tuple[Optional[asyncio.StreamReader], Optional[asyncio.StreamWriter]]:
        for attempt in range(1, max_retries + 1):
            try:
                reader, writer = await asyncio.wait_for(asyncio.open_connection(self._host, self._port), timeout=5.0)
                return reader, writer
            except asyncio.TimeoutError:
                logger.error(f"Connection to {self._host}:{self._port} timed out (attempt {attempt}/{max_retries}).")
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.error(
                    "Failed to connect to"
                    + f"{self._host}:{self._port}. Is the server running? (attempt {attempt}/{max_retries}):{e}"
                )

            if attempt < max_retries:
                await asyncio.sleep(retry_delay)
            else:
                logger.error(
                    "Maximum number of connection attempts"
                    + f"({max_retries}) reached. Unable to connect to {self._host}:{self._port}."
                )

        return None, None

    async def _execute_query(self, query: "Query", chunk_size: int) -> bool:
        reader, writer = await self._connect_to_server()

        if reader is None or writer is None:
            return False

        # Announce we want to register a query
        await write_int(int(ServerTask.REGISTER_QUERY), ID_BYTES, writer)

        # Announce metadata
        await write_int(chunk_size, ID_BYTES, writer)

        # Announce query
        await write_pickeled_object(query, SAMPLE_SIZE_BYTES, writer)

        success = bool(await read_int(ID_BYTES, reader))
        logger.debug(f"Got success = {success} from server.")
        return success

    def execute_query(self, query: "Query", chunk_size: int) -> bool:
        success = run_async_until_complete(self._execute_query(query, chunk_size))
        return success

    async def _get_query_result_meta(self, training_id: str) -> Optional[dict]:
        reader, writer = await self._connect_to_server()

        if reader is None or writer is None:
            return None

        # Announce we want to get the query meta result
        await write_int(int(ServerTask.GET_META_RESULT), ID_BYTES, writer)

        # Announce training ID
        await write_utf8_string(training_id, ID_BYTES, writer)

        # Get meta object
        return await read_pickeled_object(SAMPLE_SIZE_BYTES, reader)

    # TODO(create issue): Use some ResultChunk type
    async def _get_next_result(self, training_id: str) -> Optional["IndexType"]:
        reader, writer = await self._connect_to_server()

        if reader is None or writer is None:
            return None

        # Announce we want to get a result chunk
        await write_int(int(ServerTask.GET_NEXT_RESULT_CHUNK), ID_BYTES, writer)

        # Announce training ID
        await write_utf8_string(training_id, ID_BYTES, writer)

        # Get meta object
        return await read_pickeled_object(SAMPLE_SIZE_BYTES, reader)

    def _stream_result_chunks(self, training_id: str) -> Generator["IndexType", None, None]:
        # TODO(create issue): We might want to prefetch here
        while (next_result := run_async_until_complete(self._get_next_result(training_id))) is not None:
            yield next_result

    def _get_result_metadata(
        self, training_id: str
    ) -> tuple[dict[int, Any], dict[int, Callable[[str], str]], dict[int, str]]:
        if (meta := run_async_until_complete(self._get_query_result_meta(training_id))) is None:
            raise RuntimeError("Error while fetching meta results")

        return meta["dataset_type"], meta["parsing_func"], meta["file_path"]
