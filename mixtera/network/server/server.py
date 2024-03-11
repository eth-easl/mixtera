import asyncio
from pathlib import Path

from loguru import logger
from mixtera.core.datacollection.local import LocalDataCollection
from mixtera.core.filesystem import AbstractFilesystem
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

# TODO(#): Use actual query instead of dict of ranges
QueryType = dict[int, dict[int, list[tuple[int, int]]]]


class MixteraServer:
    def __init__(self, directory: Path, host: str, port: int):
        self._ldc = LocalDataCollection(directory)
        self._host = host
        self._port = port
        self._directory = directory

    async def _register_query(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        logger.debug("Received register training request")
        chunk_size = await read_int(ID_BYTES, reader)
        logger.debug(f"chunk_size = {chunk_size}")
        query = await read_pickeled_object(SAMPLE_SIZE_BYTES, reader)
        logger.debug(f"Received query = {str(query)}. Executing it.")
        _ = query.execute(self._ldc)
        logger.debug(f"Registered query under ID {query.query_id} in LDC and executed it.")

        await write_int(query.query_id, ID_BYTES, writer)

    async def _parse_ids(self, reader: asyncio.StreamReader) -> tuple[int, int, int]:
        query_id = await read_int(ID_BYTES, reader)
        node_id = await read_int(ID_BYTES, reader)
        worker_id = await read_int(ID_BYTES, reader)

        logger.info(f"Worker {worker_id} for node {node_id} has connected for query {query_id}.")

        return node_id, worker_id, query_id

    async def _get_query_id(self, reader: asyncio.StreamReader) -> int:
        training_id = await read_utf8_string(SAMPLE_SIZE_BYTES, reader)
        logger.debug(f"Looking up query ID for training {training_id}")
        return self._ldc.get_query_id(training_id)

    async def _read_file(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        filesys_t = await read_int(ID_BYTES, reader)

        valid_input = True

        if filesys_t is None:
            logger.warning("Did not receive filesystem type.")
            valid_input = False

        file_path = await read_utf8_string(ID_BYTES, reader)

        if file_path is None or file_path == "":
            logger.warning("Did not receive file path.")
            valid_input = False

        if not valid_input:
            return

        logger.info(f"Got a _read_file request for file {file_path}")
        file_data = "".join(AbstractFilesystem.from_id(filesys_t).get_file_iterable(file_path, None))
        logger.debug("File read.")
        await write_utf8_string(file_data, SAMPLE_SIZE_BYTES, writer, drain=False)
        logger.debug("Data written.")
        await writer.drain()
        logger.debug("Data drained.")

    async def _return_next_result_chunk(self, query_id: int, writer: asyncio.StreamWriter) -> None:
        next_chunk = self._ldc.next_query_result_chunk(query_id)  # This function is thread safe
        await write_pickeled_object(next_chunk, SAMPLE_SIZE_BYTES, writer)

    async def _dispatch_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            if (task_int := await read_int(ID_BYTES, reader)) not in ServerTask.__members__.values():
                raise RuntimeError(f"Unknown task id: {task_int}")

            task = ServerTask(task_int)

            if task == ServerTask.REGISTER_QUERY:
                await self._register_query(reader, writer)
            elif task == ServerTask.READ_FILE:
                await self._read_file(reader, writer)
            elif task == ServerTask.GET_QUERY_ID:
                query_id = await self._get_query_id(reader)
                await write_int(query_id, ID_BYTES, writer)
            elif task == ServerTask.GET_META_RESULT:
                query_id = await read_int(ID_BYTES, reader)
                await write_pickeled_object(self._ldc._queries[query_id][0].results._meta, SAMPLE_SIZE_BYTES, writer)
            elif task == ServerTask.GET_NEXT_RESULT_CHUNK:
                query_id = await read_int(ID_BYTES, reader)
                await self._return_next_result_chunk(query_id, writer)
            else:
                logger.error(f"Client sent unsupport task {task}")

        except asyncio.CancelledError:
            logger.error("asyncio.CancelledError")
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error(f"Some exception occured while handling client request: {e}")
            logger.exception(e)
        finally:
            try:
                logger.info("Closing writer...")
                writer.close()
                await writer.wait_closed()
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.error(f"Error while closing writer: {e}")
                logger.exception(e)

    async def _run_async(self) -> None:
        server = await asyncio.start_server(self._dispatch_client, self._host, self._port)
        addr = server.sockets[0].getsockname()
        logger.info(f"Serving MixteraServer on {addr}")

        async with server:
            try:
                await server.serve_forever()
            except asyncio.CancelledError:
                logger.info("Received cancellation request for server.")
            finally:
                logger.info("Cleaning up.")
                server.close()
                await server.wait_closed()
                logger.info("Server has been stopped.")

    def run(self) -> None:
        asyncio.run(self._run_async())
