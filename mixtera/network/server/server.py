import asyncio
from pathlib import Path

from torch import Generator

from loguru import logger
from mixtera.core.datacollection import MixteraDataCollection
from mixtera.core.filesystem.filesystem import FileSystem
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
from mixtera.core.client.local  import LocalStub
import multiprocessing as mp

class MixteraServer:
    def __init__(self, directory: Path, host: str, port: int):
        self._host = host
        self._port = port
        self._directory = directory
        self._local_stub = LocalStub(self._directory)
        self._result_chunk_generator_map: dict[str, Generator[str, None, None]] = {}
        self._result_chunk_generator_map_lock = mp.Lock()

    async def _register_query(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        logger.debug("Received register training request")
        chunk_size = await read_int(ID_BYTES, reader)
        logger.debug(f"chunk_size = {chunk_size}")
        query = await read_pickeled_object(SAMPLE_SIZE_BYTES, reader)
        logger.debug(f"Received query = {str(query)}. Executing it.")
        success = self._local_stub.execute_query(query, chunk_size)
        logger.debug(f"Registered query with success = {success} and executed it.")

        await write_int(int(success), ID_BYTES, writer)

    async def _read_file(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        file_path = await read_utf8_string(ID_BYTES, reader)

        if file_path is None or file_path == "":
            logger.warning("Did not receive file path.")
            return

        file_data = "".join(FileSystem.from_path(file_path).get_file_iterable(file_path))
        await write_utf8_string(file_data, SAMPLE_SIZE_BYTES, writer, drain=False)
        await writer.drain()

    async def _return_next_result_chunk(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        training_id = await read_utf8_string(ID_BYTES, reader)
        with self._result_chunk_generator_map_lock:
            if training_id not in self._result_chunk_generator_map_lock:
                self._result_chunk_generator_map_lock[training_id] = self._local_stub._get_query_result(training_id)

        next_chunk = next(self._result_chunk_generator_map_lock[training_id], None)
        await write_pickeled_object(next_chunk, SAMPLE_SIZE_BYTES, writer)

    async def _return_result_metadata(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        training_id = await read_utf8_string(ID_BYTES, reader)
        dataset_dict, parsing_dict, file_path_dict = self._local_stub._get_result_metadata(training_id)

        meta = {
            "dataset_type": dataset_dict,
            "parsing_func": parsing_dict,
            "file_path": file_path_dict,
        }
        await write_pickeled_object(meta, SAMPLE_SIZE_BYTES, writer)


    async def _dispatch_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            if (task_int := await read_int(ID_BYTES, reader)) not in ServerTask.__members__.values():
                raise RuntimeError(f"Unknown task id: {task_int}")

            task = ServerTask(task_int)
            if task == ServerTask.REGISTER_QUERY:
                await self._register_query(reader, writer)
            elif task == ServerTask.READ_FILE:
                await self._read_file(reader, writer)
            elif task == ServerTask.GET_META_RESULT:
                await self._return_result_metadata(reader, writer)
            elif task == ServerTask.GET_NEXT_RESULT_CHUNK:
                await self._return_next_result_chunk(reader, writer)
            else:
                logger.error(f"Client sent unsupport task {task}")

        except asyncio.CancelledError:
            logger.error("asyncio.CancelledError")
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error(f"Some exception occured while handling client request: {e}")
            logger.exception(e)
        finally:
            try:
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
