import asyncio
from pathlib import Path
from typing import Generator

from loguru import logger
from mixtera.core.client.local import LocalStub
from mixtera.core.filesystem.filesystem import FileSystem
from mixtera.network import NUM_BYTES_FOR_IDENTIFIERS, NUM_BYTES_FOR_SIZES
from mixtera.network.network_utils import (
    read_int,
    read_pickeled_object,
    read_utf8_string,
    write_int,
    write_pickeled_object,
    write_utf8_string,
)
from mixtera.network.server_task import ServerTask


class MixteraServer:
    def __init__(self, directory: Path, host: str, port: int):
        """
        Initializes the MixteraServer with a given directory, host, and port.

        The server uses the provided directory to initialize a LocalStub which
        executes queries and generates results. It listens on the given host and port.

        Args:
            directory (Path): The directory where Mixtera stores its metadata files.
            host (str): The host address on which the server will listen.
            port (int): The port on which the server will accept connections.

        """
        self._host = host
        self._port = port
        self._directory = directory
        self._local_stub: LocalStub = LocalStub(self._directory)
        self._result_chunk_generator_map: dict[str, Generator[str, None, None]] = {}
        self._result_chunk_generator_map_lock = asyncio.Lock()
        self._register_query_lock = asyncio.Lock()

    async def _register_query(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """
        Registers and executes a query received from the client.

        This method reads the query and its chunk size from the client,
        executes the query via the LocalStub, and writes back the success status.

        Args:
            reader (asyncio.StreamReader): The stream reader to read data from the client.
            writer (asyncio.StreamWriter): The stream writer to write data to the client.
        """
        logger.debug("Received register query request")
        chunk_size = await read_int(NUM_BYTES_FOR_IDENTIFIERS, reader)
        logger.debug(f"chunk_size = {chunk_size}")
        query = await read_pickeled_object(NUM_BYTES_FOR_SIZES, reader)
        logger.debug(f"Received query = {str(query)}. Executing it.")
        async with self._register_query_lock:
            success = self._local_stub.execute_query(query, chunk_size)
        logger.debug(f"Registered query with success = {success} and executed it.")

        await write_int(int(success), NUM_BYTES_FOR_IDENTIFIERS, writer)

    async def _read_file(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """
        Reads a file from the server's file system and sends its contents to the client.

        Args:
            reader (asyncio.StreamReader): The stream reader to read data from the client.
            writer (asyncio.StreamWriter): The stream writer to write data to the client.
        """
        file_path = await read_utf8_string(NUM_BYTES_FOR_IDENTIFIERS, reader)

        if file_path is None or file_path == "":
            logger.warning("Did not receive file path.")
            return

        file_data = "".join(FileSystem.from_path(file_path).get_file_iterable(file_path))
        await write_utf8_string(file_data, NUM_BYTES_FOR_SIZES, writer, drain=False)
        await writer.drain()

    async def _return_next_result_chunk(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """
        Sends the next chunk of results for a given job ID to the client.

        Args:
            reader (asyncio.StreamReader): The stream reader to read data from the client.
            writer (asyncio.StreamWriter): The stream writer to write data to the client.
        """
        job_id = await read_utf8_string(NUM_BYTES_FOR_IDENTIFIERS, reader)
        async with self._result_chunk_generator_map_lock:
            if job_id not in self._result_chunk_generator_map:
                self._result_chunk_generator_map[job_id] = self._local_stub._get_query_result(job_id)

        next_chunk = next(self._result_chunk_generator_map[job_id], None)
        await write_pickeled_object(next_chunk, NUM_BYTES_FOR_SIZES, writer)

    async def _return_result_metadata(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """
        Sends the metadata for the results of a given job ID to the client.

        Args:
            reader (asyncio.StreamReader): The stream reader to read data from the client.
            writer (asyncio.StreamWriter): The stream writer to write data to the client.
        """
        job_id = await read_utf8_string(NUM_BYTES_FOR_IDENTIFIERS, reader)
        dataset_dict, parsing_dict, file_path_dict = self._local_stub._get_result_metadata(job_id)

        meta = {
            "dataset_type": dataset_dict,
            "parsing_func": parsing_dict,
            "file_path": file_path_dict,
        }
        await write_pickeled_object(meta, NUM_BYTES_FOR_SIZES, writer)

    async def _dispatch_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """
        Dispatches client requests to the appropriate handlers based on the task ID.

        This function reads the task ID sent by the client and calls the corresponding
        method to handle the request. Before closing, it ensures that the writer is properly
        closed and any exceptions are logged.

        Args:
            reader (asyncio.StreamReader): The stream reader to read data from the client.
            writer (asyncio.StreamWriter): The stream writer to write data to the client.
        """
        try:
            if (task_int := await read_int(NUM_BYTES_FOR_IDENTIFIERS, reader)) not in ServerTask.__members__.values():
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
        """
        Asynchronously runs the server, accepting and handling incoming connections.

        This method starts the server and continuously serves until a cancellation
        request is received or an exception occurs. It also performs clean-up before stopping.
        """
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
        """
        Runs the Mixtera server.

        This is the main entry point to start the server. It calls the asynchronous run method
        and is responsible for handling the event loop.
        """
        asyncio.run(self._run_async())
