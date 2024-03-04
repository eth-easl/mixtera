import asyncio
from pathlib import Path

from loguru import logger
from mixtera.core.datacollection.local import LocalDataCollection
from mixtera.network.server.server_task import ServerTask
from mixtera.utils.network_utils import read_int, read_pickeled_object, read_utf8_string, write_int, write_utf8_string

ID_BYTES = 8
SAMPLE_SIZE_BYTES = 16

# TODO(#): Use actual query instead of dict of ranges
QueryType = dict[int, dict[int, list[tuple[int, int]]]]


class MixteraServer:
    def __init__(self, directory: Path, host: str, port: int):
        self._ldc = LocalDataCollection(directory)
        self._host = host
        self._port = port
        self._directory = directory

    async def _register_query(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        # TODO(#): Use actual query instead of dict of ranges

        training_id = await read_utf8_string(SAMPLE_SIZE_BYTES, reader)
        logger.debug(f"Received register training request for training id {training_id}")
        num_nodes = await read_int(ID_BYTES, reader)
        num_workers_per_node = await read_int(ID_BYTES, reader)
        logger.debug(f"num_nodes = {num_nodes}, num_workers_per_node = {num_workers_per_node}")
        query = await read_pickeled_object(SAMPLE_SIZE_BYTES, reader)
        logger.debug(f"Received query = {query}")

        query_id = self._ldc.register_query(query, training_id, num_workers_per_node, num_nodes=num_nodes)
        await write_int(query_id, ID_BYTES, writer)

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

    async def _stream_query_results(
        self, node_id: int, worker_id: int, query_id: int, writer: asyncio.StreamWriter
    ) -> None:
        for data in self._ldc.stream_query_results(query_id, worker_id, node_id=node_id):
            await write_utf8_string(data, SAMPLE_SIZE_BYTES, writer)

    async def _dispatch_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            if (task_int := await read_int(ID_BYTES, reader)) not in ServerTask.__members__.values():
                raise RuntimeError(f"Unknown task id: {task_int}")

            task = ServerTask(task_int)

            if task == ServerTask.REGISTER_QUERY:
                await self._register_query(reader, writer)
            elif task == ServerTask.STREAM_DATA:
                node_id, worker_id, query_id = await self._parse_ids(reader)
                await self._stream_query_results(node_id, worker_id, query_id, writer)
            elif task == ServerTask.GET_QUERY_ID:
                query_id = await self._get_query_id(reader)
                await write_int(query_id, ID_BYTES, writer)
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
