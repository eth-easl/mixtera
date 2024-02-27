import asyncio
import threading
from pathlib import Path

from loguru import logger
from mixtera.core.datacollection.local import LocalDataCollection
from mixtera.server.server_task import ServerTask
from mixtera.utils import wait_for_key_in_dict
from mixtera.utils.network_utils import read_int, read_pickeled_object, read_utf8_string, write_int, write_utf8_string

ID_BYTES = 8
SAMPLE_SIZE_BYTES = 16

# TODO(#): Use actual query instead of dict of ranges
QueryType = dict[int, dict[int, list[tuple[int, int]]]]

# TODO(MaxiBoether): We might actually need to move some of this functionality to the LDC and forward the calls
# That makes more sense from an abstraction POV, and also if somebody uses a local collection with multiple workers,
# that should work smoothly and can share logic
# For example, the LDC should store the query (to also support local multi-worker) - change as outlined in the new simple_client.py!


class MixteraServer:
    def __init__(self, directory: Path, host: str, port: int):
        self._ldc = LocalDataCollection(directory)
        self._host = host
        self._port = port
        self._directory = directory
        self._queries_lock = threading.Lock()
        self._queries: list[tuple[QueryType, int, int]] = list()  # (query, num nodes, num workers per node)
        self._training_query_map: dict[str, int] = {}

    async def _register_query(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        # TODO(#): Get actual query instead of dict

        training_id = await read_utf8_string(SAMPLE_SIZE_BYTES, reader)
        logger.debug(f"Received register training request for training id {training_id}")

        if training_id in self._training_query_map:
            logger.warning("We already have a query for that training!")
            await write_int(-1, ID_BYTES, writer)
            return

        num_nodes = await read_int(ID_BYTES, reader)
        num_workers_per_nodes = await read_int(ID_BYTES, reader)
        logger.debug(f"num_nodes = {num_nodes}, num_workers_per_nodes = {num_workers_per_nodes}")
        query = await read_pickeled_object(SAMPLE_SIZE_BYTES, reader)
        logger.debug(f"Received query = {query}")

        with self._queries_lock:
            self._queries.append((query, num_nodes, num_workers_per_nodes))
            index = len(self._queries) - 1
            self._training_query_map[training_id] = index

        await write_int(index, ID_BYTES, writer)

        logger.info(
            f"Registered query {index} with {num_nodes} nodes and "
            + f"{num_workers_per_nodes} workers per node for training {training_id}."
        )

        # TODO(#): We might want to start executing the query here already and even prefetch some data

    async def _parse_ids(self, reader: asyncio.StreamReader):
        query_id = await read_int(ID_BYTES, reader)
        node_id = await read_int(ID_BYTES, reader)
        worker_id = await read_int(ID_BYTES, reader)

        logger.info(f"Worker {worker_id} for node {node_id} has connected for query {query_id}.")

        return node_id, worker_id, query_id

    async def _get_query_id(self, reader: asyncio.StreamReader) -> int:
        training_id = await read_utf8_string(SAMPLE_SIZE_BYTES, reader)
        logger.debug(f"Looking up query ID for training {training_id}")
        if await wait_for_key_in_dict(self._training_query_map, training_id, 15.0):
            query_id = self._training_query_map[training_id]
            logger.debug(f"Query ID for training {training_id} is {query_id}")
            return query_id

        logger.warning(f"Did not find query ID for training {training_id} after 15 seconds.")
        return -1

    async def _serve_worker(self, node_id: int, worker_id: int, query_id: int, writer: asyncio.StreamWriter):
        # TODO(#): Actually take node_id/worker_id into consideration. We might also want to have internal prefetching.
        if query_id < 0 or query_id >= len(self._queries):
            logger.error(f"Invalid query_id {query_id}")
            return

        query, num_nodes, num_workers_per_node = self._queries[query_id]

        if node_id < 0 or node_id >= num_nodes:
            logger.error(f"Invalid node {node_id} for query {query_id} (total nodes = {num_nodes})")
            return

        if worker_id < 0 or worker_id >= num_workers_per_node:
            logger.error(f"Invalid worker {worker_id} for query {query_id} (workers per node = {num_workers_per_node})")
            return

        for data in self._ldc.get_samples_from_ranges(query):
            await write_utf8_string(data, SAMPLE_SIZE_BYTES, writer)

    async def _dispatch_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        try:
            if (task_int := await read_int(ID_BYTES, reader)) not in ServerTask.__members__.values():
                logger.error(f"Unknown task id: {task_int}")

            task = ServerTask(task_int)

            if task == ServerTask.RegisterQuery:
                await self._register_query(reader, writer)
            elif task == ServerTask.StreamData:
                node_id, worker_id, query_id = await self._parse_ids(reader)
                await self._serve_worker(node_id, worker_id, query_id, writer)
            elif task == ServerTask.GetQueryId:
                query_id = await self._get_query_id(reader)
                await write_int(query_id, ID_BYTES, writer)
            else:
                logger.error(f"Client sent unsupport task {task}")
        except asyncio.CancelledError:
            logger.error("asyncio.CancelledError")
            pass
        except Exception as e:
            logger.error(f"Some exception occured while handling client request: {e}")
            logger.exception(e)
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception as e:
                logger.error(f"Error while closing writer: {e}")
                logger.exception(e)

    async def _run_async(self):
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

    def run(self):
        asyncio.run(self._run_async())
