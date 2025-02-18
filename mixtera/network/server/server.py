import asyncio
import importlib
import socket
import sys
import traceback
import typing
import uuid
from pathlib import Path

from loguru import logger
from mixtera.core.client.local import LocalStub
from mixtera.core.client.mixtera_client import QueryExecutionArgs
from mixtera.core.datacollection.datasets.dataset import Dataset
from mixtera.core.datacollection.index.parser.metadata_parser import MetadataParser, MetadataProperty
from mixtera.core.datacollection.property_type import PropertyType
from mixtera.core.filesystem.filesystem import FileSystem
from mixtera.core.processing import ExecutionMode
from mixtera.core.query.chunk_distributor import ChunkDistributor
from mixtera.network import NUM_BYTES_FOR_IDENTIFIERS, NUM_BYTES_FOR_SIZES
from mixtera.network.client.client_feedback import ClientFeedback
from mixtera.network.network_utils import (
    read_float,
    read_int,
    read_numpy_array,
    read_pickeled_object,
    read_utf8_string,
    write_bytes_obj,
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
        self._chunk_distributor_map: dict[str, ChunkDistributor] = {}
        self._chunk_distributor_map_lock = asyncio.Lock()
        self._register_query_lock = asyncio.Lock()
        self._dataset_registration: dict[str, bool] = {}

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
        mixture = await read_pickeled_object(NUM_BYTES_FOR_SIZES, reader)
        logger.debug(f"mixture = {mixture}")

        dp_groups = await read_int(NUM_BYTES_FOR_IDENTIFIERS, reader)
        nodes_per_group = await read_int(NUM_BYTES_FOR_IDENTIFIERS, reader)
        num_workers = await read_int(NUM_BYTES_FOR_IDENTIFIERS, reader)

        args = QueryExecutionArgs(
            mixture=mixture,
            dp_groups=dp_groups,
            nodes_per_group=nodes_per_group,
            num_workers=num_workers,
        )

        query = await read_pickeled_object(NUM_BYTES_FOR_SIZES, reader)
        logger.debug(f"Received query = {str(query)}. Executing it.")
        async with self._register_query_lock:
            success = self._local_stub.execute_query(query, args)
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
        dp_group_id = await read_int(NUM_BYTES_FOR_IDENTIFIERS, reader)
        node_id = await read_int(NUM_BYTES_FOR_IDENTIFIERS, reader)
        worker_id = await read_int(NUM_BYTES_FOR_IDENTIFIERS, reader)

        async with self._chunk_distributor_map_lock:
            if job_id not in self._chunk_distributor_map:
                self._chunk_distributor_map[job_id] = self._local_stub._get_query_chunk_distributor(job_id)

        next_chunk = None
        try:
            next_chunk = self._chunk_distributor_map[job_id].next_chunk_for(dp_group_id, node_id, worker_id, False)
        except StopIteration:
            pass

        await write_bytes_obj(next_chunk, NUM_BYTES_FOR_SIZES, writer)

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

    async def _register_dataset(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """
        Registers a dataset with the server.

        This method reads the dataset identifier, location, dataset type, parsing function,
        and metadata parser identifier from the client, registers the dataset with the LocalStub,
        and writes back the success status.

        Args:
            reader (asyncio.StreamReader): The stream reader to read data from the client.
            writer (asyncio.StreamWriter): The stream writer to write data to the client.
        """
        identifier = await read_utf8_string(NUM_BYTES_FOR_IDENTIFIERS, reader)
        loc = await read_utf8_string(NUM_BYTES_FOR_IDENTIFIERS, reader)
        dataset_type_id = await read_int(NUM_BYTES_FOR_IDENTIFIERS, reader)
        dtype = Dataset.from_type_id(dataset_type_id)
        parsing_func = await read_pickeled_object(NUM_BYTES_FOR_SIZES, reader)
        metadata_parser_identifier = await read_utf8_string(NUM_BYTES_FOR_IDENTIFIERS, reader)
        job_id = str(uuid.uuid4())
        self._dataset_registration[job_id] = None
        asyncio.create_task(
            self._background_register_dataset(job_id, identifier, loc, dtype, parsing_func, metadata_parser_identifier)
        )
        await write_utf8_string(job_id, NUM_BYTES_FOR_IDENTIFIERS, writer)

    async def _background_register_dataset(
        self,
        job_id: str,
        identifier: str,
        loc: str,
        dtype: "Dataset",
        parsing_func: typing.Callable[[str], str],
        metadata_parser_identifier: str,
    ) -> None:
        try:
            success = await asyncio.to_thread(
                self._local_stub.register_dataset, identifier, loc, dtype, parsing_func, metadata_parser_identifier
            )
            self._dataset_registration[job_id] = success
        except Exception:  # pylint: disable=broad-exception-caught
            logger.error(f"Background dataset registration failed: {traceback.format_exc()}")
            self._dataset_registration[job_id] = False

    async def _check_dataset_registration(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        job_id = await read_utf8_string(NUM_BYTES_FOR_IDENTIFIERS, reader)
        success = self._dataset_registration.get(job_id, None)
        # Return 0 if not complete, 1 if registration succeeded, or 2 if complete but failed.
        if success is True:
            status = 1
        elif success is False and job_id in self._dataset_registration:
            status = 2
        else:
            status = 0
        await write_int(status, NUM_BYTES_FOR_IDENTIFIERS, writer)

    async def _register_metadata_parser(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """
        Registers a metadata parser with the server.

        This method reads the metadata parser identifier and its associated class from the client,
        registers the parser with the LocalStub, and writes back the success status.

        Args:
            reader (asyncio.StreamReader): The stream reader to read data from the client.
            writer (asyncio.StreamWriter): The stream writer to write data to the client.
        """
        identifier = await read_utf8_string(NUM_BYTES_FOR_IDENTIFIERS, reader)
        source_code = await read_utf8_string(NUM_BYTES_FOR_SIZES, reader)
        namespace = {
            "MetadataParser": MetadataParser,
            "MetadataProperty": MetadataProperty,
            "Any": typing.Any,
            "Optional": typing.Optional,
            "List": typing.List,
            "Dict": typing.Dict,
        }

        exec(source_code, namespace)  # pylint: disable=exec-used
        parser_candidates = [
            obj
            for obj in namespace.values()
            if isinstance(obj, type) and issubclass(obj, MetadataParser) and obj is not MetadataParser
        ]

        if len(parser_candidates) == 0:
            raise RuntimeError("No metadata parser subclass found in the provided source code.")
        if len(parser_candidates) > 1:
            raise RuntimeError("Multiple metadata parser subclasses found. Please ensure only one is defined.")

        parser = parser_candidates[0]
        module_name = "mixtera_udf_" + uuid.uuid4().hex

        header = (
            "from mixtera.core.datacollection.index.parser import MetadataParser\n"
            "from mixtera.core.datacollection.index.parser.metadata_parser import MetadataProperty\n"
            "from typing import Any, Optional, List, Dict\n"
        )

        final_source_code = header + "\n" + source_code

        # Write the source code to a Python file in the server directory
        # This is necessary because otherwise when the server uses multiprocessing
        # with the newly registered parser, it does not have access to the code anymore.

        udf_dir = self._directory / "udfs"
        udf_dir.mkdir(exist_ok=True)
        file_path = udf_dir / f"{module_name}.py"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(final_source_code)

        if udf_dir not in sys.path:
            sys.path.insert(0, str(udf_dir))

        # Import the module, so that it is available to spawned processes
        imported_module = importlib.import_module(module_name)
        parser.__module__ = module_name
        setattr(imported_module, parser.__name__, parser)

        success = self._local_stub.register_metadata_parser(identifier, parser)
        await write_int(int(success), NUM_BYTES_FOR_IDENTIFIERS, writer)

    async def _check_dataset_exists(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """
        Checks if a dataset with a given identifier exists in the server's file system.

        Args:
            reader (asyncio.StreamReader): The stream reader to read data from the client.
            writer (asyncio.StreamWriter): The stream writer to write data to the client.
        """
        identifier = await read_utf8_string(NUM_BYTES_FOR_IDENTIFIERS, reader)
        exists = self._local_stub.check_dataset_exists(identifier)
        await write_int(int(exists), NUM_BYTES_FOR_IDENTIFIERS, writer)

    async def _list_datasets(self, writer: asyncio.StreamWriter) -> None:
        """
        Lists all datasets stored in the server's file system.

        Args:
            reader (asyncio.StreamReader): The stream reader to read data from the client.
            writer (asyncio.StreamWriter): The stream writer to write data to the client.
        """
        datasets = self._local_stub.list_datasets()
        await write_pickeled_object(datasets, NUM_BYTES_FOR_SIZES, writer)

    async def _remove_dataset(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """
        Removes a dataset from the server's file system.

        Args:
            reader (asyncio.StreamReader): The stream reader to read data from the client.
            writer (asyncio.StreamWriter): The stream writer to write data to the client.
        """
        identifier = await read_utf8_string(NUM_BYTES_FOR_IDENTIFIERS, reader)
        success = self._local_stub.remove_dataset(identifier)
        await write_int(int(success), NUM_BYTES_FOR_IDENTIFIERS, writer)

    async def _add_property(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """
        Adds a property to the server's data collection.

        This method reads the property name, setup function, calculation function, execution mode,
        property type, and additional parameters from the client, adds the property to the LocalStub,
        and writes back the success status.

        Args:
            reader (asyncio.StreamReader): The stream reader to read data from the client.
            writer (asyncio.StreamWriter): The stream writer to write data to the client.
        """
        property_name = await read_utf8_string(NUM_BYTES_FOR_IDENTIFIERS, reader)
        setup_func = await read_pickeled_object(NUM_BYTES_FOR_SIZES, reader)
        calc_func = await read_pickeled_object(NUM_BYTES_FOR_SIZES, reader)
        execution_mode_value: int = await read_int(NUM_BYTES_FOR_IDENTIFIERS, reader)
        execution_mode = ExecutionMode(execution_mode_value)
        property_type_value: int = await read_int(NUM_BYTES_FOR_IDENTIFIERS, reader)
        property_type = PropertyType(property_type_value)
        min_val = await read_float(reader)
        max_val = await read_float(reader)
        num_buckets = await read_int(NUM_BYTES_FOR_IDENTIFIERS, reader)
        batch_size = await read_int(NUM_BYTES_FOR_IDENTIFIERS, reader)
        degree_of_parallelism = await read_int(NUM_BYTES_FOR_IDENTIFIERS, reader)
        data_only_on_primary = await read_int(NUM_BYTES_FOR_IDENTIFIERS, reader)
        success = self._local_stub.add_property(
            property_name,
            setup_func,
            calc_func,
            execution_mode,
            property_type,
            min_val,
            max_val,
            num_buckets,
            batch_size,
            degree_of_parallelism,
            data_only_on_primary,
        )
        await write_int(int(success), NUM_BYTES_FOR_IDENTIFIERS, writer)

    async def _checkpoint(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        job_id = await read_utf8_string(NUM_BYTES_FOR_IDENTIFIERS, reader)
        dp_group_id = await read_int(NUM_BYTES_FOR_IDENTIFIERS, reader)
        node_id = await read_int(NUM_BYTES_FOR_IDENTIFIERS, reader)

        worker_status_length = await read_int(NUM_BYTES_FOR_IDENTIFIERS, reader)
        worker_status = []
        for _ in range(worker_status_length):
            status = await read_int(NUM_BYTES_FOR_IDENTIFIERS, reader)
            worker_status.append(status)

        chkpnt_id = self._local_stub.checkpoint(job_id, dp_group_id, node_id, worker_status, server=True)

        await write_utf8_string(chkpnt_id, NUM_BYTES_FOR_SIZES, writer)

    async def _checkpoint_completed(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:

        job_id = await read_utf8_string(NUM_BYTES_FOR_IDENTIFIERS, reader)
        chkpnt_id = await read_utf8_string(NUM_BYTES_FOR_IDENTIFIERS, reader)

        on_disk_flag = await read_int(NUM_BYTES_FOR_IDENTIFIERS, reader)
        on_disk = bool(on_disk_flag)

        is_completed = self._local_stub.checkpoint_completed(job_id, chkpnt_id, on_disk)

        await write_int(int(is_completed), NUM_BYTES_FOR_IDENTIFIERS, writer)

    async def _restore_checkpoint(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        job_id = await read_utf8_string(NUM_BYTES_FOR_IDENTIFIERS, reader)
        chkpnt_id = await read_utf8_string(NUM_BYTES_FOR_IDENTIFIERS, reader)

        self._local_stub.restore_checkpoint(job_id, chkpnt_id)

        async with self._chunk_distributor_map_lock:
            # We have a new distributor object.
            self._chunk_distributor_map[job_id] = self._local_stub._get_query_chunk_distributor(job_id)

        success = True
        await write_int(int(success), NUM_BYTES_FOR_IDENTIFIERS, writer)

    async def _process_feedback(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        job_id = await read_utf8_string(NUM_BYTES_FOR_IDENTIFIERS, reader)
        feedback = ClientFeedback()
        feedback.training_steps = await read_int(NUM_BYTES_FOR_IDENTIFIERS, reader)
        feedback.mixture_id = await read_int(NUM_BYTES_FOR_IDENTIFIERS, reader)
        feedback.losses = await read_numpy_array(NUM_BYTES_FOR_IDENTIFIERS, NUM_BYTES_FOR_SIZES, reader)
        feedback.counts = await read_numpy_array(NUM_BYTES_FOR_IDENTIFIERS, NUM_BYTES_FOR_SIZES, reader)

        self._local_stub.process_feedback(job_id, feedback)

        success = True
        await write_int(int(success), NUM_BYTES_FOR_IDENTIFIERS, writer)

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
                logger.error(f"Unknown task id: {task_int}")
                return

            task = ServerTask(task_int)
            logger.info(f"Got task = {task}")
            if task == ServerTask.REGISTER_QUERY:
                await self._register_query(reader, writer)
            elif task == ServerTask.READ_FILE:
                await self._read_file(reader, writer)
            elif task == ServerTask.GET_META_RESULT:
                await self._return_result_metadata(reader, writer)
            elif task == ServerTask.GET_NEXT_RESULT_CHUNK:
                await self._return_next_result_chunk(reader, writer)
            elif task == ServerTask.REGISTER_DATASET:
                await self._register_dataset(reader, writer)
            elif task == ServerTask.DATASET_REGISTRATION_STATUS:
                await self._check_dataset_registration(reader, writer)
            elif task == ServerTask.REGISTER_METADATA_PARSER:
                await self._register_metadata_parser(reader, writer)
            elif task == ServerTask.CHECK_DATASET_EXISTS:
                await self._check_dataset_exists(reader, writer)
            elif task == ServerTask.LIST_DATASETS:
                await self._list_datasets(writer)
            elif task == ServerTask.REMOVE_DATASET:
                await self._remove_dataset(reader, writer)
            elif task == ServerTask.ADD_PROPERTY:
                await self._add_property(reader, writer)
            elif task == ServerTask.CHECKPOINT:
                await self._checkpoint(reader, writer)
            elif task == ServerTask.CHECKPOINT_COMPLETED:
                await self._checkpoint_completed(reader, writer)
            elif task == ServerTask.RESTORE_CHECKPOINT:
                await self._restore_checkpoint(reader, writer)
            elif task == ServerTask.RECEIVE_FEEDBACK:
                await self._process_feedback(reader, writer)
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
        server = await asyncio.start_server(self._dispatch_client, self._host, self._port, limit=2**26, backlog=2048)
        addr = server.sockets[0].getsockname()
        server.sockets[0].setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
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
