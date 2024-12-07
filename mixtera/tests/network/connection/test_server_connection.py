# pylint: disable=attribute-defined-outside-init
import asyncio
import unittest
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, call, patch

import dill
import numpy as np
from mixtera.core.client.mixtera_client import QueryExecutionArgs
from mixtera.core.datacollection.index.parser import MetadataParser
from mixtera.network import NUM_BYTES_FOR_IDENTIFIERS, NUM_BYTES_FOR_SIZES
from mixtera.network.client.client_feedback import ClientFeedback
from mixtera.network.connection.server_connection import ServerConnection
from mixtera.network.server_task import ServerTask


def create_mock_reader(*args):
    mock_reader = MagicMock(asyncio.StreamReader)
    mock_reader.readexactly = AsyncMock(side_effect=list(args))
    return mock_reader


def create_mock_writer():
    mock_writer = MagicMock(asyncio.StreamWriter)
    mock_writer.drain = AsyncMock()
    mock_writer.write = MagicMock()
    return mock_writer


class MockMetadataParser(MetadataParser):
    def parse(self, line_number: int, payload: Any, **kwargs: Optional[dict[Any, Any]]) -> None:
        pass


class TestServerConnection(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.host = "localhost"
        self.port = 12345
        self.server_connection = ServerConnection(host=self.host, port=self.port)

    @patch("asyncio.open_connection")
    async def test_connect_to_server_async(self, mock_open_connection):
        mock_reader = create_mock_reader()
        mock_writer = create_mock_writer()
        mock_open_connection.return_value = mock_reader, mock_writer

        reader, writer = await self.server_connection._connect_to_server(max_retries=1)

        self.assertEqual(reader, mock_reader)
        self.assertEqual(writer, mock_writer)
        mock_open_connection.assert_awaited_once_with(self.host, self.port)

    @patch("asyncio.open_connection")
    async def test_connect_to_server_timeout(self, mock_open_connection):
        mock_open_connection.side_effect = asyncio.TimeoutError()

        reader, writer = await self.server_connection._connect_to_server(max_retries=1)

        self.assertIsNone(reader)
        self.assertIsNone(writer)
        mock_open_connection.assert_awaited_once_with(self.host, self.port)

    @patch("asyncio.open_connection")
    async def test_connect_to_server_exception(self, mock_open_connection):
        mock_open_connection.side_effect = Exception("Test exception")

        reader, writer = await self.server_connection._connect_to_server(max_retries=1)

        self.assertIsNone(reader)
        self.assertIsNone(writer)
        mock_open_connection.assert_awaited_once_with(self.host, self.port)

    @patch("mixtera.network.connection.server_connection.ServerConnection._connect_to_server")
    @patch("mixtera.network.connection.server_connection.read_utf8_string")
    @patch("mixtera.network.connection.server_connection.write_utf8_string")
    @patch("mixtera.network.connection.server_connection.write_int")
    async def test_fetch_file_async(
        self, mock_write_int, mock_write_utf8_string, mock_read_utf8_string, mock_connect_to_server
    ):
        mock_reader = create_mock_reader()
        mock_writer = create_mock_writer()
        mock_connect_to_server.return_value = mock_reader, mock_writer
        mock_read_utf8_string.return_value = "file_data"
        file_path = "/path/to/file"

        file_data = await self.server_connection._fetch_file(file_path)

        self.assertEqual(file_data, "file_data")
        mock_connect_to_server.assert_awaited_once()
        mock_write_int.assert_has_calls(
            [
                call(int(ServerTask.READ_FILE), NUM_BYTES_FOR_IDENTIFIERS, mock_writer),
            ]
        )
        mock_write_utf8_string.assert_awaited_once_with(file_path, NUM_BYTES_FOR_IDENTIFIERS, mock_writer)
        mock_read_utf8_string.assert_awaited_once_with(NUM_BYTES_FOR_SIZES, mock_reader)

    @patch("mixtera.network.connection.server_connection.ServerConnection._fetch_file")
    def test_get_file_iterable_sync(self, mock_fetch_file):
        mock_fetch_file.return_value = "line1\nline2\nline3"
        file_path = "/path/to/file"

        file_iterable = self.server_connection.get_file_iterable(file_path)
        lines = list(file_iterable)

        self.assertEqual(lines, ["line1", "line2", "line3"])
        mock_fetch_file.assert_awaited_once_with(file_path)

    @patch("mixtera.network.connection.server_connection.ServerConnection._connect_to_server")
    @patch("mixtera.network.connection.server_connection.read_int")
    @patch("mixtera.network.connection.server_connection.write_pickeled_object")
    @patch("mixtera.network.connection.server_connection.write_int")
    async def test_execute_query_async(
        self, mock_write_int, mock_write_pickeled_object, mock_read_int, mock_connect_to_server
    ):
        mock_reader = create_mock_reader()
        mock_writer = create_mock_writer()
        mock_connect_to_server.return_value = mock_reader, mock_writer
        mock_read_int.return_value = 1
        query_mock = MagicMock()
        mixture_mock = MagicMock()
        args = QueryExecutionArgs(mixture=mixture_mock)
        success = await self.server_connection._execute_query(query_mock, args)

        self.assertTrue(success)
        mock_connect_to_server.assert_awaited_once()
        mock_write_int.assert_has_calls(
            [
                call(int(ServerTask.REGISTER_QUERY), NUM_BYTES_FOR_IDENTIFIERS, mock_writer),
            ]
        )
        mock_write_pickeled_object.assert_any_await(mixture_mock, NUM_BYTES_FOR_SIZES, mock_writer)
        mock_write_pickeled_object.assert_any_await(query_mock, NUM_BYTES_FOR_SIZES, mock_writer)
        self.assertEqual(mock_write_pickeled_object.await_count, 2)
        mock_read_int.assert_awaited_once_with(NUM_BYTES_FOR_IDENTIFIERS, mock_reader, timeout=60 * 15)

    @patch("mixtera.network.connection.server_connection.ServerConnection._connect_to_server")
    @patch("mixtera.network.connection.server_connection.read_pickeled_object")
    @patch("mixtera.network.connection.server_connection.write_int")
    @patch("mixtera.network.connection.server_connection.write_utf8_string")
    async def test_get_query_result_meta_async(
        self, mock_write_string, mock_write_int, mock_read_pickeled_object, mock_connect_to_server
    ):
        mock_reader = create_mock_reader()
        mock_writer = create_mock_writer()
        mock_connect_to_server.return_value = mock_reader, mock_writer
        mock_read_pickeled_object.return_value = {"meta": "data"}
        job_id = "job_id"

        meta_result = await self.server_connection._get_query_result_meta(job_id)

        self.assertEqual(meta_result, {"meta": "data"})
        mock_connect_to_server.assert_awaited_once()
        mock_write_int.assert_has_calls([call(int(ServerTask.GET_META_RESULT), NUM_BYTES_FOR_IDENTIFIERS, mock_writer)])
        mock_write_string.assert_has_calls([call(job_id, NUM_BYTES_FOR_IDENTIFIERS, mock_writer)])

        mock_read_pickeled_object.assert_awaited_once_with(NUM_BYTES_FOR_SIZES, mock_reader)

    @patch("mixtera.network.connection.server_connection.ServerConnection._get_next_result")
    def test_get_query_results_sync(self, mock_get_next_result):
        mock_get_next_result.side_effect = [[1, 2, 3], [4, 5, 6], None]
        job_id = "job_id"

        results = self.server_connection._stream_result_chunks(job_id, 1, 1, 1)
        result_list = list(results)

        self.assertEqual(result_list, [[1, 2, 3], [4, 5, 6]])
        mock_get_next_result.assert_has_calls([call(job_id, 1, 1, 1), call(job_id, 1, 1, 1), call(job_id, 1, 1, 1)])

    @patch("mixtera.network.connection.server_connection.ServerConnection._connect_to_server")
    @patch("mixtera.network.connection.server_connection.read_bytes_obj")
    @patch("mixtera.network.connection.server_connection.write_int")
    @patch("mixtera.network.connection.server_connection.write_utf8_string")
    async def test_get_next_result(
        self, mock_write_string, mock_write_int, mock_read_bytes_obj, mock_connect_to_server
    ):
        mock_reader = create_mock_reader()
        mock_writer = create_mock_writer()
        mock_connect_to_server.return_value = mock_reader, mock_writer
        serialized_chunk = dill.dumps([1, 2, 3])
        mock_read_bytes_obj.return_value = serialized_chunk
        job_id = "job_id"

        result_chunk = await self.server_connection._get_next_result(job_id, 1, 1, 1)

        self.assertEqual(result_chunk, [1, 2, 3])
        mock_connect_to_server.assert_awaited_once()
        mock_write_int.assert_has_calls(
            [
                call(int(ServerTask.GET_NEXT_RESULT_CHUNK), NUM_BYTES_FOR_IDENTIFIERS, mock_writer),
                call(1, NUM_BYTES_FOR_IDENTIFIERS, mock_writer),
                call(1, NUM_BYTES_FOR_IDENTIFIERS, mock_writer),
                call(1, NUM_BYTES_FOR_IDENTIFIERS, mock_writer),
            ]
        )
        mock_write_string.assert_has_calls([call(job_id, NUM_BYTES_FOR_IDENTIFIERS, mock_writer)])
        mock_read_bytes_obj.assert_awaited_once_with(NUM_BYTES_FOR_SIZES, mock_reader, timeout=600)

    @patch("mixtera.network.connection.server_connection.ServerConnection._connect_to_server")
    @patch("mixtera.network.connection.server_connection.write_utf8_string")
    @patch("mixtera.network.connection.server_connection.write_int")
    @patch("mixtera.network.connection.server_connection.write_pickeled_object")
    @patch("mixtera.network.connection.server_connection.read_int")
    async def test_register_dataset(
        self, mock_read_int, mock_write_pickeled_object, mock_write_int, mock_write_string, mock_connect_to_server
    ):
        mock_reader = create_mock_reader()
        mock_writer = create_mock_writer()
        mock_connect_to_server.return_value = mock_reader, mock_writer
        mock_read_int.return_value = 1
        mock_write_pickeled_object.return_value = True
        identifier = "identifier"
        loc = "loc"
        dtype = MagicMock()
        dtype.value = 0
        parsing_func = MagicMock()

        metadata_parser_identifier = "metadata_parser_identifier"

        success = await self.server_connection._register_dataset(
            identifier, loc, dtype, parsing_func, metadata_parser_identifier
        )

        self.assertTrue(success)
        mock_connect_to_server.assert_awaited_once()
        mock_write_int.assert_has_calls(
            [call(int(ServerTask.REGISTER_DATASET), NUM_BYTES_FOR_IDENTIFIERS, mock_writer)]
        )
        mock_write_string.assert_has_calls([call(identifier, NUM_BYTES_FOR_IDENTIFIERS, mock_writer)])
        mock_write_string.assert_has_calls([call(loc, NUM_BYTES_FOR_IDENTIFIERS, mock_writer)])
        mock_write_int.assert_has_calls([call(dtype.value, NUM_BYTES_FOR_IDENTIFIERS, mock_writer)])
        mock_write_pickeled_object.assert_has_calls([call(parsing_func, NUM_BYTES_FOR_SIZES, mock_writer)])
        mock_write_string.assert_has_calls([call(metadata_parser_identifier, NUM_BYTES_FOR_IDENTIFIERS, mock_writer)])
        mock_read_int.assert_awaited_once_with(NUM_BYTES_FOR_IDENTIFIERS, mock_reader)

    @patch("mixtera.network.connection.server_connection.ServerConnection._connect_to_server")
    @patch("mixtera.network.connection.server_connection.write_utf8_string")
    @patch("mixtera.network.connection.server_connection.write_int")
    @patch("mixtera.network.connection.server_connection.write_pickeled_object")
    @patch("mixtera.network.connection.server_connection.read_int")
    async def test_register_metadata_parser(
        self, mock_read_int, mock_write_pickeled_object, mock_write_int, mock_write_string, mock_connect_to_server
    ):
        mock_reader = create_mock_reader()
        mock_writer = create_mock_writer()
        mock_connect_to_server.return_value = mock_reader, mock_writer
        mock_read_int.return_value = 1
        identifier = "identifier"
        parser = MockMetadataParser

        await self.server_connection._register_metadata_parser(identifier, parser)

        mock_connect_to_server.assert_awaited_once()
        mock_write_int.assert_has_calls(
            [call(int(ServerTask.REGISTER_METADATA_PARSER), NUM_BYTES_FOR_IDENTIFIERS, mock_writer)]
        )
        mock_write_string.assert_has_calls([call(identifier, NUM_BYTES_FOR_IDENTIFIERS, mock_writer)])
        mock_write_pickeled_object.assert_has_calls([call(parser, NUM_BYTES_FOR_SIZES, mock_writer)])

    @patch("mixtera.network.connection.server_connection.ServerConnection._connect_to_server")
    @patch("mixtera.network.connection.server_connection.write_utf8_string")
    @patch("mixtera.network.connection.server_connection.write_int")
    @patch("mixtera.network.connection.server_connection.read_int")
    async def test_check_dataset_exists(self, mock_read_int, mock_write_int, mock_write_string, mock_connect_to_server):
        mock_reader = create_mock_reader()
        mock_writer = create_mock_writer()
        mock_connect_to_server.return_value = mock_reader, mock_writer
        mock_read_int.return_value = 1
        identifier = "identifier"

        exists = await self.server_connection._check_dataset_exists(identifier)

        self.assertTrue(exists)
        mock_connect_to_server.assert_awaited_once()
        mock_write_int.assert_has_calls(
            [call(int(ServerTask.CHECK_DATASET_EXISTS), NUM_BYTES_FOR_IDENTIFIERS, mock_writer)]
        )
        mock_write_string.assert_has_calls([call(identifier, NUM_BYTES_FOR_IDENTIFIERS, mock_writer)])
        mock_read_int.assert_awaited_once_with(NUM_BYTES_FOR_IDENTIFIERS, mock_reader)

    @patch("mixtera.network.connection.server_connection.ServerConnection._connect_to_server")
    @patch("mixtera.network.connection.server_connection.write_int")
    @patch("mixtera.network.connection.server_connection.read_pickeled_object")
    async def test_list_datasets(self, mock_read_pickeled_object, mock_write_int, mock_connect_to_server):
        mock_reader = create_mock_reader()
        mock_writer = create_mock_writer()
        mock_connect_to_server.return_value = mock_reader, mock_writer
        mock_read_pickeled_object.return_value = ["dataset1", "dataset2"]

        datasets = await self.server_connection._list_datasets()

        self.assertEqual(datasets, ["dataset1", "dataset2"])
        mock_connect_to_server.assert_awaited_once()
        mock_write_int.assert_has_calls([call(int(ServerTask.LIST_DATASETS), NUM_BYTES_FOR_IDENTIFIERS, mock_writer)])
        mock_read_pickeled_object.assert_awaited_once_with(NUM_BYTES_FOR_SIZES, mock_reader)

    @patch("mixtera.network.connection.server_connection.ServerConnection._connect_to_server")
    @patch("mixtera.network.connection.server_connection.write_int")
    @patch("mixtera.network.connection.server_connection.write_utf8_string")
    @patch("mixtera.network.connection.server_connection.read_int")
    async def test_remove_dataset(self, mock_read_int, mock_write_string, mock_write_int, mock_connect_to_server):
        mock_reader = create_mock_reader()
        mock_writer = create_mock_writer()
        mock_connect_to_server.return_value = mock_reader, mock_writer
        mock_read_int.return_value = 1
        dataset_id = "dataset_id"

        success = await self.server_connection._remove_dataset(dataset_id)

        self.assertTrue(success)
        mock_connect_to_server.assert_awaited_once()
        mock_write_int.assert_has_calls([call(int(ServerTask.REMOVE_DATASET), NUM_BYTES_FOR_IDENTIFIERS, mock_writer)])
        mock_write_string.assert_has_calls([call(dataset_id, NUM_BYTES_FOR_IDENTIFIERS, mock_writer)])
        mock_read_int.assert_awaited_once_with(NUM_BYTES_FOR_IDENTIFIERS, mock_reader)

    @patch("mixtera.network.connection.server_connection.ServerConnection._connect_to_server")
    @patch("mixtera.network.connection.server_connection.write_int")
    @patch("mixtera.network.connection.server_connection.write_utf8_string")
    @patch("mixtera.network.connection.server_connection.write_pickeled_object")
    @patch("mixtera.network.connection.server_connection.write_float")
    @patch("mixtera.network.connection.server_connection.read_int")
    async def test_add_property(
        self,
        mock_read_int,
        mock_write_float,
        mock_write_pickeled_object,
        mock_write_string,
        mock_write_int,
        mock_connect_to_server,
    ):
        mock_reader = create_mock_reader()
        mock_writer = create_mock_writer()
        mock_connect_to_server.return_value = mock_reader, mock_writer
        property_name = "property_name"
        setup_func = "setup_func"
        calc_func = "calc_func"
        execution_mode = MagicMock()
        execution_mode.value = 0
        property_type = MagicMock()
        property_type.value = 0
        min_val = 0.2
        max_val = 0.8
        num_buckets = 12
        batch_size = 2
        degree_of_parallelism = 3
        data_only_on_primary = False
        mock_read_int.return_value = 1

        await self.server_connection._add_property(
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

        mock_connect_to_server.assert_awaited_once()
        mock_write_int.assert_has_calls([call(int(ServerTask.ADD_PROPERTY), NUM_BYTES_FOR_IDENTIFIERS, mock_writer)])
        mock_write_string.assert_has_calls([call(property_name, NUM_BYTES_FOR_IDENTIFIERS, mock_writer)])
        mock_write_int.assert_has_calls([call(execution_mode.value, NUM_BYTES_FOR_IDENTIFIERS, mock_writer)])
        mock_write_int.assert_has_calls([call(property_type.value, NUM_BYTES_FOR_IDENTIFIERS, mock_writer)])
        mock_write_float.assert_has_calls([call(min_val, mock_writer)])
        mock_write_float.assert_has_calls([call(max_val, mock_writer)])
        mock_write_int.assert_has_calls([call(num_buckets, NUM_BYTES_FOR_IDENTIFIERS, mock_writer)])
        mock_write_int.assert_has_calls([call(batch_size, NUM_BYTES_FOR_IDENTIFIERS, mock_writer)])
        mock_write_int.assert_has_calls([call(degree_of_parallelism, NUM_BYTES_FOR_IDENTIFIERS, mock_writer)])
        mock_write_int.assert_has_calls([call(int(data_only_on_primary), NUM_BYTES_FOR_IDENTIFIERS, mock_writer)])
        mock_write_pickeled_object.assert_has_calls([call(setup_func, NUM_BYTES_FOR_SIZES, mock_writer)])
        mock_write_pickeled_object.assert_has_calls([call(calc_func, NUM_BYTES_FOR_SIZES, mock_writer)])

    @patch("mixtera.network.connection.server_connection.ServerConnection._connect_to_server")
    @patch("mixtera.network.connection.server_connection.read_utf8_string")
    @patch("mixtera.network.connection.server_connection.write_int")
    @patch("mixtera.network.connection.server_connection.write_utf8_string")
    async def test_checkpoint(
        self, mock_write_utf8_string, mock_write_int, mock_read_utf8_string, mock_connect_to_server
    ):
        """Test the _checkpoint method of ServerConnection."""
        mock_reader = create_mock_reader()
        mock_writer = create_mock_writer()
        mock_connect_to_server.return_value = mock_reader, mock_writer

        job_id = "job_id"
        dp_group_id = 0
        node_id = 0
        worker_status = [1, 2, 3]
        chkpnt_id = "test_checkpoint_id"
        mock_read_utf8_string.return_value = chkpnt_id

        result = await self.server_connection._checkpoint(job_id, dp_group_id, node_id, worker_status)

        self.assertEqual(result, chkpnt_id)
        mock_connect_to_server.assert_awaited_once()
        mock_write_int.assert_has_calls(
            [
                call(int(ServerTask.CHECKPOINT), NUM_BYTES_FOR_IDENTIFIERS, mock_writer),
                call(dp_group_id, NUM_BYTES_FOR_IDENTIFIERS, mock_writer),
                call(node_id, NUM_BYTES_FOR_IDENTIFIERS, mock_writer),
                call(len(worker_status), NUM_BYTES_FOR_IDENTIFIERS, mock_writer),
                call(1, NUM_BYTES_FOR_IDENTIFIERS, mock_writer),
                call(2, NUM_BYTES_FOR_IDENTIFIERS, mock_writer),
                call(3, NUM_BYTES_FOR_IDENTIFIERS, mock_writer),
            ],
            any_order=False,
        )
        mock_write_utf8_string.assert_has_calls(
            [
                call(job_id, NUM_BYTES_FOR_IDENTIFIERS, mock_writer),
            ]
        )
        mock_read_utf8_string.assert_awaited_once_with(NUM_BYTES_FOR_SIZES, mock_reader, timeout=900)

    @patch("mixtera.network.connection.server_connection.ServerConnection._connect_to_server")
    @patch("mixtera.network.connection.server_connection.read_int")
    @patch("mixtera.network.connection.server_connection.write_int")
    @patch("mixtera.network.connection.server_connection.write_utf8_string")
    async def test_checkpoint_completed(
        self, mock_write_utf8_string, mock_write_int, mock_read_int, mock_connect_to_server
    ):
        """Test the _checkpoint_completed method of ServerConnection."""
        mock_reader = create_mock_reader()
        mock_writer = create_mock_writer()
        mock_connect_to_server.return_value = mock_reader, mock_writer

        job_id = "job_id"
        chkpnt_id = "checkpoint_id"
        on_disk = True
        mock_read_int.return_value = 1  # Simulate server returning success = True

        result = await self.server_connection._checkpoint_completed(job_id, chkpnt_id, on_disk)

        self.assertTrue(result)
        mock_connect_to_server.assert_awaited_once()
        mock_write_int.assert_has_calls(
            [
                call(int(ServerTask.CHECKPOINT_COMPLETED), NUM_BYTES_FOR_IDENTIFIERS, mock_writer),
                call(int(on_disk), NUM_BYTES_FOR_IDENTIFIERS, mock_writer),
            ],
            any_order=False,
        )
        mock_write_utf8_string.assert_has_calls(
            [
                call(job_id, NUM_BYTES_FOR_IDENTIFIERS, mock_writer),
                call(chkpnt_id, NUM_BYTES_FOR_IDENTIFIERS, mock_writer),
            ],
            any_order=False,
        )
        mock_read_int.assert_awaited_once_with(NUM_BYTES_FOR_IDENTIFIERS, mock_reader, timeout=120)

    @patch("mixtera.network.connection.server_connection.ServerConnection._connect_to_server")
    @patch("mixtera.network.connection.server_connection.read_int")
    @patch("mixtera.network.connection.server_connection.write_int")
    @patch("mixtera.network.connection.server_connection.write_utf8_string")
    async def test_restore_checkpoint(
        self, mock_write_utf8_string, mock_write_int, mock_read_int, mock_connect_to_server
    ):
        """Test the _restore_checkpoint method of ServerConnection."""
        mock_reader = create_mock_reader()
        mock_writer = create_mock_writer()
        mock_connect_to_server.return_value = mock_reader, mock_writer

        job_id = "job_id"
        chkpnt_id = "checkpoint_id"
        mock_read_int.return_value = 1  # Simulate server returning success = True

        await self.server_connection._restore_checkpoint(job_id, chkpnt_id)

        mock_connect_to_server.assert_awaited_once()
        mock_write_int.assert_has_calls(
            [
                call(int(ServerTask.RESTORE_CHECKPOINT), NUM_BYTES_FOR_IDENTIFIERS, mock_writer),
            ],
            any_order=False,
        )
        mock_write_utf8_string.assert_has_calls(
            [
                call(job_id, NUM_BYTES_FOR_IDENTIFIERS, mock_writer),
                call(chkpnt_id, NUM_BYTES_FOR_IDENTIFIERS, mock_writer),
            ],
            any_order=False,
        )
        mock_read_int.assert_awaited_once_with(NUM_BYTES_FOR_IDENTIFIERS, mock_reader, timeout=900)

    @patch("mixtera.network.connection.server_connection.write_int")
    @patch("mixtera.network.connection.server_connection.write_utf8_string")
    @patch("mixtera.network.connection.server_connection.write_numpy_array")
    @patch("mixtera.network.connection.server_connection.read_int")
    @patch("mixtera.network.connection.server_connection.ServerConnection._connect_to_server")
    async def test_receive_feedback(
        self,
        mock_connect_to_server,
        mock_read_int,
        mock_write_numpy_array,
        mock_write_utf8_string,
        mock_write_int,
    ):
        job_id = "test_job_id"
        feedback = ClientFeedback(
            training_steps=100,
            losses=np.array([0.1, 0.2, 0.3], dtype=np.float32),
            counts=np.array([1, 2, 3], dtype=np.int32),
        )

        mock_reader = MagicMock()
        mock_writer = create_mock_writer()
        mock_connect_to_server.return_value = (mock_reader, mock_writer)
        mock_read_int.return_value = 1  # Simulate server returning success = True

        success = await self.server_connection._receive_feedback(job_id, feedback)

        self.assertTrue(success)
        mock_connect_to_server.assert_awaited_once()
        mock_write_int.assert_has_calls(
            [
                call(int(ServerTask.RECEIVE_FEEDBACK), NUM_BYTES_FOR_IDENTIFIERS, mock_writer),
                call(feedback.training_steps, NUM_BYTES_FOR_IDENTIFIERS, mock_writer),
            ]
        )
        mock_write_utf8_string.assert_called_once_with(job_id, NUM_BYTES_FOR_IDENTIFIERS, mock_writer)
        self.assertEqual(mock_write_numpy_array.call_count, 2)
        mock_write_numpy_array.assert_has_calls(
            [
                call(feedback.losses, NUM_BYTES_FOR_IDENTIFIERS, NUM_BYTES_FOR_SIZES, mock_writer),
                call(feedback.counts, NUM_BYTES_FOR_IDENTIFIERS, NUM_BYTES_FOR_SIZES, mock_writer),
            ]
        )
        mock_read_int.assert_awaited_once_with(NUM_BYTES_FOR_IDENTIFIERS, mock_reader, timeout=300)
