# pylint: disable=attribute-defined-outside-init
import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import ANY, AsyncMock, MagicMock, call, patch

from mixtera.network import ID_BYTES, SAMPLE_SIZE_BYTES
from mixtera.network.server import MixteraServer
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


class TestMixteraServer(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory_obj = tempfile.TemporaryDirectory()
        self.directory = Path(self.directory_obj.name)
        self.host = "localhost"
        self.port = 12345
        self.server = MixteraServer(directory=self.directory, host=self.host, port=self.port)
        self.server._ldc = MagicMock()

    async def asyncTearDown(self):
        self.directory_obj.cleanup()

    @patch("mixtera.network.server.server.write_int")
    @patch("mixtera.network.server.server.read_pickeled_object")
    @patch("mixtera.network.server.server.read_int")
    async def test_register_query(self, mock_read_int, mock_read_pickeled_object, mock_write_int):
        mock_read_int.return_value = AsyncMock(return_value=4)
        query_mock = MagicMock()
        query_mock.query_id = 42
        mock_read_pickeled_object.return_value = query_mock
        mock_writer = create_mock_writer()

        await self.server._register_query(create_mock_reader(b""), mock_writer)

        mock_read_int.assert_awaited_once_with(ID_BYTES, ANY)
        mock_read_pickeled_object.assert_awaited_once_with(SAMPLE_SIZE_BYTES, ANY)
        mock_write_int.assert_awaited_once_with(query_mock.query_id, ID_BYTES, mock_writer)

    @patch("mixtera.network.server.server.read_int")
    async def test_parse_ids(self, mock_read_int):
        mock_read_int.side_effect = [1, 2, 3]
        mock_reader = create_mock_reader()

        node_id, worker_id, query_id = await self.server._parse_ids(mock_reader)

        self.assertEqual(query_id, 1)
        self.assertEqual(node_id, 2)
        self.assertEqual(worker_id, 3)
        mock_read_int.assert_has_calls([call(ID_BYTES, mock_reader)] * 3)

    @patch("mixtera.network.server.server.read_utf8_string")
    async def test_get_query_id(self, mock_read_utf8_string):
        training_id = "test_training_id"
        expected_query_id = 42
        self.server._ldc.get_query_id.return_value = expected_query_id
        mock_read_utf8_string.return_value = training_id
        mock_reader = create_mock_reader()

        query_id = await self.server._get_query_id(mock_reader)

        self.assertEqual(query_id, expected_query_id)
        mock_read_utf8_string.assert_awaited_once_with(SAMPLE_SIZE_BYTES, mock_reader)
        self.server._ldc.get_query_id.assert_called_once_with(training_id)

    @patch("mixtera.network.server.server.write_utf8_string")
    @patch("mixtera.network.server.server.read_utf8_string")
    @patch("mixtera.network.server.server.read_int")
    async def test_read_file(self, mock_from_id, mock_read_int, mock_read_utf8_string, mock_write_utf8_string):
        filesystem_mock = MagicMock()
        mock_from_id.return_value = filesystem_mock
        file_path = "/path/to/file"
        file_data = "file_data"
        filesystem_mock.get_file_iterable.return_value = [file_data]
        mock_read_int.return_value = 1
        mock_read_utf8_string.return_value = file_path
        mock_writer = create_mock_writer()

        await self.server._read_file(create_mock_reader(b"", file_path.encode()), mock_writer)

        mock_read_int.assert_awaited_once_with(ID_BYTES, ANY)
        mock_read_utf8_string.assert_awaited_once_with(ID_BYTES, ANY)
        mock_from_id.assert_called_once_with(1)
        filesystem_mock.get_file_iterable.assert_called_once_with(file_path, None)
        mock_write_utf8_string.assert_awaited_once_with(file_data, SAMPLE_SIZE_BYTES, mock_writer, drain=False)
        mock_writer.drain.assert_awaited_once()

    @patch("mixtera.network.server.server.write_pickeled_object")
    @patch("mixtera.network.server.server.read_int")
    async def test_get_meta_result(self, mock_read_int, mock_write_pickeled_object):
        query_id = 42
        meta_result = MagicMock()
        self.server._ldc._queries = {query_id: [MagicMock(results=MagicMock(_meta=meta_result))]}
        mock_read_int.side_effect = [int(ServerTask.GET_META_RESULT), query_id]
        mock_writer = create_mock_writer()

        await self.server._dispatch_client(create_mock_reader(b""), mock_writer)
        mock_write_pickeled_object.assert_awaited_once_with(meta_result, SAMPLE_SIZE_BYTES, mock_writer)

    @patch("mixtera.network.server.server.write_pickeled_object")
    @patch("mixtera.network.server.server.read_int")
    async def test_get_next_result_chunk(self, mock_read_int, mock_write_pickeled_object):
        query_id = 42
        result_chunk = MagicMock()
        self.server._ldc.next_query_result_chunk.return_value = result_chunk
        mock_read_int.side_effect = [int(ServerTask.GET_NEXT_RESULT_CHUNK), query_id]
        mock_writer = create_mock_writer()

        await self.server._dispatch_client(create_mock_reader(b""), mock_writer)

        self.server._ldc.next_query_result_chunk.assert_called_once_with(query_id)
        mock_write_pickeled_object.assert_awaited_once_with(result_chunk, SAMPLE_SIZE_BYTES, mock_writer)

    @patch("mixtera.network.server.server.MixteraServer._dispatch_client")
    async def test_run_async(self, mock_dispatch_client):
        mock_server = AsyncMock()
        mock_server.serve_forever = AsyncMock()
        mock_server.wait_closed = AsyncMock()
        mock_server.close = MagicMock()
        with patch("asyncio.start_server", return_value=mock_server):
            await self.server._run_async()

        mock_server.serve_forever.assert_awaited_once()
        mock_server.wait_closed.assert_awaited_once()
        mock_dispatch_client.assert_not_awaited()
