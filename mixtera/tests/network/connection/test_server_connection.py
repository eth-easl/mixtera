# pylint: disable=attribute-defined-outside-init
import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, call, patch

from mixtera.network import ID_BYTES, SAMPLE_SIZE_BYTES
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
                call(int(ServerTask.READ_FILE), ID_BYTES, mock_writer),
            ]
        )
        mock_write_utf8_string.assert_awaited_once_with(file_path, ID_BYTES, mock_writer)
        mock_read_utf8_string.assert_awaited_once_with(SAMPLE_SIZE_BYTES, mock_reader)

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
        chunk_size = 4

        success = await self.server_connection._execute_query(query_mock, chunk_size)

        self.assertTrue(success)
        mock_connect_to_server.assert_awaited_once()
        mock_write_int.assert_has_calls(
            [call(int(ServerTask.REGISTER_QUERY), ID_BYTES, mock_writer), call(chunk_size, ID_BYTES, mock_writer)]
        )
        mock_write_pickeled_object.assert_awaited_once_with(query_mock, SAMPLE_SIZE_BYTES, mock_writer)
        mock_read_int.assert_awaited_once_with(ID_BYTES, mock_reader)

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
        mock_write_int.assert_has_calls([call(int(ServerTask.GET_META_RESULT), ID_BYTES, mock_writer)])
        mock_write_string.assert_has_calls([call(job_id, ID_BYTES, mock_writer)])

        mock_read_pickeled_object.assert_awaited_once_with(SAMPLE_SIZE_BYTES, mock_reader)

    @patch("mixtera.network.connection.server_connection.ServerConnection._get_next_result")
    def test_get_query_results_sync(self, mock_get_next_result):
        mock_get_next_result.side_effect = [[1, 2, 3], [4, 5, 6], None]
        job_id = "job_id"

        results = self.server_connection._stream_result_chunks(job_id)
        result_list = list(results)

        self.assertEqual(result_list, [[1, 2, 3], [4, 5, 6]])
        mock_get_next_result.assert_has_calls([call(job_id), call(job_id), call(job_id)])

    @patch("mixtera.network.connection.server_connection.ServerConnection._connect_to_server")
    @patch("mixtera.network.connection.server_connection.read_pickeled_object")
    @patch("mixtera.network.connection.server_connection.write_int")
    @patch("mixtera.network.connection.server_connection.write_utf8_string")
    async def test_get_next_result(
        self, mock_write_string, mock_write_int, mock_read_pickeled_object, mock_connect_to_server
    ):
        mock_reader = create_mock_reader()
        mock_writer = create_mock_writer()
        mock_connect_to_server.return_value = mock_reader, mock_writer
        mock_read_pickeled_object.return_value = [1, 2, 3]
        job_id = "job_id"

        result_chunk = await self.server_connection._get_next_result(job_id)

        self.assertEqual(result_chunk, [1, 2, 3])
        mock_connect_to_server.assert_awaited_once()
        mock_write_int.assert_has_calls([call(int(ServerTask.GET_NEXT_RESULT_CHUNK), ID_BYTES, mock_writer)])
        mock_write_string.assert_has_calls([call(job_id, ID_BYTES, mock_writer)])
        mock_read_pickeled_object.assert_awaited_once_with(SAMPLE_SIZE_BYTES, mock_reader)
