# pylint: disable=attribute-defined-outside-init
import asyncio
import tempfile
import unittest
from pathlib import Path
from typing import Generator
from unittest.mock import ANY, AsyncMock, MagicMock, call, patch

from mixtera.network import NUM_BYTES_FOR_IDENTIFIERS, NUM_BYTES_FOR_SIZES
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
        self.directory_obj = tempfile.TemporaryDirectory()  # pylint: disable = consider-using-with
        self.directory = Path(self.directory_obj.name)
        self.host = "localhost"
        self.port = 12345
        self.server = MixteraServer(directory=self.directory, host=self.host, port=self.port)
        self.server._ldc = MagicMock()

    async def asyncTearDown(self):
        self.directory_obj.cleanup()

    @patch("mixtera.network.server.server.write_int")
    @patch("mixtera.network.server.server.read_pickeled_object")
    async def test_register_query(self, mock_read_pickeled_object, mock_write_int):
        query_mock = MagicMock()
        query_mock.job_id = "cool_training_id"
        mock_read_pickeled_object.return_value = query_mock
        mock_writer = create_mock_writer()

        await self.server._register_query(create_mock_reader(b""), mock_writer)

        mock_read_pickeled_object.assert_awaited_with(NUM_BYTES_FOR_SIZES, ANY)
        self.assertEqual(mock_read_pickeled_object.await_count, 2)
        mock_write_int.assert_awaited_once_with(True, NUM_BYTES_FOR_IDENTIFIERS, mock_writer)

    @patch("mixtera.network.server.server.write_utf8_string")
    @patch("mixtera.network.server.server.read_utf8_string")
    @patch("mixtera.network.server.server.read_int")
    @patch("mixtera.core.filesystem.FileSystem.from_path")
    async def test_read_file(self, mock_from_path, mock_read_int, mock_read_utf8_string, mock_write_utf8_string):
        filesystem_mock = MagicMock()
        mock_from_path.return_value = filesystem_mock
        file_path = "/path/to/file"
        file_data = "file_data"
        filesystem_mock.get_file_iterable.return_value = [file_data]
        mock_read_int.return_value = 1
        mock_read_utf8_string.return_value = file_path
        mock_writer = create_mock_writer()

        await self.server._read_file(create_mock_reader(b"", file_path.encode()), mock_writer)

        mock_read_utf8_string.assert_awaited_once_with(NUM_BYTES_FOR_IDENTIFIERS, ANY)
        mock_from_path.assert_called_once_with("/path/to/file")
        filesystem_mock.get_file_iterable.assert_called_once_with(file_path)
        mock_write_utf8_string.assert_awaited_once_with(file_data, NUM_BYTES_FOR_SIZES, mock_writer, drain=False)
        mock_writer.drain.assert_awaited_once()

    @patch("mixtera.network.server.server.write_pickeled_object")
    @patch("mixtera.network.server.server.read_int")
    @patch("mixtera.network.server.server.read_utf8_string")
    @patch("mixtera.core.client.local.LocalStub._get_result_metadata")
    async def test_get_meta_result(
        self, mock_get_result_metadata, mock_read_utf8_string, mock_read_int, mock_write_pickeled_object
    ):
        job_id = "job_id"
        mock_get_result_metadata.return_value = (1, 2, 3)
        mock_read_int.return_value = int(ServerTask.GET_META_RESULT)
        mock_read_utf8_string.return_value = job_id
        mock_writer = create_mock_writer()

        await self.server._dispatch_client(create_mock_reader(b""), mock_writer)
        mock_write_pickeled_object.assert_awaited_once_with(
            {
                "dataset_type": 1,
                "parsing_func": 2,
                "file_path": 3,
            },
            NUM_BYTES_FOR_SIZES,
            mock_writer,
        )

    @patch("mixtera.network.server.server.write_pickeled_object")
    @patch("mixtera.network.server.server.read_int")
    @patch("mixtera.network.server.server.read_utf8_string")
    @patch("mixtera.core.client.local.LocalStub._get_query_result")
    async def test_get_next_result_chunk(
        self, mock_get_query_result, mock_read_utf8_string, mock_read_int, mock_write_pickeled_object
    ):
        def sample_generator() -> Generator[int, None, None]:
            yield 1
            yield 2

        job_id = "itsamemario"
        mock_read_int.return_value = int(ServerTask.GET_NEXT_RESULT_CHUNK)
        mock_read_utf8_string.return_value = job_id
        mock_get_query_result.return_value = sample_generator()
        mock_writer = create_mock_writer()

        await self.server._dispatch_client(create_mock_reader(b""), mock_writer)
        mock_get_query_result.assert_called_once_with(job_id)
        mock_write_pickeled_object.assert_awaited_once_with(1, NUM_BYTES_FOR_SIZES, mock_writer)

        await self.server._dispatch_client(create_mock_reader(b""), mock_writer)
        mock_get_query_result.assert_called_once_with(job_id)
        expected_calls = [
            call(1, NUM_BYTES_FOR_SIZES, mock_writer),  # The first call
            call(2, NUM_BYTES_FOR_SIZES, mock_writer),  # The second call
        ]
        mock_write_pickeled_object.assert_has_calls(expected_calls)
        assert mock_write_pickeled_object.await_count == 2

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
