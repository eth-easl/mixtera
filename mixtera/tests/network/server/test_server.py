# pylint: disable=attribute-defined-outside-init
import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import ANY, AsyncMock, MagicMock, call, patch

import numpy as np

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

    @patch("mixtera.network.server.server.write_utf8_string")
    @patch("mixtera.network.server.server.read_pickeled_object")
    @patch("mixtera.network.server.server.read_int")
    async def test_register_query(self, mock_read_int, mock_read_pickeled_object, mock_write_utf8_string):
        query_mock = MagicMock()
        query_mock.job_id = "cool_training_id"
        mock_read_pickeled_object.side_effect = [MagicMock(), query_mock]  # mixture, query
        mock_read_int.side_effect = [1, 2, 3]  # dp_groups, nodes_per_group, num_workers
        mock_writer = create_mock_writer()
        self.server._local_stub._query_cache.enabled = False  # mocks break pickle

        await self.server._register_query(create_mock_reader(b""), mock_writer)

        mock_read_pickeled_object.assert_awaited_with(NUM_BYTES_FOR_SIZES, ANY)
        self.assertEqual(mock_read_pickeled_object.await_count, 2)

        mock_read_int.assert_awaited_with(NUM_BYTES_FOR_IDENTIFIERS, ANY)
        self.assertEqual(mock_read_int.await_count, 3)

        mock_write_utf8_string.assert_awaited_once_with(query_mock.job_id, NUM_BYTES_FOR_IDENTIFIERS, mock_writer)

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

    @patch("mixtera.network.server.server.write_bytes_obj")
    @patch("mixtera.network.server.server.read_int")
    @patch("mixtera.network.server.server.read_utf8_string")
    @patch("mixtera.core.client.local.LocalStub._get_query_chunk_distributor")
    async def test_get_next_result_chunk(
        self, mock_get_query_chunk_distributor, mock_read_utf8_string, mock_read_int, mock_write_bytes_obj
    ):
        job_id = "itsamememario"
        mock_read_int.side_effect = [int(ServerTask.GET_NEXT_RESULT_CHUNK), 1, 2, 3] * 2
        mock_read_utf8_string.return_value = job_id
        mock_get_query_chunk_distributor.return_value.next_chunk_for = MagicMock()
        mock_get_query_chunk_distributor.return_value.next_chunk_for.side_effect = [1, 2]
        mock_writer = create_mock_writer()

        await self.server._dispatch_client(create_mock_reader(b""), mock_writer)
        mock_get_query_chunk_distributor.assert_called_once_with(job_id)
        mock_write_bytes_obj.assert_awaited_once_with(1, NUM_BYTES_FOR_SIZES, mock_writer)

        await self.server._dispatch_client(create_mock_reader(b""), mock_writer)

        mock_get_query_chunk_distributor.assert_called_once_with(job_id)
        expected_calls = [
            call(1, NUM_BYTES_FOR_SIZES, mock_writer),  # The first call
            call(2, NUM_BYTES_FOR_SIZES, mock_writer),  # The second call
        ]
        mock_write_bytes_obj.assert_has_calls(expected_calls)
        assert mock_write_bytes_obj.await_count == 2

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

    @patch("mixtera.network.server.server.write_utf8_string")
    @patch("mixtera.network.server.server.write_int")
    @patch("mixtera.network.server.server.read_int")
    @patch("mixtera.network.server.server.read_utf8_string")
    async def test_checkpoint(self, mock_read_utf8_string, mock_read_int, mock_write_int, mock_write_utf8_string):
        """Test the _checkpoint method of MixteraServer."""
        del mock_write_int
        # Setup mocks
        job_id = "test_job_id"
        dp_group_id = 0
        node_id = 0
        worker_status = [1, 2, 3]
        chkpnt_id = "test_checkpoint_id"

        mock_read_utf8_string.return_value = job_id
        mock_read_int.side_effect = [dp_group_id, node_id, len(worker_status)] + worker_status

        # Mock the checkpoint method
        self.server._local_stub.checkpoint = MagicMock(return_value=chkpnt_id)

        mock_reader = MagicMock()
        mock_writer = create_mock_writer()

        # Call the method
        await self.server._checkpoint(mock_reader, mock_writer)

        # Assertions
        self.server._local_stub.checkpoint.assert_called_once_with(
            job_id, dp_group_id, node_id, worker_status, server=True
        )
        mock_write_utf8_string.assert_awaited_once_with(chkpnt_id, NUM_BYTES_FOR_SIZES, mock_writer)

    @patch("mixtera.network.server.server.write_int")
    @patch("mixtera.network.server.server.read_int")
    @patch("mixtera.network.server.server.read_utf8_string")
    async def test_checkpoint_completed(self, mock_read_utf8_string, mock_read_int, mock_write_int):
        """Test the _checkpoint_completed method of MixteraServer."""
        # Setup mocks
        job_id = "test_job_id"
        chkpnt_id = "test_checkpoint_id"
        on_disk_flag = 1  # True
        is_completed = True

        mock_read_utf8_string.side_effect = [job_id, chkpnt_id]
        mock_read_int.return_value = on_disk_flag

        # Mock the checkpoint_completed method
        self.server._local_stub.checkpoint_completed = MagicMock(return_value=is_completed)

        mock_reader = MagicMock()
        mock_writer = create_mock_writer()

        # Call the method
        await self.server._checkpoint_completed(mock_reader, mock_writer)

        # Assertions
        self.server._local_stub.checkpoint_completed.assert_called_once_with(job_id, chkpnt_id, bool(on_disk_flag))
        mock_write_int.assert_awaited_once_with(int(is_completed), NUM_BYTES_FOR_IDENTIFIERS, mock_writer)

    @patch("mixtera.network.server.server.write_utf8_string")
    @patch("mixtera.network.server.server.read_utf8_string")
    async def test_restore_checkpoint(self, mock_read_utf8_string, mock_write_utf8_string):
        job_id = "test_job_id"
        chkpnt_id = "test_checkpoint_id"
        mock_read_utf8_string.side_effect = [job_id, chkpnt_id]

        mock_reader = MagicMock()
        mock_writer = create_mock_writer()

        # Mock the _background_restore_checkpoint method
        self.server._background_restore_checkpoint = AsyncMock()

        # Call the method under test
        await self.server._restore_checkpoint(mock_reader, mock_writer)

        # Assert that write_utf8_string was called with job_id
        mock_write_utf8_string.assert_awaited_once_with(job_id, NUM_BYTES_FOR_IDENTIFIERS, mock_writer)

        # Assert that _background_restore_checkpoint was called with the correct arguments
        self.server._background_restore_checkpoint.assert_called_once_with(job_id, chkpnt_id)

    @patch("mixtera.network.server.server.read_utf8_string")
    @patch("mixtera.network.server.server.read_int")
    @patch("mixtera.network.server.server.read_numpy_array")
    @patch("mixtera.network.server.server.write_int")
    async def test_process_feedback(
        self,
        mock_write_int,
        mock_read_numpy_array,
        mock_read_int,
        mock_read_utf8_string,
    ):
        """Test the _process_feedback method of MixteraServer."""

        # Setup mocks
        job_id = "test_job_id"
        training_steps = 100
        losses = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        counts = np.array([1, 2, 3], dtype=np.int32)

        mock_read_utf8_string.return_value = job_id
        mock_read_int.return_value = training_steps
        mock_read_numpy_array.side_effect = [losses, counts]

        self.server._local_stub.process_feedback = MagicMock()

        mock_reader = MagicMock()
        mock_writer = create_mock_writer()

        # Call the method
        await self.server._process_feedback(mock_reader, mock_writer)

        # Assertions
        mock_read_utf8_string.assert_awaited_once_with(NUM_BYTES_FOR_IDENTIFIERS, mock_reader)
        self.assertEqual(mock_read_int.await_count, 2)
        self.assertEqual(mock_read_numpy_array.await_count, 2)
        self.server._local_stub.process_feedback.assert_called_once()
        args = self.server._local_stub.process_feedback.call_args[0]
        self.assertEqual(args[0], job_id)
        feedback = args[1]
        self.assertEqual(feedback.training_steps, training_steps)
        np.testing.assert_array_equal(feedback.losses, losses)
        np.testing.assert_array_equal(feedback.counts, counts)
        self.assertEqual(mock_write_int.await_count, 1)
