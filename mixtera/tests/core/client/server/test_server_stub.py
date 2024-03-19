import unittest
from unittest.mock import MagicMock, patch

from mixtera.core.client import MixteraClient
from mixtera.core.datacollection import PropertyType
from mixtera.core.datacollection.datasets import Dataset
from mixtera.core.processing import ExecutionMode
from mixtera.core.query import Query
from mixtera.network.connection import ServerConnection

# ServerStub class code goes here...


class TestServerStub(unittest.TestCase):

    def setUp(self):
        self.host = "localhost"
        self.port = 8080
        self.server_stub = MixteraClient(self.host, self.port)

    def test_init(self):
        self.assertEqual(self.server_stub._host, "localhost")
        self.assertEqual(self.server_stub._port, 8080)
        self.assertIsInstance(self.server_stub._server_connection, ServerConnection)

    def test_is_remote(self):
        self.assertTrue(self.server_stub.is_remote())

    @patch.object(ServerConnection, "execute_query")
    def test_execute_query(self, mock_execute_query):
        query = MagicMock(spec=Query)
        chunk_size = 100
        query.job_id = "test_job_id"
        mock_execute_query.return_value = True

        result = self.server_stub.execute_query(query, chunk_size)

        mock_execute_query.assert_called_once_with(query, chunk_size)
        self.assertTrue(result)

    @patch.object(ServerConnection, "execute_query", return_value=False)
    def test_execute_query_fails(self, mock_execute_query):
        query = MagicMock(spec=Query)
        chunk_size = 100
        query.job_id = "test_job_id"

        result = self.server_stub.execute_query(query, chunk_size)

        mock_execute_query.assert_called_once_with(query, chunk_size)
        self.assertFalse(result)

    @patch.object(ServerConnection, "_stream_result_chunks")
    def test_stream_result_chunks(self, mock_stream_result_chunks):
        job_id = "test_job_id"
        mock_stream_result_chunks.return_value = iter(["chunk1", "chunk2"])
        chunks = list(self.server_stub._stream_result_chunks(job_id))

        mock_stream_result_chunks.assert_called_once_with(job_id)
        self.assertEqual(chunks, ["chunk1", "chunk2"])

    @patch.object(ServerConnection, "get_result_metadata")
    def test_get_result_metadata(self, mock_get_result_metadata):
        job_id = "test_job_id"
        expected_metadata = ({0: Dataset}, {0: MagicMock()}, {0: "path/to/file"})
        mock_get_result_metadata.return_value = expected_metadata

        metadata = self.server_stub._get_result_metadata(job_id)

        mock_get_result_metadata.assert_called_once_with(job_id)
        self.assertEqual(metadata, expected_metadata)

    # Test NotImplementedError methods to remember to add tests later
    def test_register_dataset_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.server_stub.register_dataset("id", "loc", MagicMock(), MagicMock(), "metadata_parser")

    def test_check_dataset_exists_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.server_stub.check_dataset_exists("id")

    def test_list_datasets_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.server_stub.list_datasets()

    def test_remove_dataset_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.server_stub.remove_dataset("id")

    def test_add_property_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.server_stub.add_property(
                "test_property", MagicMock(), MagicMock(), ExecutionMode.LOCAL, PropertyType.NUMERICAL
            )
