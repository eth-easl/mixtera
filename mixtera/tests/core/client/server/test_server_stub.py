import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from mixtera.core.client import MixteraClient
from mixtera.core.datacollection import PropertyType
from mixtera.core.datacollection.datasets import Dataset
from mixtera.core.processing import ExecutionMode
from mixtera.core.query import Query
from mixtera.network.connection import ServerConnection


class TestServerStub(unittest.TestCase):
    def setUp(self):
        self.host = "localhost"
        self.port = 8080
        self.server_stub = MixteraClient.from_remote(self.host, self.port)

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

        result = self.server_stub.execute_query(query, chunk_size, 1, 2, 3)

        mock_execute_query.assert_called_once_with(query, chunk_size, 1, 2, 3)
        self.assertTrue(result)

    @patch.object(ServerConnection, "execute_query", return_value=False)
    def test_execute_query_fails(self, mock_execute_query):
        query = MagicMock(spec=Query)
        chunk_size = 100
        query.job_id = "test_job_id"

        result = self.server_stub.execute_query(query, chunk_size, 1, 2, 3)

        mock_execute_query.assert_called_once_with(query, chunk_size, 1, 2, 3)
        self.assertFalse(result)

    @patch.object(ServerConnection, "_stream_result_chunks")
    def test_stream_result_chunks(self, mock_stream_result_chunks):
        job_id = "test_job_id"
        mock_stream_result_chunks.return_value = iter(["chunk1", "chunk2"])
        chunks = list(self.server_stub._stream_result_chunks(job_id, 1, 1, 1))

        mock_stream_result_chunks.assert_called_once_with(job_id, 1, 1, 1)
        self.assertEqual(chunks, ["chunk1", "chunk2"])

    @patch.object(ServerConnection, "get_result_metadata")
    def test_get_result_metadata(self, mock_get_result_metadata):
        job_id = "test_job_id"
        expected_metadata = ({0: Dataset}, {0: MagicMock()}, {0: "path/to/file"})
        mock_get_result_metadata.return_value = expected_metadata

        metadata = self.server_stub._get_result_metadata(job_id)

        mock_get_result_metadata.assert_called_once_with(job_id)
        self.assertEqual(metadata, expected_metadata)

    @patch.object(ServerConnection, "register_dataset")
    def test_register_dataset(self, mock_register_dataset):
        identifier = "test_id"
        loc = "test_loc"
        dtype = MagicMock()
        dtype.type = 0
        parsing_func = MagicMock()
        metadata_parser_identifier = "test_metadata_parser"

        result = self.server_stub.register_dataset(identifier, loc, dtype, parsing_func, metadata_parser_identifier)

        mock_register_dataset.assert_called_once_with(
            identifier, loc, dtype.type, parsing_func, metadata_parser_identifier
        )
        self.assertTrue(result)

    @patch.object(ServerConnection, "register_dataset")
    def test_register_dataset_loc_path(self, mock_register_dataset):
        identifier = "test_id"
        loc = Path("test_loc")
        dtype = MagicMock()
        dtype.type = 0
        parsing_func = MagicMock()
        metadata_parser_identifier = "test_metadata_parser"

        result = self.server_stub.register_dataset(identifier, loc, dtype, parsing_func, metadata_parser_identifier)

        mock_register_dataset.assert_called_once_with(
            identifier, str(loc), dtype.type, parsing_func, metadata_parser_identifier
        )
        self.assertTrue(result)

    @patch.object(ServerConnection, "register_metadata_parser")
    def test_register_metadata_parser(self, mock_register_metadata_parser):
        identifier = "test_id"
        parser = MagicMock()
        self.server_stub._server_connection.register_metadata_parser = mock_register_metadata_parser

        result = self.server_stub.register_metadata_parser(identifier, parser)

        mock_register_metadata_parser.assert_called_once_with(identifier, parser)
        self.assertTrue(result)

    @patch.object(ServerConnection, "check_dataset_exists")
    def test_check_dataset_exists(self, mock_check_dataset_exists):
        identifier = "test_id"

        result = self.server_stub.check_dataset_exists(identifier)

        mock_check_dataset_exists.assert_called_once_with(identifier)
        self.assertTrue(result)

    @patch.object(ServerConnection, "list_datasets")
    def test_list_datasets(self, mock_list_datasets):
        mock_list_datasets.return_value = ["test1", "test2"]

        datasets = self.server_stub.list_datasets()

        mock_list_datasets.assert_called_once()
        self.assertEqual(datasets, ["test1", "test2"])

    @patch.object(ServerConnection, "remove_dataset")
    def test_remove_dataset(self, mock_remove_dataset):
        identifier = "test_id"

        result = self.server_stub.remove_dataset(identifier)

        mock_remove_dataset.assert_called_once_with(identifier)
        self.assertTrue(result)

    @patch.object(ServerConnection, "add_property")
    def test_add_property(self, mock_add_property):
        property_name = "test_property"
        setup_func = MagicMock()
        calc_func = MagicMock()
        execution_mode = ExecutionMode.LOCAL
        property_type = PropertyType.NUMERICAL

        self.server_stub.add_property(
            property_name, setup_func, calc_func, execution_mode, property_type, 0.1, 0.9, 8, 2, 2, False
        )

        mock_add_property.assert_called_once_with(
            property_name,
            setup_func,
            calc_func,
            execution_mode,
            property_type,
            min_val=0.1,
            max_val=0.9,
            num_buckets=8,
            batch_size=2,
            degree_of_parallelism=2,
            data_only_on_primary=False,
        )
