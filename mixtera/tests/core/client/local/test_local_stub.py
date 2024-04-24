import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from mixtera.core.client import MixteraClient
from mixtera.core.datacollection.datasets import Dataset
from mixtera.core.datacollection.property_type import PropertyType
from mixtera.core.processing import ExecutionMode
from mixtera.core.query import Query, QueryResult


class TestLocalStub(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()  # pylint:disable=consider-using-with
        self.directory = Path(self.temp_dir.name)
        self.local_stub = MixteraClient.from_directory(self.directory)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_init_with_valid_directory(self):
        self.assertIsInstance(self.local_stub, MixteraClient)

    def test_init_with_invalid_directory(self):
        with self.assertRaises(RuntimeError):
            MixteraClient.from_directory("/non/existent/directory/path")

    @patch("mixtera.core.datacollection.MixteraDataCollection.register_dataset")
    def test_register_dataset(self, mock_register):
        mock_register.return_value = True
        result = self.local_stub.register_dataset("id", "loc", MagicMock(), MagicMock(), "metadata_parser")
        self.assertTrue(result)
        mock_register.assert_called_once()

    @patch("mixtera.core.datacollection.MixteraDataCollection.check_dataset_exists")
    def test_check_dataset_exists(self, mock_check):
        mock_check.return_value = True
        self.assertTrue(self.local_stub.check_dataset_exists("id"))
        mock_check.assert_called_once_with("id")

    @patch("mixtera.core.datacollection.MixteraDataCollection.list_datasets")
    def test_list_datasets(self, mock_list):
        mock_list.return_value = ["dataset1", "dataset2"]
        self.assertEqual(self.local_stub.list_datasets(), ["dataset1", "dataset2"])

    @patch("mixtera.core.datacollection.MixteraDataCollection.remove_dataset")
    def test_remove_dataset(self, mock_remove):
        mock_remove.return_value = True
        self.assertTrue(self.local_stub.remove_dataset("id"))
        mock_remove.assert_called_once_with("id")

    @patch("mixtera.core.datacollection.MixteraDataCollection")
    def test_execute_query(self, mock_mdc):
        query = MagicMock(spec=Query)
        chunk_size = 100
        query.job_id = "test_job_id"
        self.local_stub._mdc = mock_mdc
        self.local_stub._register_query = MagicMock(return_value=True)

        result = self.local_stub.execute_query(query, chunk_size=chunk_size, mixture=None)

        query.execute.assert_called_once_with(mock_mdc, chunk_size=chunk_size, mixture=None)
        self.local_stub._register_query.assert_called_once_with(query, chunk_size)
        self.assertTrue(result)

    @patch("mixtera.core.datacollection.MixteraDataCollection")
    def test_execute_query_registration_fails(self, mock_mdc):
        query = MagicMock(spec=Query)
        chunk_size = 100
        query.job_id = "test_job_id"
        self.local_stub._register_query = MagicMock(return_value=False)
        self.local_stub._mdc = mock_mdc
        result = self.local_stub.execute_query(query, chunk_size=chunk_size, mixture=None)

        query.execute.assert_called_once_with(mock_mdc, chunk_size=chunk_size, mixture=None)
        self.local_stub._register_query.assert_called_once_with(query, chunk_size)
        self.assertFalse(result)

    def test_is_remote(self):
        self.assertFalse(self.local_stub.is_remote())

    @patch("mixtera.core.datacollection.MixteraDataCollection.add_property")
    def test_add_property(self, mock_add_property):
        setup_func = MagicMock()
        calc_func = MagicMock()
        execution_mode = ExecutionMode.LOCAL
        property_type = PropertyType.NUMERICAL
        self.local_stub.add_property("test_property", setup_func, calc_func, execution_mode, property_type)

        mock_add_property.assert_called_once_with(
            "test_property",
            setup_func,
            calc_func,
            execution_mode,
            property_type,
            min_val=0.0,
            max_val=1,
            num_buckets=10,
            batch_size=1,
            dop=1,
            data_only_on_primary=True,
        )

    @patch("mixtera.core.datacollection.MixteraDataCollection")
    def test_stream_result_chunks(self, mock_mdc):
        del mock_mdc
        job_id = "test_job_id"
        query_result = MagicMock(spec=QueryResult)
        self.local_stub._get_query_result = MagicMock(return_value=query_result)
        chunks = list(self.local_stub._stream_result_chunks(job_id))
        self.local_stub._get_query_result.assert_called_once_with(job_id)
        self.assertEqual(chunks, list(query_result))

    @patch("mixtera.core.datacollection.MixteraDataCollection")
    def test_get_result_metadata(self, mock_mdc):
        del mock_mdc
        job_id = "test_job_id"
        dataset_type = {0: Dataset}
        parsing_func = {0: MagicMock()}
        file_path = {0: "path/to/file"}
        query_result = MagicMock(
            spec=QueryResult, dataset_type=dataset_type, parsing_func=parsing_func, file_path=file_path
        )
        self.local_stub._get_query_result = MagicMock(return_value=query_result)
        result = self.local_stub._get_result_metadata(job_id)
        self.local_stub._get_query_result.assert_called_once_with(job_id)
        self.assertEqual(result, (dataset_type, parsing_func, file_path))

    @patch("mixtera.core.client.local.local_stub.wait_for_key_in_dict")
    def test_register_query_already_exists(self, mock_wait_for_key):
        query = MagicMock(spec=Query)
        query.job_id = "test_job_id"
        self.local_stub._training_query_map[query.job_id] = (query, 100)

        result = self.local_stub._register_query(query, 100)
        mock_wait_for_key.assert_not_called()
        self.assertFalse(result)

    @patch("mixtera.core.client.local.local_stub.wait_for_key_in_dict")
    def test_register_query_new(self, mock_wait_for_key):
        query = MagicMock(spec=Query)
        query.job_id = "new_test_job_id"
        self.local_stub._training_query_map = {}

        result = self.local_stub._register_query(query, 100)
        mock_wait_for_key.assert_not_called()
        self.assertTrue(result)
        self.assertIn(query.job_id, self.local_stub._training_query_map)

    @patch("mixtera.core.client.local.local_stub.wait_for_key_in_dict", return_value=False)
    def test_get_query_result_timeout(self, mock_wait_for_key):
        del mock_wait_for_key
        with self.assertRaises(RuntimeError):
            self.local_stub._get_query_result("non_existent_job_id")

    @patch("mixtera.core.client.local.local_stub.wait_for_key_in_dict", return_value=True)
    def test_get_query_result_success(self, mock_wait_for_key):
        job_id = "test_job_id"
        query = MagicMock(spec=Query)
        query.results = MagicMock(spec=QueryResult)
        self.local_stub._training_query_map[job_id] = (query, 100)

        result = self.local_stub._get_query_result(job_id)
        mock_wait_for_key.assert_called_once_with(self.local_stub._training_query_map, job_id, 60.0)
        self.assertEqual(result, query.results)
