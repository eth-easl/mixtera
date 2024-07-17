import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from mixtera.core.client import MixteraClient
from mixtera.core.datacollection import MixteraDataCollection
from mixtera.core.datacollection.datasets import Dataset
from mixtera.core.datacollection.property_type import PropertyType
from mixtera.core.processing import ExecutionMode
from mixtera.core.query import Query, QueryResult, StaticMixture
from mixtera.core.query.chunk_distributor import ChunkDistributor


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

    @patch.object(MixteraDataCollection, "register_dataset")
    def test_register_dataset(self, mock_register):
        mock_register.return_value = True
        result = self.local_stub.register_dataset("id", "loc", MagicMock(), MagicMock(), "metadata_parser")
        self.assertTrue(result)
        mock_register.assert_called_once()

    @patch.object(MixteraDataCollection, "register_dataset")
    def test_register_dataset_loc_path(self, mock_register_dataset):
        identifier = "test_id"
        loc = Path("test_loc")
        dtype = MagicMock()
        dtype.type = 0
        parsing_func = MagicMock()
        metadata_parser_identifier = "test_metadata_parser"

        result = self.local_stub.register_dataset(identifier, loc, dtype, parsing_func, metadata_parser_identifier)

        mock_register_dataset.assert_called_once_with(
            identifier, str(loc), dtype, parsing_func, metadata_parser_identifier
        )
        self.assertTrue(result)

    @patch.object(MixteraDataCollection, "check_dataset_exists")
    def test_check_dataset_exists(self, mock_check):
        mock_check.return_value = True
        self.assertTrue(self.local_stub.check_dataset_exists("id"))
        mock_check.assert_called_once_with("id")

    @patch.object(MixteraDataCollection, "list_datasets")
    def test_list_datasets(self, mock_list):
        mock_list.return_value = ["dataset1", "dataset2"]
        self.assertEqual(self.local_stub.list_datasets(), ["dataset1", "dataset2"])

    @patch.object(MixteraDataCollection, "remove_dataset")
    def test_remove_dataset(self, mock_remove):
        mock_remove.return_value = True
        self.assertTrue(self.local_stub.remove_dataset("id"))
        mock_remove.assert_called_once_with("id")

    @patch("mixtera.core.datacollection.MixteraDataCollection")
    def test_execute_query(self, mock_mdc):
        query = MagicMock(spec=Query)
        chunk_size = 100
        mixture = StaticMixture(chunk_size, {"any": 1.0})
        query.job_id = "test_job_id"
        self.local_stub._mdc = mock_mdc
        self.local_stub._register_query = MagicMock(return_value=True)

        result = self.local_stub.execute_query(query, mixture, 1, 1, 1)
        query.execute.assert_called_once_with(mock_mdc, mixture)
        self.local_stub._register_query.assert_called_once_with(query, mixture, 1, 1, 1)
        self.assertTrue(result)

    @patch("mixtera.core.datacollection.MixteraDataCollection")
    def test_execute_query_registration_fails(self, mock_mdc):
        query = MagicMock(spec=Query)
        chunk_size = 100
        mixture = StaticMixture(chunk_size, {"any": 1.0})
        query.job_id = "test_job_id"
        self.local_stub._register_query = MagicMock(return_value=False)
        self.local_stub._mdc = mock_mdc
        result = self.local_stub.execute_query(query, mixture, 1, 1, 1)

        query.execute.assert_called_once_with(mock_mdc, mixture)
        self.local_stub._register_query.assert_called_once_with(query, mixture, 1, 1, 1)
        self.assertFalse(result)

    def test_is_remote(self):
        self.assertFalse(self.local_stub.is_remote())

    @patch.object(MixteraDataCollection, "add_property")
    def test_add_property(self, mock_add_property):
        setup_func = MagicMock()
        calc_func = MagicMock()
        execution_mode = ExecutionMode.LOCAL
        property_type = PropertyType.NUMERICAL
        result = self.local_stub.add_property("test_property", setup_func, calc_func, execution_mode, property_type)

        self.assertTrue(result)
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
            degree_of_parallelism=1,
            data_only_on_primary=True,
        )

    @patch("mixtera.core.datacollection.MixteraDataCollection")
    def test_stream_result_chunks(self, mock_mdc):
        del mock_mdc
        job_id = "test_job_id"
        chunk_distributor = MagicMock(spec=ChunkDistributor)
        self.local_stub._get_query_chunk_distributor = MagicMock(return_value=chunk_distributor)
        chunks = list(self.local_stub._stream_result_chunks(job_id, 1, 1, 1))
        self.local_stub._get_query_chunk_distributor.assert_called_once_with(job_id)
        self.assertEqual(chunks, [])

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

        result = self.local_stub._register_query(query, 100, 1, 1, 1)
        mock_wait_for_key.assert_not_called()
        self.assertFalse(result)

    @patch("mixtera.core.client.local.local_stub.wait_for_key_in_dict")
    def test_register_query_new(self, mock_wait_for_key):
        query = MagicMock(spec=Query)
        query.results = MagicMock(spec=QueryResult)
        query.job_id = "new_test_job_id"
        self.local_stub._training_query_map = {}

        result = self.local_stub._register_query(query, 100, 1, 1, 1)
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
        self.local_stub._training_query_map[job_id] = (ChunkDistributor(1, 1, 1, query.results, "test_job"), query, 100)

        result = self.local_stub._get_query_result(job_id)
        mock_wait_for_key.assert_called_once_with(self.local_stub._training_query_map, job_id, 60.0)
        self.assertEqual(result, query.results)
