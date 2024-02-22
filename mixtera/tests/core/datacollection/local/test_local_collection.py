import json
import sqlite3
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from unittest.mock import ANY, MagicMock, patch

from mixtera.core.datacollection import PropertyType
from mixtera.core.datacollection.datasets.jsonl_dataset import JSONLDataset
from mixtera.core.datacollection.local import LocalDataCollection
from mixtera.core.processing import ExecutionMode
from mixtera.utils import defaultdict_to_dict


class TestLocalDataCollection(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with

    def tearDown(self):
        self.temp_dir.cleanup()

    @patch("sqlite3.connect")
    @patch("mixtera.core.datacollection.local.LocalDataCollection._init_database")
    def test_init_with_non_existing_database(self, mock_init_database: MagicMock, mock_connect: MagicMock):
        mock_connection = MagicMock()
        mock_init_database.return_value = mock_connection
        mock_connect.return_value = mock_connection

        directory = Path(self.temp_dir.name)
        ldc = LocalDataCollection(directory)

        mock_init_database.assert_called_once()
        mock_connect.assert_not_called()
        self.assertEqual(ldc._connection, mock_connection)

    @patch("sqlite3.connect")
    @patch("mixtera.core.datacollection.local.LocalDataCollection._init_database")
    def test_init_with_existing_database(self, mock_init_database: MagicMock, mock_connect: MagicMock):
        mock_connection = MagicMock()
        mock_connect.return_value = mock_connection

        directory = Path(self.temp_dir.name)
        (directory / "mixtera.sqlite").touch()
        self.assertTrue((directory / "mixtera.sqlite").exists())
        ldc = LocalDataCollection(directory)

        mock_connect.assert_called_once_with(directory / "mixtera.sqlite")
        mock_init_database.assert_not_called()
        self.assertEqual(ldc._connection, mock_connection)

    @patch("sqlite3.connect")
    def test_init_database_with_mocked_sqlite(self, mock_connect):
        directory = Path(self.temp_dir.name)
        original_init_database = LocalDataCollection._init_database
        LocalDataCollection._init_database = lambda self: None
        ldc = LocalDataCollection(directory)
        LocalDataCollection._init_database = original_init_database

        mock_connection = MagicMock()
        mock_connect.return_value = mock_connection

        mock_cursor_instance = MagicMock()
        mock_connection.cursor.return_value = mock_cursor_instance

        ldc._database_path = directory / "mixtera.sqlite"

        ldc._init_database()

        mock_connect.assert_called_with(ldc._database_path)
        self.assertEqual(mock_cursor_instance.execute.call_count, 3)
        mock_connection.commit.assert_called_once()

    def test_init_database_without_mocked_sqlite(self):
        directory = Path(self.temp_dir.name)
        original_init_database = LocalDataCollection._init_database
        LocalDataCollection._init_database = lambda self: None
        ldc = LocalDataCollection(directory)
        LocalDataCollection._init_database = original_init_database

        ldc._database_path = directory / "mixtera.sqlite"

        ldc._init_database()

        # Check if the database file exists
        self.assertTrue(ldc._database_path.exists())

        # Connect to the database and check if the tables are created
        conn = sqlite3.connect(ldc._database_path)
        cursor = conn.cursor()

        # Check datasets table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='datasets';")
        self.assertIsNotNone(cursor.fetchone())

        # Check files table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='files';")
        self.assertIsNotNone(cursor.fetchone())

        conn.close()

    @patch("sqlite3.connect")
    @patch("mixtera.core.datacollection.local.LocalDataCollection._insert_dataset_into_table")
    @patch("mixtera.core.datacollection.local.LocalDataCollection._insert_file_into_table")
    def test_register_dataset(self, mock_insert_file_into_table, mock_insert_dataset_into_table, mock_connect):
        dataset_id = 42
        mock_connection = MagicMock()
        mock_connect.return_value = mock_connection
        mock_insert_file_into_table.return_value = 0
        mock_insert_dataset_into_table.return_value = dataset_id

        directory = Path(self.temp_dir.name)
        ldc = LocalDataCollection(directory)

        mocked_dtype = MagicMock()
        mocked_dtype.iterate_files = MagicMock()
        mocked_dtype.iterate_files.return_value = [Path("test1.jsonl"), Path("test2.jsonl")]

        self.assertTrue(ldc.register_dataset("test", "loc", mocked_dtype))
        mock_insert_dataset_into_table.assert_called_once_with("test", "loc", mocked_dtype)
        assert mock_insert_file_into_table.call_count == 2
        mock_insert_file_into_table.assert_any_call(dataset_id, Path("test1.jsonl"))
        mock_insert_file_into_table.assert_any_call(dataset_id, Path("test2.jsonl"))

    def test_register_dataset_with_existing_dataset(self):
        directory = Path(self.temp_dir.name)
        ldc = LocalDataCollection(directory)
        (directory / "loc").touch()

        # First time, the dataset registration should succeed.
        self.assertTrue(ldc.register_dataset("test", str(directory / "loc"), JSONLDataset))

        # Second time, the dataset registration should fail (because the dataset already exists).
        self.assertFalse(ldc.register_dataset("test", str(directory / "loc"), JSONLDataset))

    def test_register_dataset_with_non_existent_location(self):
        directory = Path(self.temp_dir.name)
        ldc = LocalDataCollection(directory)

        with self.assertRaises(RuntimeError):
            ldc.register_dataset("test", "/non/existent/location", JSONLDataset)

    @patch("sqlite3.connect")
    @patch("mixtera.core.datacollection.local.LocalDataCollection._insert_dataset_into_table")
    @patch("mixtera.core.datacollection.local.LocalDataCollection._insert_file_into_table")
    def test_register_dataset_updates_index(
        self, mock_insert_file_into_table, mock_insert_dataset_into_table, mock_connect
    ):
        mock_connection = MagicMock()
        dataset_id = 42
        mock_connect.return_value = mock_connection
        mock_insert_file_into_table.side_effect = [0, 1]
        mock_insert_dataset_into_table.return_value = dataset_id

        directory = Path(self.temp_dir.name)
        ldc = LocalDataCollection(directory)

        mocked_dtype = MagicMock()
        mocked_dtype.iterate_files = MagicMock()
        mocked_dtype.iterate_files.return_value = [Path("test1.jsonl"), Path("test2.jsonl")]

        def get_result_index(file, dataset_id, file_id):
            del file
            res = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
            res["language"]["English"][dataset_id][file_id] = [(0, 42)]
            return res

        mocked_dtype.build_file_index = MagicMock()
        mocked_dtype.build_file_index.side_effect = get_result_index

        self.assertDictEqual(defaultdict_to_dict(ldc.get_index()), {})
        self.assertTrue(ldc.register_dataset("test", "loc", mocked_dtype))
        self.assertDictEqual(
            defaultdict_to_dict(ldc.get_index()),
            {
                "language": {
                    "English": {
                        dataset_id: {
                            0: [(0, 42)],
                            1: [(0, 42)],
                        }
                    }
                }
            },
        )

    def test_register_dataset_e2e_json(self):
        directory = Path(self.temp_dir.name)
        ldc = LocalDataCollection(directory)

        jsonl_file_path1 = directory / "temp1.jsonl"
        with open(jsonl_file_path1, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "text": "",
                    "meta": {
                        "content_hash": "4765aae0af2406ea691fb001ea5a83df",
                        "language": [{"name": "Go", "bytes": "734307"}, {"name": "Makefile", "bytes": "183"}],
                    },
                },
                f,
            )
            f.write("\n")
            json.dump(
                {
                    "text": "",
                    "meta": {
                        "content_hash": "324efbc1ad28fdfe902cd1e51f7e095e",
                        "language": [{"name": "Go", "bytes": "366"}, {"name": "CSS", "bytes": "39144"}],
                    },
                },
                f,
            )

        jsonl_file_path2 = directory / "temp2.jsonl"
        with open(jsonl_file_path2, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "text": "",
                    "meta": {
                        "content_hash": "324efbc1ad28fdfe902cd1e51f7e095e",
                        "language": [{"name": "ApacheConf", "bytes": "366"}, {"name": "CSS", "bytes": "39144"}],
                    },
                },
                f,
            )

        ldc.register_dataset("test_dataset", str(directory), JSONLDataset)
        files = ldc._get_all_files()
        file1_id = [file_id for file_id, _, path in files if "temp1.jsonl" in path][0]
        file2_id = [file_id for file_id, _, path in files if "temp2.jsonl" in path][0]

        expected_index = {
            "language": {
                "Go": {1: {file1_id: [(0, 2)]}},
                "Makefile": {1: {file1_id: [(0, 1)]}},
                "ApacheConf": {1: {file2_id: [(0, 1)]}},
                "CSS": {1: {file1_id: [(1, 2)], file2_id: [(0, 1)]}},
            },
        }

        self.assertEqual(defaultdict_to_dict(ldc.get_index()), expected_index)

    def test_insert_dataset_into_table(self):
        directory = Path(self.temp_dir.name)
        ldc = LocalDataCollection(directory)

        # Inserting a new dataset should return 1 (first dataset)
        self.assertEqual(1, ldc._insert_dataset_into_table("test", "loc", JSONLDataset))

        # Inserting an existing dataset should return -1.
        self.assertEqual(-1, ldc._insert_dataset_into_table("test", "loc", JSONLDataset))

    def test_insert_file_into_table(self):
        directory = Path(self.temp_dir.name)
        ldc = LocalDataCollection(directory)

        self.assertTrue(ldc._insert_file_into_table(0, "file_path"))

    def test_check_dataset_exists(self):
        directory = Path(self.temp_dir.name)
        ldc = LocalDataCollection(directory)
        (directory / "loc").touch()

        self.assertFalse(ldc.check_dataset_exists("test"))
        self.assertFalse(ldc.check_dataset_exists("test2"))
        self.assertTrue(ldc.register_dataset("test", str(directory / "loc"), JSONLDataset))
        self.assertTrue(ldc.check_dataset_exists("test"))
        self.assertFalse(ldc.check_dataset_exists("test2"))
        self.assertTrue(ldc.register_dataset("test2", str(directory / "loc"), JSONLDataset))
        self.assertTrue(ldc.check_dataset_exists("test"))
        self.assertTrue(ldc.check_dataset_exists("test2"))

    def test_list_datasets(self):
        directory = Path(self.temp_dir.name)
        ldc = LocalDataCollection(directory)
        (directory / "loc").touch()

        self.assertListEqual([], ldc.list_datasets())
        self.assertTrue(ldc.register_dataset("test", str(directory / "loc"), JSONLDataset))
        self.assertListEqual(["test"], ldc.list_datasets())
        self.assertTrue(ldc.register_dataset("test2", str(directory / "loc"), JSONLDataset))
        self.assertListEqual(["test", "test2"], ldc.list_datasets())

    def test__get_all_files(self):
        directory = Path(self.temp_dir.name)
        ldc = LocalDataCollection(directory)

        temp_dir = directory / "temp_dir"
        temp_dir.mkdir()
        (temp_dir / "temp1.jsonl").touch()
        (temp_dir / "temp2.jsonl").touch()

        self.assertTrue(ldc.register_dataset("test", str(temp_dir), JSONLDataset))

        self.assertListEqual(
            sorted([file_path for _, _, file_path in ldc._get_all_files()]),
            sorted([str(temp_dir / "temp1.jsonl"), str(temp_dir / "temp2.jsonl")]),
        )

    @patch("mixtera.core.datacollection.local.LocalDataCollection._get_all_files")
    @patch("mixtera.core.processing.property_calculation.PropertyCalculationExecutor.from_mode")
    def test_add_property_with_mocks(
        self,
        mock_from_mode,
        mock_get_all_files,
    ):
        directory = Path(self.temp_dir.name)
        ldc = LocalDataCollection(directory)
        (directory / "loc").touch()

        # Set up the mocks
        mock_get_all_files.return_value = [(0, "ds", "file1"), (1, "ds", "file2")]
        mock_executor = mock_from_mode.return_value
        mock_executor.run.return_value = {"bucket": {"ds": {0: [(0, 1)]}}}

        # Call the method
        ldc.add_property(
            "property_name",
            lambda: None,
            lambda: None,
            ExecutionMode.LOCAL,
            PropertyType.CATEGORICAL,
            min_val=0.0,
            max_val=1.0,
            num_buckets=10,
            batch_size=1,
            dop=1,
            data_only_on_primary=True,
        )

        # Check that the mocks were called as expected
        mock_get_all_files.assert_called_once()
        mock_from_mode.assert_called_once_with(
            ExecutionMode.LOCAL,
            1,
            1,
            ANY,
            ANY,
        )
        mock_executor.load_data.assert_called_once_with([(0, "ds", "file1"), (1, "ds", "file2")], True)
        mock_executor.run.assert_called_once()

        self.assertDictEqual(ldc.get_index(property_name="property_name"), {"bucket": {"ds": {0: [(0, 1)]}}})

    def test_add_property_end_to_end(self):
        directory = Path(self.temp_dir.name)
        ldc = LocalDataCollection(directory)

        # Create test dataset
        data = [
            {"meta": {"language": [{"name": "Python"}], "publication_date": "2022"}},
            {"meta": {"language": [{"name": "Java"}], "publication_date": "2021"}},
        ]
        dataset_file = directory / "dataset.jsonl"
        with open(dataset_file, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

        ldc.register_dataset("test_dataset", str(dataset_file), JSONLDataset)
        dataset_id = 1
        # Define setup and calculation functions

        def setup_func(executor):
            executor.prefix = "pref_"

        def calc_func(executor, batch):
            return [executor.prefix + json.loads(sample)["meta"]["publication_date"] for sample in batch["data"]]

        # Add property
        ldc.add_property(
            "test_property",
            setup_func,
            calc_func,
            ExecutionMode.LOCAL,
            PropertyType.CATEGORICAL,
            batch_size=1,
            dop=1,
            data_only_on_primary=True,
        )

        index = defaultdict_to_dict(ldc.get_index())

        self.assertIn("test_property", index)
        self.assertEqual(
            index["test_property"], {"pref_2022": {dataset_id: {1: [(0, 1)]}}, "pref_2021": {dataset_id: {1: [(1, 2)]}}}
        )
