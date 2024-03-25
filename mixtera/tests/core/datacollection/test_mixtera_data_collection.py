import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import ANY, MagicMock, patch

from mixtera.core.datacollection import MixteraDataCollection, PropertyType
from mixtera.core.datacollection.datasets.jsonl_dataset import JSONLDataset
from mixtera.core.datacollection.index.index_collection import IndexFactory, IndexTypes
from mixtera.core.processing import ExecutionMode
from mixtera.utils import defaultdict_to_dict


class TestLocalDataCollection(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with

    def tearDown(self):
        self.temp_dir.cleanup()

    @patch("sqlite3.connect")
    @patch("mixtera.core.datacollection.MixteraDataCollection._init_database")
    def test_init_with_non_existing_database(self, mock_init_database: MagicMock, mock_connect: MagicMock):
        mock_connection = MagicMock()
        mock_init_database.return_value = mock_connection
        mock_connect.return_value = mock_connection

        directory = Path(self.temp_dir.name)
        mdc = MixteraDataCollection(directory)

        mock_init_database.assert_called_once()
        mock_connect.assert_not_called()
        self.assertEqual(mdc._connection, mock_connection)

    @patch("sqlite3.connect")
    @patch("mixtera.core.datacollection.MixteraDataCollection._init_database")
    def test_init_with_existing_database(self, mock_init_database: MagicMock, mock_connect: MagicMock):
        mock_connection = MagicMock()
        mock_connect.return_value = mock_connection

        directory = Path(self.temp_dir.name)
        (directory / "mixtera.sqlite").touch()
        self.assertTrue((directory / "mixtera.sqlite").exists())
        mdc = MixteraDataCollection(directory)

        mock_connect.assert_called_once_with(directory / "mixtera.sqlite")
        mock_init_database.assert_not_called()
        self.assertEqual(mdc._connection, mock_connection)

    @patch("sqlite3.connect")
    def test_init_database_with_mocked_sqlite(self, mock_connect):
        directory = Path(self.temp_dir.name)
        original_init_database = MixteraDataCollection._init_database
        MixteraDataCollection._init_database = lambda self: None
        mdc = MixteraDataCollection(directory)
        MixteraDataCollection._init_database = original_init_database

        mock_connection = MagicMock()
        mock_connect.return_value = mock_connection

        mock_cursor_instance = MagicMock()
        mock_connection.cursor.return_value = mock_cursor_instance

        mdc._database_path = directory / "mixtera.sqlite"

        mdc._init_database()

        mock_connect.assert_called_with(mdc._database_path)
        self.assertEqual(mock_cursor_instance.execute.call_count, 3)
        mock_connection.commit.assert_called_once()

    def test_init_database_without_mocked_sqlite(self):
        directory = Path(self.temp_dir.name)
        original_init_database = MixteraDataCollection._init_database
        MixteraDataCollection._init_database = lambda self: None
        mdc = MixteraDataCollection(directory)
        MixteraDataCollection._init_database = original_init_database

        mdc._database_path = directory / "mixtera.sqlite"

        mdc._init_database()

        # Check if the database file exists
        self.assertTrue(mdc._database_path.exists())

        # Connect to the database and check if the tables are created
        conn = sqlite3.connect(mdc._database_path)
        cursor = conn.cursor()

        # Check datasets table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='datasets';")
        self.assertIsNotNone(cursor.fetchone())

        # Check files table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='files';")
        self.assertIsNotNone(cursor.fetchone())

        conn.close()

    @patch("sqlite3.connect")
    @patch("mixtera.core.datacollection.MixteraDataCollection._insert_dataset_into_table")
    @patch("mixtera.core.datacollection.MixteraDataCollection._insert_file_into_table")
    def test_register_dataset(self, mock_insert_file_into_table, mock_insert_dataset_into_table, mock_connect):
        dataset_id = 42
        mock_connection = MagicMock()
        mock_connect.return_value = mock_connection
        mock_insert_file_into_table.return_value = 0
        mock_insert_dataset_into_table.return_value = dataset_id

        directory = Path(self.temp_dir.name)
        mdc = MixteraDataCollection(directory)

        mocked_dtype = MagicMock()
        mocked_dtype.iterate_files = MagicMock()
        mocked_dtype.iterate_files.return_value = [Path("test1.jsonl"), Path("test2.jsonl")]

        def proc_func(data):
            return f"prefix_{data}"

        self.assertTrue(mdc.register_dataset("test", "loc", mocked_dtype, proc_func, "RED_PAJAMA"))
        mock_insert_dataset_into_table.assert_called_once_with("test", "loc", mocked_dtype, proc_func)
        assert mock_insert_file_into_table.call_count == 2
        mock_insert_file_into_table.assert_any_call(dataset_id, Path("test1.jsonl"))
        mock_insert_file_into_table.assert_any_call(dataset_id, Path("test2.jsonl"))

    def test_register_dataset_with_existing_dataset(self):
        directory = Path(self.temp_dir.name)
        mdc = MixteraDataCollection(directory)
        (directory / "loc").touch()

        # First time, the dataset registration should succeed.
        self.assertTrue(
            mdc.register_dataset(
                "test",
                str(directory / "loc"),
                JSONLDataset,
                lambda data: f"prefix_{data}",
                "RED_PAJAMA",
            )
        )

        # Second time, the dataset registration should fail (because the dataset already exists).
        self.assertFalse(
            mdc.register_dataset(
                "test",
                str(directory / "loc"),
                JSONLDataset,
                lambda data: f"prefix_{data}",
                "RED_PAJAMA",
            )
        )

    def test_register_dataset_with_non_existent_location(self):
        directory = Path(self.temp_dir.name)
        mdc = MixteraDataCollection(directory)

        with self.assertRaises(RuntimeError):
            mdc.register_dataset(
                "test",
                "/non/existent/location",
                JSONLDataset,
                lambda data: f"prefix_{data}",
                "RED_PAJAMA",
            )

    def test_register_dataset_e2e_json(self):
        directory = Path(self.temp_dir.name)
        mdc = MixteraDataCollection(directory)

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

        mdc.register_dataset("test_dataset", str(directory), JSONLDataset, lambda data: f"prefix_{data}", "RED_PAJAMA")
        files = mdc._get_all_files()
        file1_id = [file_id for file_id, _, _, path in files if "temp1.jsonl" in path][0]
        file2_id = [file_id for file_id, _, _, path in files if "temp2.jsonl" in path][0]

        expected_index = {
            "language": {
                "Go": {1: {file1_id: [(0, 2)]}},
                "Makefile": {1: {file1_id: [(0, 1)]}},
                "ApacheConf": {1: {file2_id: [(0, 1)]}},
                "CSS": {1: {file1_id: [(1, 2)], file2_id: [(0, 1)]}},
            },
        }

        self.assertEqual(defaultdict_to_dict(mdc.get_index().get_full_dict_index()), expected_index)

    def test_insert_dataset_into_table(self):
        directory = Path(self.temp_dir.name)
        mdc = MixteraDataCollection(directory)

        # Inserting a new dataset should return 1 (first dataset)
        self.assertEqual(
            1,
            mdc._insert_dataset_into_table("test", "loc", JSONLDataset, lambda data: f"prefix_{data}"),
        )

        # Inserting an existing dataset should return -1.
        self.assertEqual(
            -1,
            mdc._insert_dataset_into_table("test", "loc", JSONLDataset, lambda data: f"prefix_{data}"),
        )

    def test_insert_file_into_table(self):
        directory = Path(self.temp_dir.name)
        mdc = MixteraDataCollection(directory)

        self.assertTrue(mdc._insert_file_into_table(0, "file_path"))

    def test_check_dataset_exists(self):
        directory = Path(self.temp_dir.name)
        mdc = MixteraDataCollection(directory)
        (directory / "loc").touch()

        self.assertFalse(mdc.check_dataset_exists("test"))
        self.assertFalse(mdc.check_dataset_exists("test2"))
        self.assertTrue(
            mdc.register_dataset(
                "test",
                str(directory / "loc"),
                JSONLDataset,
                lambda data: f"prefix_{data}",
                "RED_PAJAMA",
            )
        )
        self.assertTrue(mdc.check_dataset_exists("test"))
        self.assertFalse(mdc.check_dataset_exists("test2"))
        self.assertTrue(
            mdc.register_dataset(
                "test2",
                str(directory / "loc"),
                JSONLDataset,
                lambda data: f"prefix_{data}",
                "RED_PAJAMA",
            )
        )
        self.assertTrue(mdc.check_dataset_exists("test"))
        self.assertTrue(mdc.check_dataset_exists("test2"))

    def test_list_datasets(self):
        directory = Path(self.temp_dir.name)
        mdc = MixteraDataCollection(directory)
        (directory / "loc").touch()

        self.assertListEqual([], mdc.list_datasets())
        self.assertTrue(
            mdc.register_dataset(
                "test",
                str(directory / "loc"),
                JSONLDataset,
                lambda data: f"prefix_{data}",
                "RED_PAJAMA",
            )
        )
        self.assertListEqual(["test"], mdc.list_datasets())
        self.assertTrue(
            mdc.register_dataset(
                "test2",
                str(directory / "loc"),
                JSONLDataset,
                lambda data: f"prefix_{data}",
                "RED_PAJAMA",
            )
        )
        self.assertListEqual(["test", "test2"], mdc.list_datasets())

    def test__get_all_files(self):
        directory = Path(self.temp_dir.name)
        mdc = MixteraDataCollection(directory)

        temp_dir = directory / "temp_dir"
        temp_dir.mkdir()
        (temp_dir / "temp1.jsonl").touch()
        (temp_dir / "temp2.jsonl").touch()

        self.assertTrue(
            mdc.register_dataset("test", str(temp_dir), JSONLDataset, lambda data: f"prefix_{data}", "RED_PAJAMA")
        )

        self.assertListEqual(
            sorted([file_path for _, _, _, file_path in mdc._get_all_files()]),
            sorted([str(temp_dir / "temp1.jsonl"), str(temp_dir / "temp2.jsonl")]),
        )

        self.assertSetEqual(set(dtype for _, _, dtype, _ in mdc._get_all_files()), set([JSONLDataset]))

    def test__get_dataset_func_by_id(self):
        directory = Path(self.temp_dir.name)
        mdc = MixteraDataCollection(directory)

        did = mdc._insert_dataset_into_table(
            "test_dataset", str(directory), JSONLDataset, lambda data: f"prefix_{data}"
        )
        func = mdc._get_dataset_func_by_id(did)

        self.assertEqual(func("abc"), "prefix_abc")

    def test__get_dataset_type_by_id(self):
        directory = Path(self.temp_dir.name)
        mdc = MixteraDataCollection(directory)

        did = mdc._insert_dataset_into_table(
            "test_dataset", str(directory), JSONLDataset, lambda data: f"prefix_{data}"
        )
        dtype = mdc._get_dataset_type_by_id(did)
        self.assertEqual(dtype, JSONLDataset)

    def test__get_file_path_by_id(self):
        directory = Path(self.temp_dir.name)
        mdc = MixteraDataCollection(directory)

        temp_dir = directory / "temp_dir"
        temp_dir.mkdir()
        (temp_dir / "temp1.jsonl").touch()

        self.assertTrue(
            mdc.register_dataset("test", str(temp_dir), JSONLDataset, lambda data: f"prefix_{data}", "RED_PAJAMA")
        )

        self.assertEqual(mdc._get_file_path_by_id(1), str(temp_dir / "temp1.jsonl"))

    @patch("mixtera.core.datacollection.MixteraDataCollection._get_all_files")
    @patch("mixtera.core.processing.property_calculation.PropertyCalculationExecutor.from_mode")
    def test_add_property_with_mocks(
        self,
        mock_from_mode,
        mock_get_all_files,
    ):
        directory = Path(self.temp_dir.name)
        mdc = MixteraDataCollection(directory)
        (directory / "loc").touch()

        # Set up the mocks
        mock_get_all_files.return_value = [(0, "ds", "file1"), (1, "ds", "file2")]
        mock_executor = mock_from_mode.return_value
        mock_executor.run.return_value = {"bucket": {"ds": {0: [(0, 1)]}}}

        # Call the method
        mdc.add_property(
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

        self.assertDictEqual(
            defaultdict_to_dict(mdc.get_index(property_name="property_name").get_full_dict_index()),
            {"property_name": {"bucket": {"ds": {0: [(0, 1)]}}}},
        )

    def test_add_property_end_to_end(self):
        directory = Path(self.temp_dir.name)
        mdc = MixteraDataCollection(directory)

        # Create test dataset
        data = [
            {"meta": {"language": [{"name": "Python"}], "publication_date": "2022"}},
            {"meta": {"language": [{"name": "Java"}], "publication_date": "2021"}},
        ]
        dataset_file = directory / "dataset.jsonl"
        with open(dataset_file, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

        mdc.register_dataset(
            "test_dataset",
            str(dataset_file),
            JSONLDataset,
            lambda data: f"prefix_{data}",
            "RED_PAJAMA",
        )
        dataset_id = 1
        # Define setup and calculation functions

        def setup_func(executor):
            executor.prefix = "pref_"

        def calc_func(executor, batch):
            return [executor.prefix + json.loads(sample)["meta"]["publication_date"] for sample in batch["data"]]

        # Add property
        mdc.add_property(
            "test_property",
            setup_func,
            calc_func,
            ExecutionMode.LOCAL,
            PropertyType.CATEGORICAL,
            batch_size=1,
            dop=1,
            data_only_on_primary=True,
        )

        index = mdc.get_index()

        self.assertTrue(index.has_feature("test_property"))
        self.assertEqual(
            defaultdict_to_dict(index.get_dict_index_by_feature("test_property")),
            {"pref_2022": {dataset_id: {1: [(0, 1)]}}, "pref_2021": {dataset_id: {1: [(1, 2)]}}},
        )

    @patch("sqlite3.connect")
    def test_insert_partial_index_into_table(self, mock_connect):
        mock_connection = MagicMock()
        mock_connect.return_value = mock_connection
        mock_cursor = MagicMock()
        mock_connection.cursor.return_value = mock_cursor
        directory = Path(self.temp_dir.name)
        mdc = MixteraDataCollection(directory)
        index = {"prediction1": {"dataset1": {"file1": [(1, 2)]}}}
        # Test successful insertion
        mock_cursor.lastrowid = 1
        mock_cursor.rowcount = 1
        result = mdc._insert_partial_index_into_table("property1", index)
        self.assertEqual(result, 1)
        # Test sqlite error during insertion
        mock_cursor.execute.side_effect = sqlite3.Error("Test error")
        result = mdc._insert_partial_index_into_table("property1", index)
        self.assertEqual(result, -1)
        # Test failed insertion (no rows affected)
        mock_cursor.execute.side_effect = None
        mock_cursor.rowcount = 0
        result = mdc._insert_partial_index_into_table("property1", index)
        self.assertEqual(result, -1)

    @patch("sqlite3.connect")
    def test_insert_index_into_table(self, mock_connect):
        mock_connection = MagicMock()
        mock_connect.return_value = mock_connection
        mock_cursor = MagicMock()
        mock_connection.cursor.return_value = mock_cursor
        directory = Path(self.temp_dir.name)
        mdc = MixteraDataCollection(directory)

        # 8 rows
        index = IndexFactory.create_index(IndexTypes.IN_MEMORY_DICT_RANGE)
        index._index = {
            "language": {
                "C": {0: {0: [(0, 1), (2, 3), (4, 5), (9, 10)]}},
                "PHP": {0: {0: [(1, 2)]}},
            },
            "publication_date": {"val1": {0: {0: [(0, 1), (2, 6), (9, 11)]}}},
        }

        # Test successful insertion
        mock_cursor.rowcount = 8
        result = mdc._insert_index_into_table(index)
        self.assertEqual(result, 8)

        # Test partial insertion
        mock_cursor.rowcount = 7
        result = mdc._insert_index_into_table(index)
        self.assertEqual(result, 7)

        # Test sqlite error during insertion
        mock_cursor.executemany.side_effect = sqlite3.Error("Test error")
        result = mdc._insert_index_into_table(index)
        self.assertEqual(result, -1)

        # Test hard partial insertion
        mock_cursor.rowcount = 7
        mock_cursor.executemany.side_effect = None
        self.assertRaises(AssertionError, mdc._insert_index_into_table, index, full_or_fail=True)

    def test_reformat_index(self):
        mdc = MixteraDataCollection(Path(self.temp_dir.name))

        # Test with empty list
        raw_indices = []
        result = mdc._reformat_index(raw_indices)
        self.assertEqual(result.get_full_dict_index(), {})

        # Test with single item in list
        raw_indices = [("prop1", "val1", 1, 0, 1, 2)]
        result = mdc._reformat_index(raw_indices)
        self.assertEqual(result.get_full_dict_index(), {"prop1": {"val1": {1: {0: [(1, 2)]}}}})

        # Test with multiple items in list
        raw_indices = [
            ("prop1", "val1", 0, 0, 1, 2),
            ("prop1", "val1", 0, 1, 3, 4),
            ("prop1", "val2", 1, 0, 5, 6),
            ("prop2", "val1", 0, 0, 7, 8),
        ]
        expected = {
            "prop1": {"val1": {0: {0: [(1, 2)], 1: [(3, 4)]}}, "val2": {1: {0: [(5, 6)]}}},
            "prop2": {"val1": {0: {0: [(7, 8)]}}},
        }
        result = mdc._reformat_index(raw_indices)
        self.assertEqual(result.get_full_dict_index(), expected)

    @patch("sqlite3.connect")
    def test_read_index_from_database_with_property_name(self, mock_connect: MagicMock):
        mock_connection = MagicMock()
        mock_connect.return_value = mock_connection
        mock_cursor_instance = MagicMock()
        mock_connection.cursor.return_value = mock_cursor_instance
        mock_cursor_instance.fetchall.return_value = [
            ("property_name", "property_value", "dataset_id", "file_id", 1, 2)
        ]
        directory = Path(self.temp_dir.name)
        mdc = MixteraDataCollection(directory)
        mdc._connection = mock_connection

        result = mdc._read_index_from_database("property_name")
        self.assertEqual(
            result.get_full_dict_index(), {"property_name": {"property_value": {"dataset_id": {"file_id": [(1, 2)]}}}}
        )
        # test sqlite error
        mock_cursor_instance.execute.side_effect = sqlite3.Error("Test error")
        result = mdc._read_index_from_database("property_name")
        self.assertEqual(result.get_full_dict_index(), {})

    @patch("sqlite3.connect")
    def test_read_index_from_database_without_property_name(self, mock_connect: MagicMock):
        mock_connection = MagicMock()
        mock_connect.return_value = mock_connection
        mock_cursor_instance = MagicMock()
        mock_connection.cursor.return_value = mock_cursor_instance
        mock_cursor_instance.fetchall.return_value = [
            ("property_name", "property_value", "dataset_id", "file_id", 1, 2)
        ]

        directory = Path(self.temp_dir.name)
        mdc = MixteraDataCollection(directory)
        mdc._connection = mock_connection

        result = mdc._read_index_from_database()

        self.assertEqual(
            result.get_full_dict_index(), {"property_name": {"property_value": {"dataset_id": {"file_id": [(1, 2)]}}}}
        )

    @patch("mixtera.core.datacollection.MixteraDataCollection._read_index_from_database")
    def test_get_index(self, mock_read_index_from_database: MagicMock):
        target_index = IndexFactory.create_index(IndexTypes.IN_MEMORY_DICT_LINES)
        target_index._index = {"property1": "value1", "property2": "value2"}

        mock_read_index_from_database.return_value = target_index
        mdc = MixteraDataCollection(Path(self.temp_dir.name))
        result = mdc.get_index()
        mock_read_index_from_database.assert_called_once()
        self.assertEqual(result.get_full_dict_index(), {"property1": "value1", "property2": "value2"})
        result = mdc.get_index("property1")
        # here it should still be called once, because the result is already cached
        mock_read_index_from_database.assert_called_once()
        self.assertEqual(result.get_full_dict_index(), {"property1": "value1"})

        # test with non-existing property
        result = mdc.get_index("property3")
        mock_read_index_from_database.assert_called_with("property3")
        self.assertEqual(result, None)
