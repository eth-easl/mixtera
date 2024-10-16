import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import ANY, MagicMock, patch

import duckdb
from mixtera.core.datacollection import MixteraDataCollection
from mixtera.core.datacollection.datasets.jsonl_dataset import JSONLDataset


class TestLocalDataCollection(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with

    def tearDown(self):
        self.temp_dir.cleanup()

    @patch("mixtera.core.datacollection.MixteraDataCollection._load_db_from_disk")
    @patch("mixtera.core.datacollection.MixteraDataCollection._init_database")
    def test_init_with_non_existing_database(self, mock_init_database: MagicMock, mock_load_db_from_disk: MagicMock):
        mock_connection = MagicMock()
        mock_init_database.return_value = mock_connection
        mock_load_db_from_disk.return_value = mock_connection

        directory = Path(self.temp_dir.name)
        mdc = MixteraDataCollection(directory)

        mock_init_database.assert_called_once()
        mock_load_db_from_disk.assert_not_called()
        self.assertEqual(mdc._connection, mock_connection)

    @patch("mixtera.core.datacollection.MixteraDataCollection._load_db_from_disk")
    @patch("mixtera.core.datacollection.MixteraDataCollection._init_database")
    def test_init_with_existing_database(self, mock_init_database: MagicMock, mock_load_db_from_disk: MagicMock):
        mock_connection = MagicMock()
        mock_load_db_from_disk.return_value = mock_connection

        directory = Path(self.temp_dir.name)
        (directory / "mixtera.duckdb").touch()
        self.assertTrue((directory / "mixtera.duckdb").exists())
        mdc = MixteraDataCollection(directory)

        mock_load_db_from_disk.assert_called_once()
        mock_init_database.assert_not_called()
        self.assertEqual(mdc._connection, mock_connection)

    @patch("duckdb.connect")
    @patch.object(MixteraDataCollection, "_vacuum")
    @patch.object(MixteraDataCollection, "_configure_duckdb")
    def test_init_database_with_mocked_duckdb(
        self, mock_configure_duckdb: MagicMock, mock_vacuum: MagicMock, mock_connect: MagicMock
    ):
        del mock_configure_duckdb
        del mock_vacuum
        directory = Path(self.temp_dir.name)

        # Create instance without calling __init__
        mdc = MixteraDataCollection.__new__(MixteraDataCollection)

        # Set the necessary attributes and ensure the database path does not exist
        mdc._directory = directory
        mdc._database_path = directory / "mixtera.duckdb"
        if mdc._database_path.exists():
            mdc._database_path.unlink()

        # Prepare mock connection and cursor
        mock_connection = MagicMock()
        mock_connect.return_value = mock_connection

        mock_cursor_instance = MagicMock()
        mock_connection.cursor.return_value = mock_cursor_instance

        # Now, call _init_database to test it
        mdc._init_database()

        mock_connect.assert_called_with(str(mdc._database_path))

        execute_call_count = mock_cursor_instance.execute.call_count
        self.assertEqual(execute_call_count, 8)
        mock_connection.commit.assert_called_once()

        expected_calls = [
            (("CREATE SEQUENCE seq_dataset_id START 1;",),),
            (
                (
                    "CREATE TABLE IF NOT EXISTS datasets"
                    " (id INTEGER PRIMARY KEY NOT NULL DEFAULT nextval('seq_dataset_id'), name TEXT NOT NULL UNIQUE,"
                    " location TEXT NOT NULL, type INTEGER NOT NULL,"
                    " parsing_func BLOB NOT NULL);",
                ),
            ),
            (("CREATE SEQUENCE seq_file_id START 1;",),),
            (
                (
                    "CREATE TABLE IF NOT EXISTS files"
                    " (id INTEGER PRIMARY KEY NOT NULL DEFAULT nextval('seq_file_id'),"
                    " dataset_id INTEGER NOT NULL,"
                    " location TEXT NOT NULL,"
                    " FOREIGN KEY(dataset_id) REFERENCES datasets(id));",
                ),
            ),
            (("CREATE SEQUENCE seq_sample_id START 1;",),),
            (
                (
                    "CREATE TABLE IF NOT EXISTS samples"
                    " (dataset_id INTEGER NOT NULL, file_id INTEGER NOT NULL,"
                    " sample_id INTEGER NOT NULL DEFAULT nextval('seq_sample_id'),"
                    " PRIMARY KEY (dataset_id, file_id, sample_id));",
                ),
            ),
            (("CREATE TABLE IF NOT EXISTS version (id INTEGER PRIMARY KEY, version_number INTEGER)",),),
            (("INSERT INTO version (id, version_number) VALUES (1, 1)",),),
        ]
        actual_calls = mock_cursor_instance.execute.call_args_list

        for expected_call in expected_calls:
            self.assertIn(expected_call, actual_calls)

        # Ensure that all expected SQL commands were called
        self.assertEqual(len(actual_calls), len(expected_calls))

    def test_init_database_without_mocked_duckdb(self):
        directory = Path(self.temp_dir.name)

        # Create instance without calling __init__
        mdc = MixteraDataCollection.__new__(MixteraDataCollection)

        # Manually set the required attributes
        mdc._directory = directory
        mdc._database_path = directory / "mixtera.duckdb"
        mdc._connection = None  # It will be set in _init_database

        # Ensure the database path does not exist
        if mdc._database_path.exists():
            mdc._database_path.unlink()

        # Now, call _init_database to test it
        mdc._init_database()

        # Check if the database file exists
        self.assertTrue(mdc._database_path.exists())

        # Connect to the database and check if the tables are created
        conn = duckdb.connect(str(mdc._database_path))
        cursor = conn.cursor()

        # Check datasets table
        cursor.execute("SELECT * FROM information_schema.tables WHERE table_name='datasets';")
        self.assertIsNotNone(cursor.fetchone())

        # Check files table
        cursor.execute("SELECT * FROM information_schema.tables WHERE table_name='files';")
        self.assertIsNotNone(cursor.fetchone())

        # Check samples table
        cursor.execute("SELECT * FROM information_schema.tables WHERE table_name='samples';")
        self.assertIsNotNone(cursor.fetchone())

        conn.close()

    @patch("mixtera.core.datacollection.mixtera_data_collection.MixteraDataCollection._insert_samples_with_metadata")
    @patch("mixtera.core.datacollection.mixtera_data_collection.MixteraDataCollection._insert_file_into_table")
    @patch("mixtera.core.datacollection.mixtera_data_collection.MixteraDataCollection._insert_dataset_into_table")
    @patch.object(MixteraDataCollection, "_configure_duckdb")
    def test_register_dataset(
        self,
        mock_configure_duckdb,
        mock_insert_dataset_into_table,
        mock_insert_file_into_table,
        mock_insert_samples_with_metadata,
    ):
        del mock_configure_duckdb
        dataset_id = 42
        mock_insert_dataset_into_table.return_value = dataset_id
        mock_insert_file_into_table.return_value = 0  # File ID

        # Create instance without calling __init__
        directory = Path(self.temp_dir.name)
        mdc = MixteraDataCollection.__new__(MixteraDataCollection)

        # Set required attributes
        mdc._directory = directory
        mdc._database_path = directory / "mixtera.duckdb"
        mdc._connection = MagicMock()
        mdc._metadata_factory = MagicMock()

        mocked_dtype = MagicMock()
        mocked_dtype.iterate_files.return_value = [Path("test1.jsonl"), Path("test2.jsonl")]
        mocked_dtype.inform_metadata_parser = MagicMock()

        def proc_func(data):
            return f"prefix_{data}"

        # Run the test
        self.assertTrue(mdc.register_dataset("test", "loc", mocked_dtype, proc_func, "RED_PAJAMA"))

        # Assertions
        mock_insert_dataset_into_table.assert_called_once_with("test", "loc", mocked_dtype, proc_func)
        self.assertEqual(mock_insert_file_into_table.call_count, 2)
        mock_insert_file_into_table.assert_any_call(dataset_id, Path("test1.jsonl"))
        mock_insert_file_into_table.assert_any_call(dataset_id, Path("test2.jsonl"))
        self.assertEqual(mock_insert_samples_with_metadata.call_count, 2)
        mocked_dtype.inform_metadata_parser.assert_any_call(Path("test1.jsonl"), ANY)
        mocked_dtype.inform_metadata_parser.assert_any_call(Path("test2.jsonl"), ANY)
        self.assertEqual(mocked_dtype.inform_metadata_parser.call_count, 2)

    def test_register_dataset_with_existing_dataset(self):
        directory = Path(self.temp_dir.name)
        mdc = MixteraDataCollection(directory)
        (directory / "loc").mkdir(exist_ok=True)

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
                    "sample_id": 0,
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
                    "sample_id": 1,
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
                    "sample_id": 0,
                    "meta": {
                        "content_hash": "324efbc1ad28fdfe902cd1e51f7e095e",
                        "language": [{"name": "ApacheConf", "bytes": "366"}, {"name": "CSS", "bytes": "39144"}],
                    },
                },
                f,
            )

        mdc.register_dataset("test_dataset", str(directory), JSONLDataset, lambda data: f"prefix_{data}", "RED_PAJAMA")

        # Now, query the samples table to check if the data was inserted correctly
        conn = mdc._connection

        # Fetch dataset_id
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM datasets WHERE name = ?", ("test_dataset",))
        result = cursor.fetchone()
        self.assertIsNotNone(result)
        dataset_id = result[0]

        # Fetch file_ids
        cursor.execute("SELECT id, location FROM files WHERE dataset_id = ?", (dataset_id,))
        file_records = cursor.fetchall()
        self.assertEqual(len(file_records), 2)  # We have two files

        # Create a mapping from file paths to file_ids
        file_id_map = {record[1]: record[0] for record in file_records}

        # Fetch samples
        query = "SELECT dataset_id, file_id, sample_id, language FROM samples WHERE dataset_id = ?;"
        cursor.execute(query, (dataset_id,))
        samples = cursor.fetchall()

        # Build expected samples data
        expected_samples = [
            {
                "dataset_id": dataset_id,
                "file_id": file_id_map[str(jsonl_file_path1)],
                "sample_id": 0,
                "language": ["Go", "Makefile"],
            },
            {
                "dataset_id": dataset_id,
                "file_id": file_id_map[str(jsonl_file_path1)],
                "sample_id": 1,
                "language": ["Go", "CSS"],
            },
            {
                "dataset_id": dataset_id,
                "file_id": file_id_map[str(jsonl_file_path2)],
                "sample_id": 0,
                "language": ["ApacheConf", "CSS"],
            },
        ]

        # Convert fetched samples to list of dictionaries
        samples_list = []
        for row in samples:
            sample_dict = {
                "dataset_id": row[0],
                "file_id": row[1],
                "sample_id": row[2],
                "language": row[3],  # This should be a list
            }
            samples_list.append(sample_dict)

        # Sort both lists for comparison
        def sort_key(s):
            return (s["dataset_id"], s["file_id"], s["sample_id"])

        expected_samples_sorted = sorted(expected_samples, key=sort_key)
        samples_list_sorted = sorted(samples_list, key=sort_key)

        # Check that the number of samples matches
        self.assertEqual(len(samples_list_sorted), len(expected_samples_sorted))

        # Compare each sample
        for expected_sample, actual_sample in zip(expected_samples_sorted, samples_list_sorted):
            self.assertEqual(expected_sample["dataset_id"], actual_sample["dataset_id"])
            self.assertEqual(expected_sample["file_id"], actual_sample["file_id"])
            self.assertEqual(expected_sample["sample_id"], actual_sample["sample_id"])
            self.assertEqual(
                set(expected_sample["language"]),
                set(actual_sample["language"]),
                f"Languages do not match for sample_id {expected_sample['sample_id']}",
            )

    def test_insert_dataset_into_table(self):
        directory = Path(self.temp_dir.name)
        mdc = MixteraDataCollection.__new__(MixteraDataCollection)
        mdc._directory = directory
        mdc._database_path = directory / "mixtera.duckdb"
        # Ensure the database file does not exist
        if mdc._database_path.exists():
            mdc._database_path.unlink()
        # Initialize the database and set the connection
        mdc._connection = mdc._init_database()
        # Now, we can proceed to test _insert_dataset_into_table
        dataset_id = mdc._insert_dataset_into_table("test", "loc", JSONLDataset, lambda data: f"prefix_{data}")
        self.assertEqual(dataset_id, 1)
        # Inserting the same dataset again should return -1
        dataset_id = mdc._insert_dataset_into_table("test", "loc", JSONLDataset, lambda data: f"prefix_{data}")
        self.assertEqual(dataset_id, -1)
        mdc._connection.close()

    def test_insert_file_into_table(self):
        directory = Path(self.temp_dir.name)
        mdc = MixteraDataCollection.__new__(MixteraDataCollection)
        mdc._directory = directory
        mdc._database_path = directory / "mixtera.duckdb"
        # Ensure the database file does not exist
        if mdc._database_path.exists():
            mdc._database_path.unlink()
        # Initialize the database and set the connection
        mdc._connection = mdc._init_database()
        # First, we need a dataset to associate the file with
        dataset_id = mdc._insert_dataset_into_table("test_ds", "loc", JSONLDataset, lambda data: f"prefix_{data}")
        self.assertTrue(dataset_id >= 1)
        # Now insert a file into the table
        file_id = mdc._insert_file_into_table(dataset_id, "file_path")
        self.assertTrue(file_id >= 1)
        mdc._connection.close()

    def test_check_dataset_exists(self):
        directory = Path(self.temp_dir.name)
        mdc = MixteraDataCollection(directory)
        (directory / "loc").mkdir(exist_ok=True)

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
        (directory / "loc").mkdir(exist_ok=True)

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

        # Get file ID
        files = mdc._get_all_files()
        file_id = files[0][0]  # Assuming only one file, get the first one

        self.assertEqual(mdc._get_file_path_by_id(file_id), str(temp_dir / "temp1.jsonl"))

    @patch("mixtera.core.datacollection.mixtera_data_collection.MixteraDataCollection._insert_property_values")
    @patch("mixtera.core.datacollection.mixtera_data_collection.MixteraDataCollection._add_columns_to_samples_table")
    @patch("mixtera.core.datacollection.mixtera_data_collection.MixteraDataCollection._get_all_files")
    @patch("mixtera.core.processing.property_calculation.executor.PropertyCalculationExecutor.from_mode")
    @patch.object(MixteraDataCollection, "_configure_duckdb")
    def test_add_property_with_mocks(
        self,
        mock_configure_duckdb,
        mock_from_mode,
        mock_get_all_files,
        mock_add_columns,
        mock_insert_property_values,
    ):
        # TODO(create issue): Due to an issue in DuckDB, adding properties does currently not work.
        """
            directory = Path(self.temp_dir.name)
            mdc = MixteraDataCollection.__new__(MixteraDataCollection)
            mdc._directory = directory
            mdc._database_path = directory / "mixtera.duckdb"
            mdc._connection = MagicMock()
            mdc._metadata_factory = MagicMock()

            # Set up the mocks
            mock_get_all_files.return_value = [
                (0, 0, JSONLDataset, "file1.jsonl"),
                (1, 0, JSONLDataset, "file2.jsonl")
            ]
            mock_executor = mock_from_mode.return_value
            mock_executor.run.return_value = [
                {"dataset_id": 0, "file_id": 0, "sample_id": 0, "property_value": "value1"},
                {"dataset_id": 0, "file_id": 1, "sample_id": 1, "property_value": "value2"},
            ]

            # Call the method
            mdc.add_property(
                "property_name",
                setup_func=lambda executor: None,
                calc_func=lambda executor, batch: ["value1", "value2"],
                execution_mode=ExecutionMode.LOCAL,
                property_type=PropertyType.CATEGORICAL,
                batch_size=1,
                degree_of_parallelism=1,
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
            mock_executor.load_data.assert_called_once_with(mock_get_all_files.return_value, True)
            mock_executor.run.assert_called_once()
            mock_add_columns.assert_called_once_with({"property_name"})
            mock_insert_property_values.assert_called_once_with("property_name", mock_executor.run.return_value)

        def test_add_property_end_to_end(self):
            directory = Path(self.temp_dir.name)
            mdc = MixteraDataCollection(directory)

            # Create test dataset
            data = [
                {"sample_id": 0, "meta": {"publication_date": "2022"}},
                {"sample_id": 1, "meta": {"publication_date": "2021"}},
            ]
            dataset_file = directory / "dataset.jsonl"
            with open(dataset_file, "w", encoding="utf-8") as f:
                for item in data:
                    f.write(json.dumps(item) + "\n")

            mdc.register_dataset(
                "test_dataset",
                str(directory),
                JSONLDataset,
                lambda data: f"prefix_{data}",
                "RED_PAJAMA",
            )

            # Define setup and calculation functions
            def setup_func(executor):
                executor.prefix = "pref_"

            def calc_func(executor, batch):
                predictions = []
                for sample in batch["data"]:
                    prediction = executor.prefix + json.loads(sample)["meta"]["publication_date"]
                    predictions.append(prediction)
                return predictions

            # Add property
            mdc.add_property(
                "test_property",
                setup_func,
                calc_func,
                ExecutionMode.LOCAL,
                PropertyType.CATEGORICAL,
                batch_size=2,
                degree_of_parallelism=1,
                data_only_on_primary=True,
            )

            # Query the samples table to check if the property was added correctly
            conn = mdc._connection
            df = conn.execute(
                "SELECT sample_id, test_property FROM samples WHERE dataset_id = 1 ORDER BY sample_id"
            ).fetchdf()

            expected_df = pl.DataFrame(
                {
                    "sample_id": [0, 1],
                    "test_property": [["pref_2022"], ["pref_2021"]],
                }
            )

            # Convert DuckDB dataframe to Polars dataframe for comparison
            actual_df = pl.from_pandas(df)

            self.assertTrue(expected_df.frame_equal(actual_df, null_equal=True))
        """
