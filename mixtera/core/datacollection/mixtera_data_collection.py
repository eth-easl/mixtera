from pathlib import Path
from typing import Callable, List, Type

import dill
import duckdb
from loguru import logger
from mixtera.core.datacollection.datasets import Dataset
from mixtera.core.datacollection.index.parser import MetadataParserFactory
from mixtera.core.datacollection.property import Property
from mixtera.core.datacollection.property_type import PropertyType
from mixtera.core.processing import ExecutionMode
from mixtera.core.processing.property_calculation.executor import PropertyCalculationExecutor


class MixteraDataCollection:
    def __init__(self, directory: Path) -> None:
        if not directory.exists():
            raise RuntimeError(f"Directory {directory} does not exist.")

        self._directory = directory
        self._database_path = self._directory / "mixtera.duckdb"

        self._properties: list[Property] = []
        self._datasets: list[Dataset] = []

        self._metadata_factory = MetadataParserFactory()

        if not self._database_path.exists():
            self._connection = self._init_database()
        else:
            self._connection = self._load_db_from_disk()

    def _load_db_from_disk(self) -> duckdb.DuckDBPyConnection:
        assert self._database_path.exists()
        logger.info(f"Loading database from {self._database_path}")
        conn = duckdb.connect(str(self._database_path))
        logger.info("Database loaded.")
        return conn

    def _init_database(self) -> duckdb.DuckDBPyConnection:
        assert hasattr(self, "_database_path")
        assert not self._database_path.exists()
        logger.info("Initializing database.")
        conn = duckdb.connect(str(self._database_path))
        cur = conn.cursor()
        cur.execute("CREATE SEQUENCE seq_dataset_id START 1;")
        cur.execute(
            "CREATE TABLE IF NOT EXISTS datasets"
            " (id INTEGER PRIMARY KEY NOT NULL DEFAULT nextval('seq_dataset_id'), name TEXT NOT NULL UNIQUE,"
            " location TEXT NOT NULL, type INTEGER NOT NULL,"
            " parsing_func BLOB NOT NULL);"
        )
        cur.execute("CREATE SEQUENCE seq_file_id START 1;")
        cur.execute(
            "CREATE TABLE IF NOT EXISTS files"
            " (id INTEGER PRIMARY KEY NOT NULL DEFAULT nextval('seq_file_id'),"
            " dataset_id INTEGER NOT NULL,"
            " location TEXT NOT NULL,"
            " FOREIGN KEY(dataset_id) REFERENCES datasets(id));"
        )

        cur.execute("CREATE SEQUENCE seq_sample_id START 1;")
        # We don't use foreign key constraints here for insert performance reasons if we have a lot of samples
        cur.execute(
            "CREATE TABLE IF NOT EXISTS samples"
            " (dataset_id INTEGER NOT NULL, file_id INTEGER NOT NULL,"
            " sample_id INTEGER NOT NULL DEFAULT nextval('seq_sample_id'),"
            " PRIMARY KEY (dataset_id, file_id, sample_id));"
        )

        cur.execute("CREATE TABLE IF NOT EXISTS version (id INTEGER PRIMARY KEY, version_number INTEGER)")
        cur.execute("INSERT INTO version (id, version_number) VALUES (1, 1)")
        conn.commit()
        logger.info("Database initialized.")
        return conn

    def get_db_version(self) -> int:
        assert self._connection, "Not connected to db!"
        cur = self._connection.cursor()
        cur.execute("SELECT version_number FROM version WHERE id = 1")
        version = cur.fetchone()
        assert version, "Could not fetch version from DB!"
        return version[0]

    def _db_incr_version(self) -> None:
        assert self._connection, "Not connected to db!"
        current_version = self.get_db_version()
        cur = self._connection.cursor()
        new_version = current_version + 1
        cur.execute("UPDATE version SET version_number = ? WHERE id = 1", (new_version,))
        self._connection.commit()

    def register_dataset(
        self,
        identifier: str,
        loc: str,
        dtype: Type[Dataset],
        parsing_func: Callable[[str], str],
        metadata_parser_type: str,
    ) -> bool:
        if (dataset_id := self._insert_dataset_into_table(identifier, loc, dtype, parsing_func)) == -1:
            return False

        file: Path
        for file in dtype.iterate_files(loc):
            if (file_id := self._insert_file_into_table(dataset_id, file)) == -1:
                logger.error(f"Error while inserting file {file}")
                return False
            metadata_parser = self._metadata_factory.create_metadata_parser(metadata_parser_type, dataset_id, file_id)
            dtype.inform_metadata_parser(file, metadata_parser)
            self._insert_samples_with_metadata(dataset_id, file_id, metadata_parser.metadata)

        self._db_incr_version()
        return True

    def _insert_dataset_into_table(
        self,
        identifier: str,
        loc: str,
        dtype: Type[Dataset],
        parsing_func: Callable[[str], str],
    ) -> int:
        valid_types = True
        if not issubclass(dtype, Dataset):
            logger.error(f"Invalid dataset type: {dtype}")
            valid_types = False

        type_id = dtype.type.value
        if type_id == 0:
            logger.error("Cannot use generic Dataset class as dtype.")
            valid_types = False

        if not valid_types:
            return -1

        serialized_parsing_func = dill.dumps(parsing_func)

        try:
            query = "INSERT INTO datasets (name, location, type, parsing_func) VALUES (?, ?, ?, ?) RETURNING id;"
            cur = self._connection.cursor()
            result = cur.execute(query, (identifier, loc, type_id, serialized_parsing_func)).fetchone()
            self._connection.commit()
            inserted_id = result[0]
            self._db_incr_version()
        except duckdb.Error as err:
            logger.error(f"A DuckDB error occurred during insertion: {err}")
            return -1

        if inserted_id:
            logger.info(f"Successfully registered dataset {identifier} with id {inserted_id}.")
            return inserted_id

        logger.error(f"Failed to register dataset {identifier}.")
        return -1

    def _insert_file_into_table(self, dataset_id: int, loc: Path) -> int:
        query = "INSERT INTO files (dataset_id, location) VALUES (?, ?) RETURNING id;"
        cur = self._connection.cursor()
        logger.info(f"Inserting file at {loc} for dataset id = {dataset_id}")
        result = cur.execute(query, (dataset_id, str(loc))).fetchone()
        self._connection.commit()
        self._db_incr_version()

        if result:
            return result[0]

        logger.error(f"Failed to register file {loc}.")
        return -1

    def _add_columns_to_samples_table(self, columns: list[str]):
        cur = self._connection.cursor()
        for column in columns:
            # Check if the column already exists
            cur.execute(f"SELECT 1 FROM pragma_table_info('samples') WHERE name='{column}';")
            if not cur.fetchone():
                # If the column doesn't exist, add it
                # We're using VARCHAR as a default type, but we might want to adjust this
                # [] indicates a duckdb list
                cur.execute(f"ALTER TABLE samples ADD COLUMN {column} VARCHAR[];")
        self._connection.commit()

    def _insert_samples_with_metadata(self, dataset_id: int, file_id: int, metadata: list[dict]):
        if not metadata:
            logger.warning(f"No metadata extracted from file {file_id} in dataset {dataset_id}")
            return

        assert "sample_id" in metadata[0].keys(), "Your metadata parser needs to provide the sample_id"

        # Get the metadata keys (excluding 'sample_id' which is already in the table)
        metadata_keys = [key for key in metadata[0].keys() if key != "sample_id"]

        # Add new columns to the samples table if needed
        self._add_columns_to_samples_table(metadata_keys)

        # Prepare the INSERT query
        columns = ["dataset_id", "file_id", "sample_id"] + metadata_keys
        placeholders = ", ".join(["?"] * len(columns))
        query = f"INSERT INTO samples ({', '.join(columns)}) VALUES ({placeholders})"

        # Prepare the values for batch insert
        values = [
            [dataset_id, file_id, sample["sample_id"]] + [
                sample.get(key) if isinstance(sample.get(key), list) else [sample.get(key)] for key in metadata_keys
            ]
            for sample in metadata
        ]

        # Execute the batch insert
        cur = self._connection.cursor()
        cur.executemany(query, values)
        self._connection.commit()


    def check_dataset_exists(self, identifier: str) -> bool:
        try:
            query = "SELECT COUNT(*) from datasets WHERE name = ?;"
            cur = self._connection.cursor()
            result = cur.execute(query, (identifier,)).fetchone()
        except duckdb.Error as err:
            logger.error(f"A DuckDB error occurred during selection: {err}")
            return False

        assert result[0] <= 1
        return result[0] == 1

    def list_datasets(self) -> List[str]:
        try:
            query = "SELECT name from datasets;"
            cur = self._connection.cursor()
            result = cur.execute(query).fetchall()
        except duckdb.Error as err:
            logger.error(f"A DuckDB error occurred during selection: {err}")
            return []

        return [dataset[0] for dataset in result]

    def remove_dataset(self, identifier: str) -> bool:
        if not self.check_dataset_exists(identifier):
            logger.error(f"Dataset {identifier} does not exist.")
            return False

        try:
            delete_samples_query = """
            DELETE FROM samples
            WHERE dataset_id IN (
                SELECT id FROM datasets WHERE name = ?
            );
            """
            cur = self._connection.cursor()
            cur.execute(delete_samples_query, (identifier,))

            delete_dataset_query = "DELETE FROM datasets WHERE name = ?;"
            cur.execute(delete_dataset_query, (identifier,))

            self._connection.commit()
            self._db_incr_version()
        except duckdb.Error as err:
            logger.error(f"A DuckDB error occurred during deletion: {err}")
            return False

        return True

    def _get_all_files(self) -> list[tuple[int, int, Type[Dataset], str]]:
        try:
            query = (
                "SELECT files.id, files.dataset_id, files.location, datasets.type"
                + " from files JOIN datasets ON files.dataset_id = datasets.id;"
            )
            cur = self._connection.cursor()
            result = cur.execute(query).fetchall()
        except duckdb.Error as err:
            logger.error(f"A DuckDB error occurred during selection: {err}")
            return []

        return [(fid, did, Dataset.from_type_id(dtype), loc) for fid, did, loc, dtype in result]

    def _get_dataset_func_by_id(self, did: int) -> Callable[[str], str]:
        try:
            query = "SELECT parsing_func from datasets WHERE id = ?;"
            cur = self._connection.cursor()
            result = cur.execute(query, (did,)).fetchone()
        except duckdb.Error as err:
            logger.error(f"Error while selecting parsing_func for did {did}")
            raise RuntimeError(f"A DuckDB error occurred during selection: {err}") from err

        if result is None:
            raise RuntimeError(f"Could not get dataset parsing func by id for did {did}")

        return dill.loads(result[0])

    def _get_dataset_type_by_id(self, did: int) -> Type[Dataset]:
        try:
            query = "SELECT type from datasets WHERE id = ?;"
            cur = self._connection.cursor()
            result = cur.execute(query, (did,)).fetchone()
        except duckdb.Error as err:
            logger.error(f"Error while selecting parsing_func for did {did}")
            raise RuntimeError(f"A DuckDB error occured during selection: {err}") from err

        if result is None:
            raise RuntimeError(f"Could not get dataset type by id for did {did}")

        result = result[0]

        if not isinstance(result, int):
            raise RuntimeError(f"Dataset type {result} for dataset {did} is not an int")

        return Dataset.from_type_id(result)

    def _get_file_path_by_id(self, fid: int) -> str:
        try:
            query = "SELECT location from files WHERE id = ?;"
            cur = self._connection.cursor()
            result = cur.execute(query, (fid,)).fetchone()
        except duckdb.Error as err:
            logger.error(f"Error while selecting location for fid {fid}")
            raise RuntimeError(f"A DuckDB error occurred during selection: {err}") from err

        if result is None:
            raise RuntimeError(f"Could not get file path by id for file id {fid}")

        return result[0]

    def add_property(
        self,
        property_name: str,
        setup_func: Callable,
        calc_func: Callable,
        execution_mode: ExecutionMode,
        property_type: PropertyType,
        min_val: float = 0.0,
        max_val: float = 1,
        num_buckets: int = 10,
        batch_size: int = 1,
        degree_of_parallelism: int = 1,
        data_only_on_primary: bool = True,
    ) -> bool:
        if len(property_name) <= 0:
            logger.error("Property name must be non-empty.")
            return False

        if property_type == PropertyType.NUMERICAL and max_val <= min_val:
            logger.error(f"max_val (= {max_val}) <= min_val (= {min_val})")
            return False

        if num_buckets < 2:
            logger.error(f"num_buckets = {num_buckets} < 2")
            return False

        if batch_size < 1:
            logger.error(f"batch_size = {batch_size} < 1")
            return False

        if property_type == PropertyType.CATEGORICAL and (min_val != 0.0 or max_val != 1.0 or num_buckets != 10):
            logger.warning(
                "For categorical properties, min_val/max_val/num_buckets do not have meaning,"
                " but deviate from their default value. Please ensure correct parameters."
            )

        if property_type == PropertyType.NUMERICAL:
            logger.error("Numerical properties are not yet implemented.")
            return False

        files = self._get_all_files()
        logger.info(f"Adding property {property_name} for {len(files)} files.")

        executor = PropertyCalculationExecutor.from_mode(
            execution_mode, degree_of_parallelism, batch_size, setup_func, calc_func
        )
        executor.load_data(files, data_only_on_primary)
        new_properties = executor.run()

        # TODO(MaxiBoether): clean this up with register dataset codepath
        self._add_property_column(property_name, property_type)
        self._insert_property_values(property_name, new_properties)
        self._db_incr_version()
        return True

    def _add_property_column(self, property_name: str, property_type: PropertyType):
        data_type = "VARCHAR" if property_type == PropertyType.CATEGORICAL else "DOUBLE"
        query = f"ALTER TABLE samples ADD COLUMN {property_name} {data_type};"
        cur = self._connection.cursor()
        cur.execute(query)
        self._connection.commit()

    def _insert_property_values(self, property_name: str, new_properties: dict):
        query = f"UPDATE samples SET {property_name} = ? WHERE dataset_id = ? AND file_id = ? AND sample_id = ?;"
        cur = self._connection.cursor()

        for dataset_id, file_data in new_properties.items():
            for file_id, sample_data in file_data.items():
                for sample_id, value in sample_data.items():
                    cur.execute(query, (value, dataset_id, file_id, sample_id))

        self._connection.commit()

    def __getstate__(self) -> dict:
        # We cannot pickle the DuckDB connection.
        d = dict(self.__dict__)
        del d["_connection"]
        return d

    def __setstate__(self, state: dict):
        self.__dict__.update(state)
        self._connection = self._load_db_from_disk()
