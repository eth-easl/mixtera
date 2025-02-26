import os
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Callable, List, Type

import dill
import duckdb
import polars as pl
import psutil
import pyarrow as pa
from loguru import logger
from mixtera.core.datacollection.datasets import Dataset
from mixtera.core.datacollection.index.parser import MetadataParserFactory
from mixtera.core.datacollection.index.parser.metadata_parser import MetadataProperty
from mixtera.core.datacollection.property import Property
from mixtera.core.datacollection.property_type import PropertyType
from mixtera.core.processing import ExecutionMode
from mixtera.core.processing.property_calculation.executor import PropertyCalculationExecutor
from mixtera.utils.utils import DummyPool, numpy_to_native


def process_file_for_metadata(
    task: tuple[int, int, str, MetadataParserFactory, str, type[Dataset]],
) -> tuple[int, list[dict]]:
    # This function is outside of the class in order to be pickable
    dataset_id, file_id, file_path_str, metadata_factory, metadata_parser_type, dtype_class = task
    file = Path(file_path_str)
    metadata_parser = metadata_factory.create_metadata_parser(metadata_parser_type, dataset_id, file_id)
    dtype_class.inform_metadata_parser(file, metadata_parser)
    return (file_id, metadata_parser.metadata)


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

        self._configure_duckdb()
        self._vacuum()

    def _configure_duckdb(self) -> None:
        # TODO(#118): Make number of cores and memory configurable
        assert self._connection is not None, "Cannot configure DuckDB as connection is None"

        # Set cores
        num_cores = os.cpu_count() or 1
        num_duckdb_threads = max(num_cores - 4, 1)
        self._connection.execute(f"SET threads TO {num_duckdb_threads}")

        # Set DRAM
        total_memory_bytes = psutil.virtual_memory().total
        # We allow duckdb to use 2/3 of the available DRAM
        duckdb_mem_gb = round((total_memory_bytes * 0.66) / (1024**3))
        self._connection.execute(f"SET memory_limit = '{duckdb_mem_gb}GB'")

        # Set tmpdir (to use fast SSD, potentially)
        duckdb_tmp_dir = self._directory / "duckdbtmp"
        duckdb_tmp_dir.mkdir(exist_ok=True)
        self._connection.execute(f"PRAGMA temp_directory = '{duckdb_tmp_dir}'")

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

        # Dataset table
        cur.execute("CREATE SEQUENCE seq_dataset_id START 1;")
        cur.execute(
            "CREATE TABLE IF NOT EXISTS datasets"
            " (id INTEGER PRIMARY KEY NOT NULL DEFAULT nextval('seq_dataset_id'), name TEXT NOT NULL UNIQUE,"
            " location TEXT NOT NULL, type INTEGER NOT NULL,"
            " parsing_func BLOB NOT NULL);"
        )

        # File table
        cur.execute("CREATE SEQUENCE seq_file_id START 1;")
        cur.execute(
            "CREATE TABLE IF NOT EXISTS files"
            " (id INTEGER PRIMARY KEY NOT NULL DEFAULT nextval('seq_file_id'),"
            " dataset_id INTEGER NOT NULL,"
            " location TEXT NOT NULL,"
            " FOREIGN KEY(dataset_id) REFERENCES datasets(id));"
        )

        # Sample table
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

    def _vacuum(self) -> None:
        logger.info("Vacuuming the DuckDB.")
        self._connection.execute("VACUUM")
        logger.info("Vacuumd.")

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

        files = list(dtype.iterate_files(loc))
        if not files:
            logger.warning(f"No files found in {loc} for dataset {identifier}")
            return False

        logger.info(f"Gathered {len(files)} files, ready to insert")

        parser_class = self._metadata_factory._registry[metadata_parser_type]
        properties = parser_class.get_properties()
        self._add_columns_to_samples_table(properties)
        logger.info("Columns added to samples table based on parser schema.")

        # Insert files into the files table and get file IDs
        file_ids = self._insert_files_into_table(dataset_id, files)
        if not file_ids or len(file_ids) != len(files):
            logger.error(f"Error while inserting files for dataset {identifier}")
            return False
        tasks = [
            (dataset_id, file_id, str(file), self._metadata_factory, metadata_parser_type, dtype)
            for file_id, file in zip(file_ids, files)
        ]

        # Determine the number of worker processes
        num_cores = os.cpu_count() or 1
        num_workers = max(num_cores - 4, 1)
        logger.info("Prepared tasks for reading")

        # We should make this configurable at some point, but it is not on the hot path of query execution...
        chunk_size = 2000

        # If we used mocking in unit tests, they get lost when we use a mp.Pool
        # Hence, we need to disable multiprocessing for tests here
        pool_c = DummyPool if os.environ.get("PYTEST_CURRENT_TEST") else Pool

        with pool_c(num_workers) as pool:
            for i in range(0, len(tasks), chunk_size):
                chunk = tasks[i : i + chunk_size]
                results = pool.map(process_file_for_metadata, chunk)
                logger.info(f"Processed chunk {i // chunk_size + 1}, inserting samples.")
                # Insert collected metadata into the database in the main process
                self._insert_samples_with_metadata(dataset_id, results, properties)
        logger.info("All tasks finished.")

        self._db_incr_version()
        self._vacuum()

        logger.info("Finished dataset registration.")
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

            if result is None:
                logger.error("result is None without any DuckDB error. This should not happen.")
                return -1

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

    def _insert_files_into_table(self, dataset_id: int, locs: List[Path]) -> List[int]:
        df_files = pl.DataFrame(
            [(dataset_id, str(loc)) for loc in locs], schema=["dataset_id", "location"], orient="row"
        )
        self._connection.register("df_files", df_files)
        logger.info(f"Inserting {len(locs)} files for dataset id = {dataset_id}")

        try:
            # Insert data and return IDs with associated locations
            insert_query = """
            INSERT INTO files (dataset_id, location)
            SELECT dataset_id, location FROM df_files
            RETURNING id, location;
            """
            result = self._connection.execute(insert_query).fetchall()

            # Create a mapping from location to id
            id_map = {loc: fid for fid, loc in result}

            # Retrieve file IDs in the order of locs
            file_ids = [id_map[str(loc)] for loc in locs]

        except duckdb.Error as err:
            logger.error(f"DuckDB error during insertion of files: {err}")
            return []
        finally:
            self._connection.unregister("df_files")

        self._connection.commit()
        self._db_incr_version()
        return file_ids

    def _add_columns_to_samples_table(self, properties: list[MetadataProperty]) -> None:
        cur = self._connection.cursor()

        # Fetch all existing column names once
        cur.execute("SELECT name FROM pragma_table_info('samples');")
        existing_columns = set(row[0] for row in cur.fetchall())
        columns_to_add = [prop for prop in properties if prop.name not in existing_columns]

        if columns_to_add:
            for prop in columns_to_add:
                # Decide column type
                if prop.dtype == "STRING":
                    column_type = "VARCHAR[]" if prop.multiple else "VARCHAR"
                elif prop.dtype == "ENUM":
                    # Create ENUM type if not exists
                    enum_type_name = f"enum_{prop.name}"
                    cur.execute(
                        "SELECT COUNT(*) FROM duckdb_types WHERE LOWER(type_name) = LOWER(?)", (enum_type_name,)
                    )
                    result = cur.fetchone()
                    if result is None or result[0] == 0:
                        # It is important to sort here,
                        # otherwise across runs (e.g., in unit tests), we get inconsistent ordering.
                        enum_values = "', '".join(sorted(prop.enum_options))
                        create_enum_query = f"CREATE TYPE {enum_type_name} AS ENUM ('{enum_values}');"
                        cur.execute(create_enum_query)
                    column_type = f"{enum_type_name}[]" if prop.multiple else enum_type_name
                else:
                    raise RuntimeError(f"Unsupported dtype {prop.dtype} for property {prop.name}")

                # Define NULL or NOT NULL
                nullable_str = "NULL" if prop.nullable else "NOT NULL"
                # TODO(https://github.com/duckdb/duckdb/issues/57): DuckDB currently does not support this.
                nullable_str = ""
                # Add the column
                alter_query = f"ALTER TABLE samples ADD COLUMN {prop.name} {column_type} {nullable_str};"
                cur.execute(alter_query)

            self._connection.commit()

    def _insert_samples_with_metadata(
        self, dataset_id: int, results: list[tuple[int, list[dict]]], properties: list[MetadataProperty]
    ) -> None:
        if not results:
            logger.warning(f"No metadata extracted for dataset {dataset_id}")
            return

        sorted_metadata_keys = sorted([prop.name for prop in properties])
        all_columns = ["dataset_id", "file_id", "sample_id"] + sorted_metadata_keys

        # Initialize the data dict with empty lists for each column
        data_columns: dict[str, Any] = {key: [] for key in all_columns}

        # Collect data column-wise
        for file_id, metadata in results:
            for sample in metadata:
                data_columns["dataset_id"].append(dataset_id)
                data_columns["file_id"].append(file_id)
                data_columns["sample_id"].append(sample["sample_id"])
                for key in sorted_metadata_keys:
                    data_columns[key].append(sample.get(key))

        logger.debug("Collected column-wise data for constructing pyarrow table.")

        # We need to construct the pyarrow table with consistent column ordering
        # We could do pa.Table.from_arrays([pa.array(data_columns[key]) for key in all_columns],
        # but that uses pyarrows type inference.
        # Let's be explicit in typing to avoid issues down the line

        arrays = []
        for col_name in all_columns:
            values = data_columns[col_name]
            if col_name in ["dataset_id", "file_id", "sample_id"]:
                arrays.append(pa.array(values, type=pa.int64()))
            else:
                prop = next((p for p in properties if p.name == col_name), None)
                assert prop is not None, "Should not happen"
                if prop.dtype in ["STRING", "ENUM"]:
                    array = pa.array(values, type=pa.list_(pa.string()) if prop.multiple else pa.string())
                else:
                    # Extend later for numeric columns
                    raise RuntimeError(f"Unsupported dtype {prop.dtype} for property {prop.name}")
                arrays.append(array)

        table = pa.Table.from_arrays(arrays, names=all_columns)

        logger.debug("Constructed PyArrow Table.")

        # It is important to specify the columns in the insertion query
        # Otherwise we do not guarantee that the values silently land in a different column.
        columns = ", ".join(all_columns)
        try:
            self._connection.register("samples_data", table)
            self._connection.execute(f"INSERT INTO samples ({columns}) SELECT {columns} FROM samples_data")
            self._connection.commit()
            logger.debug("Data inserted successfully.")
        except Exception as e:
            logger.error(f"Error during data insertion using PyArrow Table: {e}")
            raise
        finally:
            self._connection.unregister("samples_data")

    def check_dataset_exists(self, identifier: str) -> bool:
        try:
            query = "SELECT COUNT(*) from datasets WHERE name = ?;"
            cur = self._connection.cursor()
            result = cur.execute(query, (identifier,)).fetchone()
        except duckdb.Error as err:
            logger.error(f"A DuckDB error occurred during selection: {err}")
            return False

        if result is None:
            logger.error("result is None without any DuckDB error. This should not happen.")
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

            delete_files_query = """
            DELETE FROM files
            WHERE dataset_id IN (
                SELECT id FROM datasets WHERE name = ?
            );
            """
            cur.execute(delete_files_query, (identifier,))

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

        self._add_columns_to_samples_table([property_name])
        self._insert_property_values(property_name, new_properties)
        self._db_incr_version()
        return True

    def _insert_property_values(self, property_name: str, new_properties: list[dict[str, Any]]) -> None:
        if not new_properties:
            logger.warning(f"No new properties to insert for {property_name}.")
            return

        df = pl.DataFrame(new_properties)

        conn = self._connection

        # Updating samples table with the new property values
        for row in df.iter_rows():
            dataset_id = int(row[0])
            file_id = int(row[1])
            sample_id = int(row[2])
            property_value = row[3]

            property_value_native = numpy_to_native(property_value)

            # Since the property columns are VARCHAR[] (list of strings), ensure property_value is a list
            if not isinstance(property_value_native, list):
                property_value_native = [property_value_native]

            logger.error(property_name)
            logger.error((property_value_native, dataset_id, file_id, sample_id))

            # TODO(#117): See: https://github.com/duckdb/duckdb/issues/3265
            # We cannot update the property column here as it is a list column.
            # We need to either find a hack for this (remove samples & reinsert?) or wait for DuckDB to fix this.
            # This has been an issue in DuckDB for some years, so probably we want to hack this.
            _ = """cursor.execute(
                f"UPDATE samples SET {property_name} = ? WHERE dataset_id = ? AND file_id = ? AND sample_id = ?;",
                (property_value_native, dataset_id, file_id, sample_id),
            )"""
            raise NotImplementedError("DuckDB currently does not support updates on list columns.")

        conn.commit()

    def __getstate__(self) -> dict:
        # We cannot pickle the DuckDB connection.
        d = dict(self.__dict__)
        del d["_connection"]
        return d

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        # self._connection = self._load_db_from_disk()
        logger.warning(
            "Re-instantiating the MDC after pickling. "
            + "This should only happen within a dataloader worker running locally using spawn. "
            + "We will not hold a connection to the DuckDB anymore, since the DuckDB does not allow this. "
        )
