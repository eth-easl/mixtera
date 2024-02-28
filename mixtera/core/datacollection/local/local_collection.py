import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Type

import dill
from loguru import logger
from mixtera.core.datacollection import IndexType, MixteraDataCollection, Property, PropertyType
from mixtera.core.datacollection.datasets import Dataset
from mixtera.core.datacollection.index import MetadataParserFactory
from mixtera.core.processing import ExecutionMode
from mixtera.core.processing.property_calculation.executor import PropertyCalculationExecutor
from mixtera.utils.utils import defaultdict_to_dict, merge_defaultdicts, numpy_to_native_type


class LocalDataCollection(MixteraDataCollection):

    def __init__(self, directory: Path) -> None:
        if not directory.exists():
            raise RuntimeError(f"Directory {directory} does not exist.")

        self._directory = directory
        self._database_path = self._directory / "mixtera.sqlite"

        self._properties: list[Property] = []
        self._datasets: list[Dataset] = []
        # 1st level: Variable 2nd Level: Buckets for that Variable 3rd level: datasets 4th: files -> ranges
        self._index: IndexType = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))

        self._metadata_factory = MetadataParserFactory()

        if not self._database_path.exists():
            self._connection = self._init_database()
        else:
            self._connection = sqlite3.connect(self._database_path)

    def _init_database(self) -> sqlite3.Connection:
        assert hasattr(self, "_database_path")
        assert not self._database_path.exists()
        logger.info("Initializing database.")
        conn = sqlite3.connect(self._database_path)
        cur = conn.cursor()
        cur.execute(
            "CREATE TABLE IF NOT EXISTS datasets"
            " (id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL, name TEXT NOT NULL UNIQUE,"
            " location TEXT NOT NULL, type INTEGER NOT NULL, parsing_func BLOB NOT NULL);"
        )

        cur.execute(
            "CREATE TABLE IF NOT EXISTS files"
            " (id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,"
            " dataset_id INTEGER NOT NULL,"
            " location TEXT NOT NULL,"
            " FOREIGN KEY(dataset_id) REFERENCES datasets(id)"
            " ON DELETE CASCADE);"
        )
        cur.execute(
            "CREATE TABLE IF NOT EXISTS indices"
            " (id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,"
            " property_name TEXT NOT NULL,"
            " property_value TEXT NOT NULL,"
            " dataset_id INTEGER NOT NULL,"
            " file_id INTEGER NOT NULL,"
            " line_start INTEGER NOT NULL,"
            " line_end INTEGER NOT NULL);"
        )
        conn.commit()
        logger.info("Database initialized.")

        return conn

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
            pre_index = dtype.build_file_index(file, metadata_parser)
            for property_name in pre_index:
                self._insert_index_into_table(property_name, pre_index[property_name])
        return True

    def _insert_dataset_into_table(
        self, identifier: str, loc: str, dtype: Type[Dataset], parsing_func: Callable[[str], str]
    ) -> int:
        if not issubclass(dtype, Dataset):
            logger.error(f"Invalid dataset type: {dtype}")
            return -1

        type_id = dtype.type_id
        if type_id == 0:
            logger.error("Cannot use generic Dataset class as dtype.")
            return -1

        serialized_parsing_func = sqlite3.Binary(dill.dumps(parsing_func))

        try:
            query = "INSERT INTO datasets (name, location, type, parsing_func) VALUES (?, ?, ?, ?);"
            cur = self._connection.cursor()
            cur.execute(query, (identifier, loc, type_id, serialized_parsing_func))
            self._connection.commit()
            inserted_id = cur.lastrowid
        except sqlite3.Error as err:
            logger.error(f"A sqlite error occured during insertion: {err}")
            return -1

        if inserted_id:
            logger.info(f"Successfully registered dataset {identifier} with id {inserted_id}.")
            return inserted_id  # Return the last inserted id

        logger.error(f"Failed to register dataset {identifier}.")
        return -1

    def _insert_file_into_table(self, dataset_id: int, loc: Path) -> int:
        # TODO(create issue): Potentially batch inserts of multiple files if this is too slow.
        query = "INSERT INTO files (dataset_id, location) VALUES (?, ?);"
        cur = self._connection.cursor()
        logger.info(f"Inserting file at {loc} for dataset id = {dataset_id}")
        cur.execute(
            query,
            (
                dataset_id,
                str(loc),
            ),
        )
        self._connection.commit()

        if cur.rowcount == 1:
            assert cur.lastrowid is not None and cur.lastrowid >= 0
            return cur.lastrowid

        logger.error(f"Failed to register file {loc}.")
        return -1

    def _insert_index_into_table(self, property_name: str, index: IndexType) -> int:
        query = "INSERT INTO indices (property_name, property_value, dataset_id, file_id, line_start, line_end) \
            VALUES (?, ?, ?, ?, ?, ?);"
        cur = self._connection.cursor()
        index = defaultdict_to_dict(index)
        index = numpy_to_native_type(index)
        try:
            for prediction in index:
                for dataset_id in index[prediction]:
                    for file_id in index[prediction][dataset_id]:
                        for line_id in index[prediction][dataset_id][file_id]:
                            cur.execute(
                                query,
                                (
                                    property_name,
                                    prediction,
                                    dataset_id,
                                    file_id,
                                    line_id[0],
                                    line_id[1],
                                ),
                            )
            self._connection.commit()

        except sqlite3.Error as err:
            logger.error(f"A sqlite error occured during insertion: {err}")
            return -1
        if cur.rowcount == 1:
            assert cur.lastrowid is not None and cur.lastrowid >= 0
            return cur.lastrowid

        logger.error(f"Failed to register index for property {property_name}.")

        return -1

    def _reformat_index(self, raw_indices: List) -> IndexType:
        # received from database: [(property_name, property_value, dataset_id, file_id, line_ids), ...]
        # converts to: {property_name: {property_value: {dataset_id: {file_id: [line_ids]}}}}
        index: IndexType = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
        for prop_name, prop_val, dataset_id, file_id, line_start, line_end in raw_indices:
            index[prop_name][prop_val][dataset_id][file_id] = [(line_start, line_end)]
        return index

    def _read_index_from_database(self, property_name: Optional[str] = None) -> IndexType:
        cur = self._connection.cursor()
        try:
            query = "SELECT property_name, property_value, dataset_id, file_id, line_start, line_end from indices"
            if property_name:
                query += " WHERE property_name = ?;"
                cur.execute(query, (property_name,))
            else:
                query += ";"
                cur.execute(query)
            results = cur.fetchall()
        except sqlite3.Error as err:
            logger.error(f"A sqlite error occured during selection: {err}")
            results = []
        results = self._reformat_index(results)
        return results

    def check_dataset_exists(self, identifier: str) -> bool:
        try:
            query = "SELECT COUNT(*) from datasets WHERE name = ?;"
            cur = self._connection.cursor()
            cur.execute(query, (identifier,))
            result = cur.fetchone()[0]
        except sqlite3.Error as err:
            logger.error(f"A sqlite error occured during selection: {err}")
            return False

        assert result <= 1
        return result == 1

    def list_datasets(self) -> List[str]:
        try:
            query = "SELECT name from datasets;"
            cur = self._connection.cursor()
            cur.execute(
                query,
            )
            result = cur.fetchall()
        except sqlite3.Error as err:
            logger.error(f"A sqlite error occured during selection: {err}")
            return []

        return [dataset for (dataset,) in result]

    def remove_dataset(self, identifier: str) -> bool:
        # Need to delete the dataset, and update the index to remove all pointers to files in the dataset
        raise NotImplementedError("Not implemented for LocalCollection")

    def _get_all_files(self) -> list[tuple[int, int, Type[Dataset], str]]:
        try:
            query = "SELECT files.id, files.dataset_id, files.location from files;"
            cur = self._connection.cursor()
            cur.execute(
                query,
            )
            result = cur.fetchall()
        except sqlite3.Error as err:
            logger.error(f"A sqlite error occured during selection: {err}")
            return []

        dataset_ids = set(did for _, did, _ in result)
        id_to_type_map = {did: self._get_dataset_type_by_id(did) for did in dataset_ids}

        return [(fid, did, id_to_type_map[did], loc) for fid, did, loc in result]

    def _get_dataset_func_by_id(self, did: int) -> Callable[[str], str]:
        try:
            query = "SELECT parsing_func from datasets WHERE id = ?;"
            cur = self._connection.cursor()
            cur.execute(query, (did,))
            result = cur.fetchone()
        except sqlite3.Error as err:
            logger.error(f"Error while selecting parsing_func for did {did}")
            raise RuntimeError(f"A sqlite error occured during selection: {err}") from err

        if result is None:
            raise RuntimeError(f"Could not get dataset parsing func by id for did {did}")

        return dill.loads(result[0])

    def _get_dataset_type_by_id(self, did: int) -> Type[Dataset]:
        try:
            query = "SELECT type from datasets WHERE id = ?;"
            cur = self._connection.cursor()
            cur.execute(query, (did,))
            result = cur.fetchone()
        except sqlite3.Error as err:
            logger.error(f"Error while selecting parsing_func for did {did}")
            raise RuntimeError(f"A sqlite error occured during selection: {err}") from err

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
            cur.execute(query, (fid,))
            result = cur.fetchone()
        except sqlite3.Error as err:
            logger.error(f"Error while selecting location for fid {fid}")
            raise RuntimeError(f"A sqlite error occured during selection: {err}") from err

        if result is None:
            raise RuntimeError(f"Could not get file path by id for file id {fid}")

        return result[0]

    def get_samples_from_ranges(
        self, ranges_per_dataset_and_file: dict[int, dict[int, list[tuple[int, int]]]]
    ) -> Iterable[str]:
        for dataset_id, file_dict in ranges_per_dataset_and_file.items():
            dataset_parsing_func = self._get_dataset_func_by_id(dataset_id)
            filename_dict = {
                self._get_file_path_by_id(file_id): file_ranges for file_id, file_ranges in file_dict.items()
            }
            yield from self._get_dataset_type_by_id(dataset_id).read_ranges_from_files(
                filename_dict, dataset_parsing_func
            )

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
        dop: int = 1,
        data_only_on_primary: bool = True,
    ) -> None:
        if len(property_name) <= 0:
            raise RuntimeError(f"Invalid property name: {property_name}")

        if property_type == PropertyType.NUMERICAL and max_val <= min_val:
            raise RuntimeError(f"max_val (= {max_val}) <= min_val (= {min_val})")

        if num_buckets < 2:
            raise RuntimeError(f"num_buckets = {num_buckets} < 2")

        if batch_size < 1:
            raise RuntimeError(f"batch_size = {batch_size} < 1")

        if property_type == PropertyType.CATEGORICAL and (min_val != 0.0 or max_val != 1.0 or num_buckets != 10):
            logger.warning(
                "For categorical properties, min_val/max_val/num_buckets do not have meaning,"
                " but deviate from their default value. Please ensure correct parameters."
            )

        # TODO(#11): support for numerical buckets
        if property_type == PropertyType.NUMERICAL:
            raise NotImplementedError("Numerical properties are not yet implemented")

        files = self._get_all_files()
        logger.info(f"Extending index for {len(files)} files.")

        executor = PropertyCalculationExecutor.from_mode(execution_mode, dop, batch_size, setup_func, calc_func)
        executor.load_data(files, data_only_on_primary)
        new_index = executor.run()
        self._insert_index_into_table(property_name, new_index)

    def get_index(self, property_name: Optional[str] = None) -> Optional[IndexType]:
        if property_name is None:
            logger.warning(
                "No property name provided, returning all indices from database. ",
                "This may be slow, consider providing a property name.",
            )
            self._index = self._read_index_from_database()
            return self._index
        if property_name not in self._index:
            # If the property is not in the index, it may be in the database, so we check it there
            # TODO(xiaozhe): user may also interested to force refresh the index from database.
            self._index = merge_defaultdicts(self._index, self._read_index_from_database(property_name))
        if property_name not in self._index:
            logger.warning(f"Property {property_name} not found in index, returning None.")
            return None
        return self._index[property_name]
