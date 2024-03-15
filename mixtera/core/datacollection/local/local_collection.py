import multiprocessing as mp
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Generator, List, Optional, Type

import dill
from loguru import logger
from mixtera.core.datacollection import MixteraDataCollection, Property, PropertyType
from mixtera.core.datacollection.datasets import Dataset
from mixtera.core.datacollection.index import Index, IndexType
from mixtera.core.datacollection.index.index_collection import IndexFactory, IndexTypes, InMemoryDictionaryRangeIndex
from mixtera.core.datacollection.index.parser import MetadataParserFactory
from mixtera.core.processing import ExecutionMode
from mixtera.core.processing.property_calculation.executor import PropertyCalculationExecutor
from mixtera.utils.utils import defaultdict_to_dict, numpy_to_native_type, wait_for_key_in_dict

if TYPE_CHECKING:
    from mixtera.core.query import LocalQueryResult, Query, QueryResult


class LocalDataCollection(MixteraDataCollection):

    def __init__(self, directory: Path) -> None:
        if not directory.exists():
            raise RuntimeError(f"Directory {directory} does not exist.")

        self._directory = directory
        self._database_path = self._directory / "mixtera.sqlite"

        self._properties: list[Property] = []
        self._datasets: list[Dataset] = []

        # Index instantiations and parameters
        self._index: Index = IndexFactory.create_index(IndexTypes.IN_MEMORY_DICT_RANGE)
        self._metadata_factory = MetadataParserFactory()

        if not self._database_path.exists():
            self._connection = self._init_database()
        else:
            self._connection = sqlite3.connect(self._database_path)

        self._queries_lock = mp.Lock()
        self._queries: list[tuple[Query, int]] = []  # (query, chunk_size)
        self._training_query_map: dict[str, int] = {}

    def __getstate__(self) -> dict:
        # We cannot pickle the sqlite connection.
        d = dict(self.__dict__)
        del d["_connection"]
        return d

    def _init_database(self) -> sqlite3.Connection:
        assert hasattr(self, "_database_path")
        assert not self._database_path.exists()
        logger.info("Initializing database.")
        conn = sqlite3.connect(self._database_path)
        cur = conn.cursor()
        cur.execute(
            "CREATE TABLE IF NOT EXISTS datasets"
            " (id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL, name TEXT NOT NULL UNIQUE,"
            " location TEXT NOT NULL, type INTEGER NOT NULL,"
            " parsing_func BLOB NOT NULL);"
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
        if isinstance(loc, Path):
            loc = str(loc)

        if (dataset_id := self._insert_dataset_into_table(identifier, loc, dtype, parsing_func)) == -1:
            return False

        file: Path
        for file in dtype.iterate_files(loc):
            if (file_id := self._insert_file_into_table(dataset_id, file)) == -1:
                logger.error(f"Error while inserting file {file}")
                return False
            metadata_parser = self._metadata_factory.create_metadata_parser(metadata_parser_type, dataset_id, file_id)
            dtype.build_file_index(file, metadata_parser)
            metadata_parser.finalize()
            self._insert_index_into_table(metadata_parser.get_index())
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

        type_id = dtype.type_id
        if type_id == 0:
            logger.error("Cannot use generic Dataset class as dtype.")
            valid_types = False

        if not valid_types:
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

    def _insert_index_into_table(self, index: InMemoryDictionaryRangeIndex, full_or_fail: bool = False) -> int:
        """
        Inserts an `InMemoryDictionaryRangeIndex` into the index table. This
        method bulk-schedules the insertion of each row.

        Args:
            index: the index to be inserted
            full_or_fail: if True, and not all rows are inserted, the method will fail.
                If False, partial insertion in the DB are allowed and are only reported.

        Returns:
            The number of inserted rows. If insertion fails -1 will be returned.
        """
        cur = self._connection.cursor()
        query = "INSERT INTO indices (property_name, property_value, dataset_id, file_id, line_start, line_end) \
            VALUES (?, ?, ?, ?, ?, ?);"

        # Build a large payload to schedule the execution of many SQL statements
        query_payload = []
        raw_index = index.get_full_dict_index()
        for property_name, property_values in raw_index.items():
            for property_value, dataset_ids in property_values.items():
                for dataset_id, file_ids in dataset_ids.items():
                    for file_id, ranges in file_ids.items():
                        for range_values in ranges:
                            query_payload.append(
                                (
                                    property_name,
                                    property_value,
                                    dataset_id,
                                    file_id,
                                    range_values[0],
                                    range_values[1],
                                )
                            )

        # Try to execute statement
        try:
            cur.executemany(query, query_payload)
            self._connection.commit()
        except sqlite3.Error as err:
            logger.error(f"An sqlite error occurred when bulk inserting index: {err}")
            return -1

        # Assert that insertion completed fully or partially
        if cur.rowcount != len(query_payload):
            error_message = f"Failed to insert fully: {cur.rowcount} out of {len(query_payload)} rows inserted!"
            logger.error(error_message)
            if full_or_fail:
                raise AssertionError(error_message)

        return cur.rowcount

    def _insert_partial_index_into_table(self, property_name: str, partial_index: Index) -> int:
        query = "INSERT INTO indices (property_name, property_value, dataset_id, file_id, line_start, line_end) \
            VALUES (?, ?, ?, ?, ?, ?);"
        cur = self._connection.cursor()
        partial_index = defaultdict_to_dict(partial_index)
        partial_index = numpy_to_native_type(partial_index)
        try:
            for prediction in partial_index:
                for dataset_id in partial_index[prediction]:
                    for file_id in partial_index[prediction][dataset_id]:
                        for line_id in partial_index[prediction][dataset_id][file_id]:
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

    def _reformat_index(self, raw_indices: List) -> InMemoryDictionaryRangeIndex:
        # received from database: [(property_name, property_value, dataset_id, file_id, line_ids), ...]
        # converts to: {property_name: {property_value: {dataset_id: {file_id: [line_ids]}}}}
        index = IndexFactory.create_index(IndexTypes.IN_MEMORY_DICT_RANGE)
        for prop_name, prop_val, dataset_id, file_id, line_start, line_end in raw_indices:
            index.append_entry(prop_name, prop_val, dataset_id, file_id, (line_start, line_end))
        return index

    def _read_index_from_database(self, property_name: Optional[str] = None) -> InMemoryDictionaryRangeIndex:
        cur = self._connection.cursor()
        try:
            query = "SELECT property_name, property_value, dataset_id, file_id, line_start, line_end from indices"
            if property_name:
                query += " WHERE property_name = ?;"
                cur.execute(query, (property_name,))
            else:
                query += ";"
                cur.execute(query)
            results = self._reformat_index(cur.fetchall())
        except sqlite3.Error as err:
            logger.error(f"A sqlite error occured during selection: {err}")
            results = IndexFactory.create_index(IndexTypes.IN_MEMORY_DICT_RANGE)
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

    def is_remote(self) -> bool:
        return False

    def stream_query_results(
        self, query_result: "QueryResult", tunnel_via_server: bool = False
    ) -> Generator[str, None, None]:
        if tunnel_via_server:
            raise RuntimeError("Cannot tunnel via server on a LocalDataCollection, can only do this remotely!")

        yield from MixteraDataCollection._stream_query_results(query_result, None)

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
        self._insert_partial_index_into_table(property_name, new_index)

    def get_index(self, property_name: Optional[str] = None) -> Optional[InMemoryDictionaryRangeIndex]:
        if property_name is None:
            logger.warning(
                "No property name provided, returning all indices from database. ",
                "This may be slow, consider providing a property name.",
            )
            self._index = self._read_index_from_database()
            return self._index
        if not self._index.has_feature(property_name):
            # If the property is not in the index, it may be in the database, so we check it there
            # TODO(xiaozhe): user may also interested to force refresh the index from database.
            self._index.merge(self._read_index_from_database(property_name))
        if not self._index.has_feature(property_name):
            logger.warning(f"Property {property_name} not found in index, returning None.")
            return None
        # The type of self._index and the returned value is `InMemoryDictionaryRangeIndex`
        return self._index.get_index_by_features(property_name)

    def register_query(self, query: "Query", chunk_size: int) -> int:
        if query.training_id in self._training_query_map:
            logger.warning(f"We already have a query for training {query.training_id}!")
            return -1

        with self._queries_lock:
            self._queries.append((query, chunk_size))
            index = len(self._queries) - 1
            self._training_query_map[query.training_id] = index

        logger.info(
            f"Registered query {str(query)} with chunk_size {chunk_size}" + f" for training {query.training_id}."
        )

        return index

    def get_query_result(self, training_id: str) -> "LocalQueryResult":
        if (query_id := self.get_query_id(training_id)) < 0:
            raise RuntimeError(f"Unknown training {training_id}")
        # Since queries are only registered after they are executed, results is guaranteed to not be None
        return self._queries[query_id][0].results

    def get_query_id(self, training_id: str) -> int:
        if wait_for_key_in_dict(self._training_query_map, training_id, 120.0):
            query_id = self._training_query_map[training_id]
            logger.debug(f"Query ID for training {training_id} is {query_id}")
            return query_id

        logger.warning(f"Did not find query ID for training {training_id} after 15 seconds.")
        return -1

    def next_query_result_chunk(self, query_id: int) -> Optional[IndexType]:
        # Note that this call is inherently thread-safe
        return next(self._queries[query_id][0].results, None)
