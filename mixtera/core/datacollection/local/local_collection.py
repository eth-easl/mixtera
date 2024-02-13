import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, List

from loguru import logger
from mixtera.core.datacollection import DatasetTypes, MixteraDataCollection, PropertyType
from mixtera.core.processing import ExecutionMode
from mixtera.core.processing.property_calculation.executor import PropertyCalculationExecutor
from mixtera.utils import dict_into_dict, ranges


class LocalDataCollection(MixteraDataCollection):

    def __init__(self, directory: Path) -> None:
        if not directory.exists():
            raise RuntimeError(f"Directory {directory} does not exist.")

        self._directory = directory
        self._database_path = self._directory / "mixtera.sqlite"

        # 1st level: categories 2nd level: buckets
        # TODO(#8): Actually store index in sqlite instead of memory
        self._hacky_indx: defaultdict[str, defaultdict[str, list]] = defaultdict(lambda: defaultdict(list))

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
            " location TEXT NOT NULL, type INTEGER NOT NULL);"
        )
        cur.execute(
            "CREATE TABLE IF NOT EXISTS files (id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL, location TEXT NOT NULL);"
        )
        conn.commit()
        logger.info("Database initialized.")

        return conn

    def register_dataset(self, identifier: str, loc: str, dtype: DatasetTypes) -> bool:
        if not self._insert_dataset_into_table(identifier, loc, dtype):
            return False

        if dtype in [DatasetTypes.JSONL_COLLECTION, DatasetTypes.JSONL_SINGLEFILE]:
            return self._register_jsonl_collection_or_file(identifier, loc)

        raise NotImplementedError(f"Unsupported dataset type: {dtype}")

    def _insert_dataset_into_table(self, identifier: str, loc: str, dtype: DatasetTypes) -> bool:
        try:
            query = "INSERT INTO datasets (name, location, type) VALUES (?, ?, ?);"
            cur = self._connection.cursor()
            cur.execute(query, (identifier, loc, dtype.value))
            self._connection.commit()
        except sqlite3.Error as err:
            logger.error(f"A sqlite error occured during insertion: {err}")
            return False

        if cur.rowcount == 1:
            logger.info(f"Sucessfully registered dataset {identifier}.")
            return True

        logger.error(f"Failed to register dataset {identifier}.")
        return False

    def _insert_file_into_table(self, loc: str) -> int:
        # TODO(create issue): Potentially batch inserts of multiple files if this is too slow.
        query = "INSERT INTO files (location) VALUES (?);"
        cur = self._connection.cursor()
        logger.info(f"Inserting file at {loc}")
        cur.execute(query, (loc,))
        self._connection.commit()

        if cur.rowcount == 1:
            assert cur.lastrowid is not None and cur.lastrowid >= 0
            return cur.lastrowid

        logger.error(f"Failed to register file {loc}.")
        return -1

    def _register_jsonl_collection_or_file(self, identifier: str, loc: str) -> bool:
        # TODO(#20): Can we make this idempotent to allow users to update? / Allow for recalculation

        loc_path = Path(loc)

        if not loc_path.exists():
            raise RuntimeError(f"Could not find directory {loc_path}")

        # TODO(#8): Initialize index here

        file_list = [loc_path] if loc_path.is_file() else loc_path.glob("*.jsonl")
        for jsonl_file in file_list:
            if not self._register_jsonl_file(identifier, jsonl_file):
                logger.error(f"Error while registering file {jsonl_file}.")
                return False

        return True

    def _register_jsonl_file(self, identifier: str, file: Path) -> bool:
        assert file.exists()

        logger.debug(f"Registering file {file}")

        # TODO(#7, #8): Extend dataset index correctly with this file
        file_id = self._insert_file_into_table(str(file))
        if file_id == -1:
            logger.error(f"Error while inserting file {file}")
            return False

        index = self._build_index_for_jsonl_file(identifier, file, file_id)

        # TODO(#8): Extend sqlite index instead of in-memory index
        self._merge_index(index)

        return True

    def _build_index_for_jsonl_file(self, identifier: str, file: Path, file_id: int) -> dict[str, dict[str, list]]:
        # For now, I just hardcode the SlimPajama example. We have to generalize this to UDFs
        # index is the local index which we merge into the global index
        index: dict[str, dict[str, list]] = {
            "language": defaultdict(list),
            "publication_date": defaultdict(list),
        }
        max_line = 0
        with open(file, encoding="utf-8") as fd:
            for line_id, line in enumerate(fd):
                max_line = line_id
                json_obj = json.loads(line)
                if "meta" not in json_obj:
                    continue

                meta = json_obj["meta"]
                del json_obj

                for index_field in index:  # pylint: disable=consider-using-dict-items
                    if index_field not in meta:
                        continue
                    value = meta[index_field]

                    if index_field == "language":
                        # This is especially critical to generalize.
                        for lang in value:
                            lang_name = lang["name"]
                            index[index_field][lang_name].append(line_id)
                    else:
                        # TODO(#11): Support numerical buckets, not just categorical
                        # logger.info(f"for index {index_field} the value is {value}")
                        index[index_field][value].append(line_id)

        # Rangi-fy the list for each index, i.e., we go from [1,2,3,5,6,7] to [(1,4), (5,8)] to compress
        # Additionally, we add the current file ID
        for index_field, buckets in index.items():
            for bucket_key, bucket_vals in buckets.copy().items():
                buckets[bucket_key] = [(file_id,) + rang for rang in ranges(bucket_vals)]

        index["dataset"] = {identifier: [(file_id, 0, max_line + 1)]}

        return index

    def _merge_index(self, new_index: dict[str, Any]) -> None:
        # TODO(#8): Extend sqlite index instead of in-memory index
        dict_into_dict(self._hacky_indx, new_index)

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

    def _get_all_files(self) -> list[tuple[int, str]]:
        try:
            query = "SELECT id,location from files;"
            cur = self._connection.cursor()
            cur.execute(
                query,
            )
            result = cur.fetchall()
        except sqlite3.Error as err:
            logger.error(f"A sqlite error occured during selection: {err}")
            return []

        return result

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
        self._merge_index({property_name: executor.run()})

        # TODO(create issue): add functions to query all available properties
