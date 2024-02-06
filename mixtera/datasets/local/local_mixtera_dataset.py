import json
import sqlite3
from pathlib import Path
from typing import Any

from loguru import logger
from mixtera.datasets import DatasetTypes, MixteraDataset
from mixtera.utils import ranges


class LocalMixteraDataset(MixteraDataset):

    def __init__(self, directory: Path) -> None:
        if not directory.exists():
            raise RuntimeError(f"Directory {directory} does not exist.")

        self._directory = directory
        self._database_path = self._directory / "mixtera.sqlite"

        self._hacky_indx: dict[Any, Any] = {}  # TODO(#8): Actually store index in sqlite instead of memory

        if not self._database_path.exists():
            self._init_database()
            assert self._database_path.exists()
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
            " (id PRIMARY KEY AUTOINCREMENT NOT NULL, name TEXT NOT NULL,"
            " location TEXT NOT NULL, type INTEGER NOT NULL);"
        )
        cur.execute(
            "CREATE TABLE IF NOT EXISTS files (id PRIMARY KEY AUTOINCREMENT NOT NULL, location TEXT NOT NULL,);"
        )
        conn.commit()
        logger.info("Database initialized.")

        return conn

    def register_dataset(self, identifier: str, loc: str, dtype: DatasetTypes) -> bool:
        if not self._insert_dataset_into_table(identifier, loc, dtype):
            return False

        if dtype == DatasetTypes.JSONL:
            return self._register_jsonl_dataset(identifier, loc)

        raise NotImplementedError(f"Unsupported dataset type: {dtype}")

    def check_dataset_exists(self, identifier: str) -> bool:
        raise NotImplementedError("Not yet implemented")

    def _insert_dataset_into_table(self, identifier: str, loc: str, dtype: DatasetTypes) -> bool:
        query = "INSERT INTO datasets (name, location, type) VALUES (?, ?, ?)"
        cur = self._connection.cursor()
        cur.execute(query, (identifier, loc, dtype.value))
        self._connection.commit()

        if cur.rowcount == 1:
            logger.info(f"Sucessfully registered dataset {identifier}.")
            return True

        logger.error(f"Failed to register dataset {identifier}.")
        return False

    def _insert_file_into_table(self, loc: str) -> int:
        # TODO(create issue): Potentially batch inserts of multiple files if this is too slow.
        query = "INSERT INTO files (location) VALUES (?)"
        cur = self._connection.cursor()
        cur.execute(query, (loc))
        self._connection.commit()

        if cur.rowcount == 1:
            assert cur.lastrowid is not None and cur.lastrowid >= 0
            return cur.lastrowid

        logger.error(f"Failed to register file {loc}.")
        return -1

    def _register_jsonl_dataset(self, identifier: str, loc: str) -> bool:
        # TODO(create issue): Can we make this idempotent to allow users to update? / Allow for recalculation

        loc_path = Path(loc)

        if not loc_path.exists():
            raise RuntimeError(f"Could not find directory {loc_path}")

        # TODO(#8): Initialize index here

        for jsonl_file in loc_path.glob("*.jsonl"):
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

        # For now, I just hardcode the SlimPajama example. We have to generalize this to UDFs
        index: dict[str, Any] = {"language": {}, "publication_date": {}}
        with open(file, encoding="utf-8") as fd:
            for line_id, line in enumerate(fd):
                json_obj = json.loads(line)
                if "meta" not in json_obj:
                    continue

                meta = json_obj["meta"]
                del json_obj

                for index_field in index:  # pylint: disable=consider-using-dict-items
                    if index_field not in meta:
                        continue

                    # TODO(#11): Support numerical buckets, not just categorical

                    value = meta[index_field]
                    if value not in index[index_field]:
                        index[index_field][value] = [line_id]
                    else:
                        index[index_field][value].append(line_id)

        # Rangi-fy the list for each index, i.e., we go from [1,2,3,5,6,7] to [(1,4), (5,8)] to compress
        # Additionally, we add the current file ID
        for index_field, buckets in index.items():
            for bucket_key, bucket_vals in buckets.copy():
                buckets[bucket_key] = ((file_id,) + rang for rang in ranges(bucket_vals))

        # TODO(#8): Extend sqlite index instead of in-memory index
        if identifier not in self._hacky_indx:
            self._hacky_indx[identifier] = {}

        for index_field, buckets in index.items():
            if index_field not in self._hacky_indx[identifier]:
                self._hacky_indx[identifier][index_field] = buckets
            else:
                for bucket_key, bucket_vals in buckets:
                    if bucket_key not in self._hacky_indx[identifier][index_field]:
                        self._hacky_indx[identifier][index_field][bucket_key] = bucket_vals
                    else:
                        self._hacky_indx[identifier][index_field][bucket_key].extend(bucket_vals)

        return True
