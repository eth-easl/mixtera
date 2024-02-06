import json
import sqlite3
from pathlib import Path

from loguru import logger
from mixtera.datasets import DatasetTypes, MixteraDataset
from mixtera.utils import ranges


class LocalMixteraDataset(MixteraDataset):

    def __init__(self, directory: Path) -> None:
        if not directory.exists():
            raise RuntimeError(f"Directory {directory} does not exist.")

        self._directory = directory
        self._database_path = self._directory / "mixtera.sqlite"

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

    def _register_jsonl_dataset(self, identifier: str, loc: str) -> bool:
        # TODO(create issue): Can we make this idempotent to allow users to update?

        loc = Path(loc)

        if not loc.exists():
            raise RuntimeError(f"Could not find directory {loc}")

        # TODO(#8): Initialize index here

        for jsonl_file in loc.glob("*.jsonl"):
            if not self._register_jsonl_file(identifier, jsonl_file):
                logger.error(f"Error while registering file {jsonl_file}.")
                return False

    def _register_jsonl_file(self, identifier: str, file: Path) -> bool:
        assert file.exists()

        # TODO(#7, #8): Extend dataset index correctly with this file

        file_id = 42  # todo obtain by inserting into a new sqlite table

        # For now, I just hardcode the SlimPajama example. We somehow have to generalize this.
        index = {"language": {}, "publication_date": {}}
        with open(file) as fd:
            for line_id, line in enumerate(fd):
                file = json.loads(line)
                if "meta" not in file:
                    continue

                meta = file["meta"]
                del file

                for index_field in index.keys():
                    if index_field not in meta:
                        continue

                    # TODO: support numerical buckets, not just categorical

                    value = meta[index_field]
                    if value not in index[index_field]:
                        index[index_field][value] = [line_id]
                    else:
                        index[index_field][value].append(line_id)

        # Rangi-fy the list for each index, i.e., we go from [1,2,3,5,6,7] to [(1,4), (5,8)] to compress
        for index_field, buckets in index.items():
            for bucket_key, bucket_vals in buckets:
                index[index_field][bucket_key] = ((file_id,) + rang for rang in ranges(bucket_vals))

        for index_field, buckets in index.items():
            pass  # TODO extend sqlite index for each field by this
