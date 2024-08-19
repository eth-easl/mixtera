import hashlib
import os
import shutil
from pathlib import Path

import dill
from loguru import logger
from mixtera.core.datacollection.mixtera_data_collection import MixteraDataCollection
from mixtera.core.query.query import Query
from mixtera.core.query.query_result import QueryResult


class QueryCache:
    def __init__(self, directory: str | Path, mdc: MixteraDataCollection):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._mdc = mdc
        logger.debug(f"Initializing QueryCache at {self.directory}")
        self.enabled = True

    def _get_query_hash(self, query: Query) -> str:
        query_string = str(query)
        return hashlib.sha256(query_string.encode()).hexdigest()

    def cache_query(self, query: Query) -> None:
        if not self.enabled:
            return

        query_hash = self._get_query_hash(query)
        hash_dir = self.directory / query_hash
        hash_dir.mkdir(exist_ok=True)

        existing_files = sorted(hash_dir.glob("*.pkl"))
        if existing_files:
            last_file = existing_files[-1]
            file_number = int(last_file.stem) + 1
        else:
            file_number = 0

        cache_path = hash_dir / f"{file_number}.pkl"
        db_ver = self._mdc.get_db_version()
        cache_obj = {"db_version": db_ver, "query": query}
        logger.debug(f"Caching query with hash {query_hash} and db version {db_ver}")
        _lock, _index = query.results._lock, query.results._index
        del query.results._lock
        del query.results._index
        with open(cache_path, "wb") as file:
            dill.dump(cache_obj, file)
        query.results._lock = _lock
        query.results._index = _index

    def get_queryresults_if_cached(self, query: Query) -> None | QueryResult:
        if not self.enabled:
            return None

        if str(query) == "" or str(query) is None or query is None:
            raise RuntimeError(f"Invalid string representation of query: {str(query)}")

        query_hash = self._get_query_hash(query)
        hash_dir = self.directory / query_hash

        if not hash_dir.exists():
            logger.debug(f"No directory found at {hash_dir} for query with hash {query_hash}.")
            return None

        if len(os.listdir(hash_dir)) == 0:
            logger.debug(f"Directory {hash_dir} for {query_hash} is empty: {os.listdir(hash_dir)}")
            shutil.rmtree(hash_dir)
            return None

        for cache_file in hash_dir.glob("*.pkl"):
            with open(cache_file, "rb") as file:
                logger.debug(f"Checking file {cache_file} in cache.")
                cached_query = dill.load(file)
                if str(cached_query["query"]) == str(query):
                    logger.debug(f"Found matching query for version {cached_query['db_version']}.")
                    # Check if cache is still valid
                    if cached_query["db_version"] != self._mdc.get_db_version():
                        logger.debug("Database has been updated, removing file.")
                        # Cache is outdated
                        os.remove(cache_file)
                        if not os.listdir(hash_dir):
                            logger.debug(f"Directory for {query_hash} is empty after removing the file.")
                            shutil.rmtree(hash_dir)
                        return None
                    logger.debug("Returning results from cache!")
                    return cached_query["query"].results
                logger.debug(f"'{cached_query['query']}' does not match '{str(query)}'.")

        return None
