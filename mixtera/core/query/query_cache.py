import hashlib
import os
import pickle
import shutil
from pathlib import Path

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

    def cache_query(self, query: Query) -> None | Path:
        if not self.enabled:
            return None

        query_hash = self._get_query_hash(query)
        hash_dir = self.directory / query_hash
        hash_dir.mkdir(exist_ok=True)

        existing_dirs = sorted([int(d.name) for d in hash_dir.iterdir() if d.is_dir() and d.name.isdigit()])
        if existing_dirs:
            next_number = existing_dirs[-1] + 1
        else:
            next_number = 0

        query_dir = hash_dir / str(next_number)
        query_dir.mkdir()

        # Prepare the cache metadata
        db_ver = self._mdc.get_db_version()
        cache_obj = {"db_version": db_ver, "query_str": str(query)}

        # Save the metadata
        meta_path = query_dir / "meta.pkl"
        with meta_path.open("wb") as f:
            pickle.dump(cache_obj, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Call the to_cache method on query.results to serialize it
        query.results.to_cache(query_dir)

        logger.debug(f"QueryResult saved to {query_dir}")
        return query_dir

    def get_queryresults_if_cached(self, query: Query) -> None | tuple[QueryResult, Path]:
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

        for query_dir in hash_dir.iterdir():
            if query_dir.is_dir():
                meta_path = query_dir / "meta.pkl"

                if not meta_path.exists():
                    logger.warning(f"Meta file missing, invalid query dir? ({query_dir})")
                    continue

                with meta_path.open("rb") as f:
                    cache_obj = pickle.load(f)
                    cached_query_str = cache_obj.get("query_str")
                    db_ver = cache_obj.get("db_version")

                if cached_query_str == str(query):
                    if db_ver != self._mdc.get_db_version():
                        logger.debug(f"Database has been updated, removing cached query at {query_dir}.")
                        shutil.rmtree(query_dir)
                        if not os.listdir(hash_dir):
                            logger.debug(f"Directory for {query_hash} is empty after removing the cache.")
                            shutil.rmtree(hash_dir)
                        return None
                    logger.debug("Returning results from cache!")
                    # Load the QueryResult from the cache
                    query_result = QueryResult.from_cache(query_dir)
                    return query_result, query_dir

                logger.debug(f"Cached query does not match: '{cached_query_str}' != '{str(query)}'.")
        return None
