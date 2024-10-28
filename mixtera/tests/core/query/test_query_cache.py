import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from mixtera.core.datacollection.mixtera_data_collection import MixteraDataCollection
from mixtera.core.query.query import Query
from mixtera.core.query.query_cache import QueryCache


class MockResult:
    def __init__(self):
        self._lock = 42
        self._index = 1337
        self._id = "test"

    def to_cache(self, path):
        # Simulate the caching process
        pass

    @classmethod
    def from_cache(cls, path):
        del path
        # Simulate loading from cache
        instance = cls()
        return instance


class TestQueryCache(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        self.directory = Path(self.temp_dir.name)
        self.mdc = MixteraDataCollection(self.directory)
        self.query_cache = QueryCache(self.directory, self.mdc)
        self.query = Query("SELECT * FROM table")
        self.query.results = MockResult()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_init(self):
        self.assertTrue(self.query_cache.enabled)
        self.assertEqual(self.query_cache.directory, self.directory)

    def test_get_query_hash(self):
        query_hash = self.query_cache._get_query_hash(self.query)
        expected_hash = hashlib.sha256(str(self.query).encode()).hexdigest()
        self.assertEqual(query_hash, expected_hash)

    def test_cache_query_enabled(self):
        self.query_cache.enabled = True
        with patch.object(self.query.results, "to_cache") as mock_to_cache:
            self.query_cache.cache_query(self.query)
            mock_to_cache.assert_called_once()

            hash_dir = self.directory / self.query_cache._get_query_hash(self.query)
            existing_dirs = sorted([int(d.name) for d in hash_dir.iterdir() if d.is_dir() and d.name.isdigit()])
            self.assertTrue(existing_dirs)
            last_dir = hash_dir / str(existing_dirs[-1])
            self.assertTrue(last_dir.exists())
            self.assertTrue((last_dir / "meta.pkl").exists())

    def test_cache_query_disabled(self):
        self.query_cache.enabled = False
        with patch.object(self.query.results, "to_cache") as mock_to_cache:
            self.query_cache.cache_query(self.query)
            mock_to_cache.assert_not_called()

    @patch("mixtera.core.query.query_result.QueryResult.from_cache")
    def test_get_queryresults_if_cached_found(self, mock_from_cache):
        # Simulate that from_cache returns our MockResult
        mock_from_cache.return_value = self.query.results

        self.query.results._id = "specialtest"
        self.query_cache.cache_query(self.query)
        result = self.query_cache.get_queryresults_if_cached(self.query)
        self.assertIsInstance(result, MockResult)
        self.assertEqual(result._id, "specialtest")

    def test_get_queryresults_if_cached_not_found(self):
        result = self.query_cache.get_queryresults_if_cached(self.query)
        self.assertIsNone(result)

    @patch("mixtera.core.query.query_result.QueryResult.from_cache")
    def test_get_queryresults_if_cached_outdated(self, mock_from_cache):
        self.query_cache.cache_query(self.query)
        # Simulate database version change
        self.mdc.get_db_version = MagicMock(return_value="new_version")
        result = self.query_cache.get_queryresults_if_cached(self.query)
        self.assertIsNone(result)
        mock_from_cache.assert_not_called()

    @patch("shutil.rmtree")
    @patch("mixtera.core.query.query_result.QueryResult.from_cache")
    def test_get_queryresults_if_cached_cleanup(self, mock_from_cache, mock_rmtree):
        del mock_from_cache
        self.query_cache.cache_query(self.query)
        # Simulate database version change
        self.mdc.get_db_version = MagicMock(return_value="new_version")
        result = self.query_cache.get_queryresults_if_cached(self.query)
        self.assertIsNone(result)
        # Check that shutil.rmtree is called to clean up the outdated cache
        hash_dir = self.directory / self.query_cache._get_query_hash(self.query)
        existing_dirs = sorted([int(d.name) for d in hash_dir.iterdir() if d.is_dir() and d.name.isdigit()])
        if existing_dirs:
            last_dir = hash_dir / str(existing_dirs[-1])
            mock_rmtree.assert_any_call(last_dir)
        else:
            mock_rmtree.assert_not_called()
