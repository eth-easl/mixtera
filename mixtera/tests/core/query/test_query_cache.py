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

    @patch("dill.dump")
    def test_cache_query_enabled(self, mock_dump):
        self.query_cache.enabled = True
        self.query_cache.cache_query(self.query)
        hash_dir = self.directory / self.query_cache._get_query_hash(self.query)
        cache_path = hash_dir / "0.pkl"
        mock_dump.assert_called_once()
        self.assertTrue(cache_path.exists())

    def test_cache_query_disabled(self):
        self.query_cache.enabled = False
        with patch("dill.dump") as mock_dump:
            self.query_cache.cache_query(self.query)
            mock_dump.assert_not_called()

    def test_get_queryresults_if_cached_found(self):
        self.query.results._id = "specialtest"
        self.query_cache.cache_query(self.query)
        result = self.query_cache.get_queryresults_if_cached(self.query)
        self.assertIsInstance(result, MockResult)
        self.assertEqual(result._id, "specialtest")

    def test_get_queryresults_if_cached_not_found(self):
        result = self.query_cache.get_queryresults_if_cached(self.query)
        self.assertIsNone(result)

    def test_get_queryresults_if_cached_outdated(self):
        self.query_cache.cache_query(self.query)
        # Simulate database version change
        self.mdc.get_db_version = MagicMock(return_value="new_version")
        result = self.query_cache.get_queryresults_if_cached(self.query)
        self.assertIsNone(result)

    def test_get_queryresults_if_cached_cleanup(self):
        self.query_cache.cache_query(self.query)
        # Simulate database version change
        self.mdc.get_db_version = MagicMock(return_value="new_version")
        assert self.query_cache.get_queryresults_if_cached(self.query) is None
