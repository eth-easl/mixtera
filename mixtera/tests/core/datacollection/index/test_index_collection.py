import unittest

from mixtera.core.datacollection.index.index_collection import IndexFactory, IndexTypes, InMemoryDictionaryRangeIndex


class TestInMemoryDictionaryIndex(unittest.TestCase):
    def test_get_full_index(self):
        index = IndexFactory.create_index(IndexTypes.IN_MEMORY_DICT_LINES)
        target_index = {
            "language": {
                "C": {0: {0: [0, 2, 4, 9]}},
                "PHP": {0: {0: [1]}},
            },
            "publication_date": {"val1": {0: {0: [0, 2, 3, 4, 5, 9, 10]}}},
        }

        index._index = target_index.copy()
        self.assertEqual(index.get_full_index(), target_index)

    def test_get_by_feature(self):
        index = IndexFactory.create_index(IndexTypes.IN_MEMORY_DICT_LINES)
        target_index = {
            "language": {
                "C": {0: {0: [0]}},
                "PHP": {0: {0: [1]}},
            },
            "publication_date": {"val1": {0: {0: [2]}}},
        }
        language_sub_dict = target_index["language"].copy()
        pub_date_sub_dict = target_index["publication_date"].copy()

        index._index = target_index.copy()
        self.assertEqual(index.get_by_feature("language"), language_sub_dict)
        self.assertEqual(index.get_by_feature("publication_date"), pub_date_sub_dict)
        self.assertEqual(index.get_by_feature("non_existent"), {})

    def test_get_by_feature_value(self):
        index = IndexFactory.create_index(IndexTypes.IN_MEMORY_DICT_LINES)
        target_index = {
            "language": {
                "C": {0: {0: [0]}},
                "PHP": {0: {0: [1]}},
            },
            "publication_date": {"val1": {0: {0: [2]}}},
        }
        c_sub_dict = target_index["language"]["C"].copy()
        val1_sub_dict = target_index["publication_date"]["val1"].copy()

        index._index = target_index.copy()
        self.assertEqual(index.get_by_feature_value("language", "C"), c_sub_dict)
        self.assertEqual(index.get_by_feature_value("publication_date", "val1"), val1_sub_dict)
        self.assertEqual(index.get_by_feature_value("non_existent", "non_existent"), {})
        self.assertEqual(index.get_by_feature_value("language", "non_existent"), {})

    def test_get_all_features(self):
        index = IndexFactory.create_index(IndexTypes.IN_MEMORY_DICT_LINES)
        index._index = {
            "language": {
                "C": {0: {0: [0, 2, 4, 9]}},
                "PHP": {0: {0: [1]}},
            },
            "publication_date": {"val1": {0: {0: [0, 2, 3, 4, 5, 9, 10]}}},
        }
        expected = ["language", "publication_date"]

        self.assertEqual(expected, index.get_all_features())

    def test_has_feature(self):
        index = IndexFactory.create_index(IndexTypes.IN_MEMORY_DICT_LINES)
        index._index = {
            "language": {},
            "publication_date": {},
        }
        self.assertTrue(index.has_feature("language"))
        self.assertTrue(index.has_feature("publication_date"))
        self.assertFalse(index.has_feature("non_existent"))

    def test_keep_only_feature(self):
        index = IndexFactory.create_index(IndexTypes.IN_MEMORY_DICT_LINES)
        index._index = {
            "language": {},
            "publication_date": {},
            "publication_venue": {},
        }
        index.keep_only_feature(["language", "publication_date"])
        self.assertEqual(index.get_full_index(), {"language": {}, "publication_date": {}})

        index.keep_only_feature("language")
        self.assertEqual(index.get_full_index(), {"language": {}})


class TestInMemoryDictionaryLineIndex(unittest.TestCase):

    def test_compress(self):
        index = IndexFactory.create_index(IndexTypes.IN_MEMORY_DICT_LINES)

        index._index = {
            "language": {
                "C": {0: {0: [0, 2, 4, 9]}},
                "PHP": {0: {0: [1]}},
            },
            "publication_date": {"val1": {0: {0: [0, 2, 3, 4, 5, 9, 10]}}},
        }

        target_index = {
            "language": {
                "C": {0: {0: [(0, 1), (2, 3), (4, 5), (9, 10)]}},
                "PHP": {0: {0: [(1, 2)]}},
            },
            "publication_date": {"val1": {0: {0: [(0, 1), (2, 6), (9, 11)]}}},
        }

        compressed_index = index.compress()
        self.assertIsInstance(compressed_index, InMemoryDictionaryRangeIndex)
        self.assertEqual(target_index, compressed_index.get_full_index())
        self.assertFalse(index.is_compressed)
        self.assertTrue(compressed_index.is_compressed)

    def test_append_index_entry(self):
        index = IndexFactory.create_index(IndexTypes.IN_MEMORY_DICT_LINES)
        base_index = {
            "language": {
                "C": {0: {0: [0, 2, 4, 9]}},
                "PHP": {0: {0: [1]}},
            },
            "publication_date": {"val1": {0: {0: [0, 2, 3, 4, 5, 9, 10]}}},
        }

        for feature, feature_entries in base_index.items():
            for value, value_entries in feature_entries.items():
                for dataset_id, dataset_id_entries in value_entries.items():
                    for file_id, file_id_entries in dataset_id_entries.items():
                        for line_number in file_id_entries:
                            index.append_entry(feature, value, dataset_id, file_id, line_number)

        self.assertEqual(index.get_full_index(), base_index)

    def test_merge(self):
        index1 = IndexFactory.create_index(IndexTypes.IN_MEMORY_DICT_LINES)
        index2 = IndexFactory.create_index(IndexTypes.IN_MEMORY_DICT_LINES)

        target_index = {
            "language": {
                "C": {0: {0: [0, 1], 1: [0, 1]}},
                "C++": {0: {0: [0, 1], 2: [3, 4]}},
                "CoffeeScript": {0: {0: [0]}, 1: {0: [0]}},
                "PHP": {0: {0: [1]}},
                "Java": {0: {0: [0]}},
            },
            "publication_date": {"val1": {0: {0: [0], 1: [0, 1]}}},
        }

        index1._index = {
            "language": {
                "C": {0: {0: [0, 1]}},
                "C++": {0: {0: [0, 1]}},
                "CoffeeScript": {0: {0: [0]}},
                "PHP": {0: {0: [1]}},
            },
            "publication_date": {"val1": {0: {0: [0]}}},
        }

        index2._index = {
            "language": {
                "C": {0: {1: [0, 1]}},
                "C++": {0: {2: [3, 4]}},
                "CoffeeScript": {1: {0: [0]}},
                "Java": {0: {0: [0]}},
            },
            "publication_date": {"val1": {0: {1: [0, 1]}}},
        }

        index1.merge(index2)
        self.assertEqual(index1.get_full_index(), target_index)

    def test_invalid_merge(self):
        index1 = IndexFactory.create_index(IndexTypes.IN_MEMORY_DICT_LINES)
        index2 = IndexFactory.create_index(IndexTypes.IN_MEMORY_DICT_RANGE)

        self.assertRaises(AssertionError, index1.merge, index2)


class TestInMemoryDictionaryRangeIndex(unittest.TestCase):
    def test_append_index_entry(self):
        index = IndexFactory.create_index(IndexTypes.IN_MEMORY_DICT_RANGE)
        base_index = {
            "language": {
                "C": {0: {0: [(0, 1), (2, 3), (4, 5), (9, 10)]}},
                "PHP": {0: {0: [(1, 2)]}},
            },
            "publication_date": {"val1": {0: {0: [(0, 1), (2, 6), (9, 11)]}}},
        }

        for feature, feature_entries in base_index.items():
            for value, value_entries in feature_entries.items():
                for dataset_id, dataset_id_entries in value_entries.items():
                    for file_id, file_id_entries in dataset_id_entries.items():
                        for line_range in file_id_entries:
                            index.append_entry(feature, value, dataset_id, file_id, line_range)

        self.assertEqual(index.get_full_index(), base_index)

    def test_merge(self):
        index1 = IndexFactory.create_index(IndexTypes.IN_MEMORY_DICT_RANGE)
        index2 = IndexFactory.create_index(IndexTypes.IN_MEMORY_DICT_RANGE)

        target_index = {
            "language": {
                "C": {0: {0: [(0, 2)], 1: [(0, 2)]}},
                "C++": {0: {0: [(0, 2)], 2: [(3, 5)]}},
                "CoffeeScript": {0: {0: [(0, 1)]}, 1: {0: [(0, 1)]}},
                "PHP": {0: {0: [(1, 2)]}},
                "Java": {0: {0: [(0, 1)]}},
            },
            "publication_date": {"val1": {0: {0: [(0, 1)], 1: [(0, 2)]}}},
        }

        index1._index = {
            "language": {
                "C": {0: {0: [(0, 2)]}},
                "C++": {0: {0: [(0, 2)]}},
                "CoffeeScript": {0: {0: [(0, 1)]}},
                "PHP": {0: {0: [(1, 2)]}},
            },
            "publication_date": {"val1": {0: {0: [(0, 1)]}}},
        }

        index2._index = {
            "language": {
                "C": {0: {1: [(0, 2)]}},
                "C++": {0: {2: [(3, 5)]}},
                "CoffeeScript": {1: {0: [(0, 1)]}},
                "Java": {0: {0: [(0, 1)]}},
            },
            "publication_date": {"val1": {0: {1: [(0, 2)]}}},
        }

        index1.merge(index2)
        self.assertEqual(index1.get_full_index(), target_index)
