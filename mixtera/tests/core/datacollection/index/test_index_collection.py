import unittest

from mixtera.core.datacollection.index.index_collection import (
    IndexFactory,
    IndexTypes,
    InMemoryDictionaryLineIndex,
    InMemoryDictionaryRangeIndex,
)


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
        self.assertEqual(index.get_full_dict_index(), target_index)

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
        self.assertEqual(index.get_dict_index_by_feature("language"), language_sub_dict)
        self.assertEqual(index.get_dict_index_by_feature("publication_date"), pub_date_sub_dict)
        self.assertEqual(index.get_dict_index_by_feature("non_existent"), {})

    def test_get_dict_index_by_predicate(self):
        index = IndexFactory.create_index(IndexTypes.IN_MEMORY_DICT_LINES)
        target_index = {
            "property_name": {
                -1: {0: {0: [0]}},
                1: {0: {0: [1]}},
                -10: {0: {0: [2]}},
                10: {0: {0: [3]}},
            },
            "other_property_name": {"val1": {0: {0: [2]}}},
        }
        index._index = target_index.copy()

        lt_0 = {
            -1: {0: {0: [0]}},
            -10: {0: {0: [2]}},
        }
        gt_0 = {
            1: {0: {0: [1]}},
            10: {0: {0: [3]}},
        }

        lt_100 = target_index["property_name"].copy()
        gt_100 = {}

        self.assertEqual(index.get_dict_index_by_predicate("property_name", lambda x: x < 0), lt_0)
        self.assertEqual(index.get_dict_index_by_predicate("property_name", lambda x: x > 0), gt_0)
        self.assertEqual(index.get_dict_index_by_predicate("property_name", lambda x: x > 100), gt_100)
        self.assertEqual(index.get_dict_index_by_predicate("property_name", lambda x: x < 100), lt_100)

    def test_get_index_by_predicate(self):
        index = IndexFactory.create_index(IndexTypes.IN_MEMORY_DICT_LINES)
        target_index = {
            "property_name": {
                -1: {0: {0: [0]}},
                1: {0: {0: [1]}},
                -10: {0: {0: [2]}},
                10: {0: {0: [3]}},
            },
            "other_property_name": {"val1": {0: {0: [2]}}},
        }
        index._index = target_index.copy()

        lt_0 = {
            "property_name": {
                -1: {0: {0: [0]}},
                -10: {0: {0: [2]}},
            }
        }
        gt_0 = {
            "property_name": {
                1: {0: {0: [1]}},
                10: {0: {0: [3]}},
            }
        }

        lt_100 = {"property_name": target_index["property_name"].copy()}
        gt_100 = {}

        lt_0_res = index.get_index_by_predicate("property_name", lambda x: x < 0)
        gt_0_res = index.get_index_by_predicate("property_name", lambda x: x > 0)
        gt_100_res = index.get_index_by_predicate("property_name", lambda x: x > 100)
        lt_100_res = index.get_index_by_predicate("property_name", lambda x: x < 100)

        self.assertIsInstance(lt_0_res, InMemoryDictionaryLineIndex)
        self.assertIsInstance(gt_0_res, InMemoryDictionaryLineIndex)
        self.assertIsInstance(gt_100_res, InMemoryDictionaryLineIndex)
        self.assertIsInstance(lt_100_res, InMemoryDictionaryLineIndex)

        self.assertEqual(lt_0_res._index, lt_0)
        self.assertEqual(gt_0_res._index, gt_0)
        self.assertEqual(gt_100_res._index, gt_100)
        self.assertEqual(lt_100_res._index, lt_100)

    def test_get_by_many_features(self):
        index = IndexFactory.create_index(IndexTypes.IN_MEMORY_DICT_LINES)
        target_index = {
            "language": {
                "C": {0: {0: [0]}},
                "PHP": {0: {0: [1]}},
            },
            "publication_date": {"val1": {0: {0: [2]}}},
            "publication_venue": {"venue1": {2: {3: [1, 4, 5]}}, "venue2": {1: {0: [9, 10]}}},
        }
        target = {
            "language": target_index["language"].copy(),
            "publication_date": target_index["publication_date"].copy(),
        }

        index._index = target_index.copy()
        self.assertEqual(index.get_dict_index_by_many_features("language"), {"language": target["language"]})
        self.assertEqual(index.get_dict_index_by_many_features(["language", "publication_date"]), target)
        self.assertEqual(index.get_dict_index_by_many_features("non_existent"), {})

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
        self.assertEqual(index.get_dict_index_by_feature_value("language", "C"), c_sub_dict)
        self.assertEqual(index.get_dict_index_by_feature_value("publication_date", "val1"), val1_sub_dict)
        self.assertEqual(index.get_dict_index_by_feature_value("non_existent", "non_existent"), {})
        self.assertEqual(index.get_dict_index_by_feature_value("language", "non_existent"), {})

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
        index.drop_other_features(["language", "publication_date"])
        self.assertEqual(index.get_full_dict_index(), {"language": {}, "publication_date": {}})

        index.drop_other_features("language")
        self.assertEqual(index.get_full_dict_index(), {"language": {}})


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
        self.assertEqual(target_index, compressed_index.get_full_dict_index())
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

        self.assertEqual(index.get_full_dict_index(), base_index)

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
        self.assertEqual(index1.get_full_dict_index(), target_index)

    def test_invalid_merge(self):
        index1 = IndexFactory.create_index(IndexTypes.IN_MEMORY_DICT_LINES)
        index2 = IndexFactory.create_index(IndexTypes.IN_MEMORY_DICT_RANGE)

        self.assertRaises(AssertionError, index1.merge, index2)

    def test_get_index_by_features(self):
        index = IndexFactory.create_index(IndexTypes.IN_MEMORY_DICT_LINES)

        base_index = {
            "language": {"C": {0: {0: [0, 1, 2, 6, 7]}}},
            "publication_date": {"val1": {0: {0: [10, 12, 14, 20, 21]}}},
        }
        index._index = base_index
        new_index = index.get_index_by_features("publication_date")
        self.assertIsInstance(new_index, InMemoryDictionaryLineIndex)
        self.assertEqual(new_index.get_full_dict_index(), {"publication_date": base_index["publication_date"]})

        empty_index = index.get_index_by_features(["non_existent"])
        self.assertIsInstance(empty_index, InMemoryDictionaryLineIndex)
        self.assertEqual(empty_index.get_full_dict_index(), {})

        # Assert the new indexes cannot be interpreted as the other class
        self.assertFalse(isinstance(new_index, InMemoryDictionaryRangeIndex))
        self.assertFalse(isinstance(empty_index, InMemoryDictionaryRangeIndex))


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

        self.assertEqual(index.get_full_dict_index(), base_index)

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
        self.assertEqual(index1.get_full_dict_index(), target_index)

    def test_get_index_by_features(self):
        index = IndexFactory.create_index(IndexTypes.IN_MEMORY_DICT_RANGE)

        base_index = {
            "language": {"C": {0: {0: [(0, 2)]}}},
            "publication_date": {"val1": {0: {0: [(0, 1)]}}},
        }
        index._index = base_index
        new_index = index.get_index_by_features("publication_date")
        self.assertIsInstance(new_index, InMemoryDictionaryRangeIndex)
        self.assertEqual(new_index.get_full_dict_index(), {"publication_date": base_index["publication_date"]})

        empty_index = index.get_index_by_features(["non_existent"])
        self.assertIsInstance(empty_index, InMemoryDictionaryRangeIndex)
        self.assertEqual(empty_index.get_full_dict_index(), {})

        # Assert the new indexes cannot be interpreted as the other class
        self.assertFalse(isinstance(new_index, InMemoryDictionaryLineIndex))
        self.assertFalse(isinstance(empty_index, InMemoryDictionaryLineIndex))
