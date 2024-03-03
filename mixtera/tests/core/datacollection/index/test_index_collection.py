import unittest

from mixtera.core.datacollection.index import InMemoryDictionaryIndex


class TestInMemoryDictionaryIndex(unittest.TestCase):

    def test_compress(self):
        index = InMemoryDictionaryIndex()

        index._index = {
            "language": {
                "C": {0: {0: [0, 2, 4, 9]}},
                "PHP": {0: {0: [1]}},
            },
            "publication_date": {"asd123": {0: {0: [0, 2, 3, 4, 5, 9, 10]}}},
        }

        target_index = {
            "language": {
                "C": {0: {0: [(0, 1), (2, 3), (4, 5), (9, 10)]}},
                "PHP": {0: {0: [(1, 2)]}},
            },
            "publication_date": {"asd123": {0: {0: [(0, 1), (2, 6), (9, 11)]}}},
        }

        index.compress()
        self.assertEqual(target_index, index.get_full_index())

    def test_get_all_features(self):
        index = InMemoryDictionaryIndex()
        index._index = {
            "language": {
                "C": {0: {0: [0, 2, 4, 9]}},
                "PHP": {0: {0: [1]}},
            },
            "publication_date": {"asd123": {0: {0: [0, 2, 3, 4, 5, 9, 10]}}},
        }
        expected = ["language", "publication_date"]

        index.compress()
        self.assertEqual(expected, index.get_all_features())

    def test_append_index_entry(self):
        index = InMemoryDictionaryIndex()
        base_index = {
            "language": {
                "C": {0: {0: [0, 2, 4, 9]}},
                "PHP": {0: {0: [1]}},
            },
            "publication_date": {"asd123": {0: {0: [0, 2, 3, 4, 5, 9, 10]}}},
        }

        target_index = {
            "language": {
                "C": {0: {0: [(0, 1), (2, 3), (4, 5), (9, 10)]}},
                "PHP": {0: {0: [(1, 2)]}},
            },
            "publication_date": {"asd123": {0: {0: [(0, 1), (2, 6), (9, 11)]}}},
        }

        for feature, feature_entries in base_index.items():
            for value, value_entries in feature_entries.items():
                for dataset_id, dataset_id_entries in value_entries.items():
                    for file_id, file_id_entries in dataset_id_entries.items():
                        for line_number in file_id_entries:
                            index.append_index_entry(feature, value, dataset_id, file_id, line_number)

        index.compress()
        self.assertEqual(index.get_full_index(), target_index)

    def test_merge(self):
        index1 = InMemoryDictionaryIndex()
        index2 = InMemoryDictionaryIndex()

        target_index = {
            "language": {
                "C": {0: {0: [(0, 2)], 1: [(0, 2)]}},
                "C++": {0: {0: [(0, 2)], 2: [(3, 5)]}},
                "CoffeeScript": {0: {0: [(0, 1)]}, 1: {0: [(0, 1)]}},
                "PHP": {0: {0: [(1, 2)]}},
                "Java": {0: {0: [(0, 1)]}},
            },
            "publication_date": {"asd123": {0: {0: [(0, 1)], 1: [(0, 2)]}}},
        }

        index1._index = {
            "language": {
                "C": {0: {0: [(0, 2)]}},  # value with document and list of lines
                "C++": {0: {0: [(0, 2)]}},
                "CoffeeScript": {0: {0: [(0, 1)]}},
                "PHP": {0: {0: [(1, 2)]}},
            },
            "publication_date": {"asd123": {0: {0: [(0, 1)]}}},
        }

        index2._index = {
            "language": {
                "C": {0: {1: [(0, 2)]}},  # value with document and list of lines
                "C++": {0: {2: [(3, 5)]}},
                "CoffeeScript": {1: {0: [(0, 1)]}},
                "Java": {0: {0: [(0, 1)]}},
            },
            "publication_date": {"asd123": {0: {1: [(0, 2)]}}},
        }

        index1._is_compressed = True
        index2._is_compressed = True

        index1.merge(index2)
        print(index1.get_full_index())

        self.assertEqual(target_index, index1.get_full_index())
