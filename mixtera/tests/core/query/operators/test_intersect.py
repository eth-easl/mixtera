import unittest

from mixtera.core.client import MixteraClient
from mixtera.core.datacollection.index.index_collection import IndexFactory, IndexTypes
from mixtera.core.query.operators.intersect import Intersection
from mixtera.core.query.query import Query
from mixtera.utils import defaultdict_to_dict


class TestIntersection(unittest.TestCase):
    def setUp(self):
        self.client = MixteraClient.from_directory(".")
        self.query_a = Query.for_job("job_id").select(("field1", "==", "value1"))
        self.query_a.root.results = IndexFactory.create_index(IndexTypes.IN_MEMORY_DICT_RANGE)

        self.query_a.root.results.append_entry("field1", "value1", "did", "fid", (0, 2))
        self.intersection = Intersection(self.query_a)

    def test_init(self):
        self.query_a.display()
        self.assertEqual(len(self.intersection.children), 1)
        self.assertEqual(self.intersection.children[0], self.query_a.root)

    def test_execute_overlapping_field(self):
        query_b = Query.for_job("job_id").select(("field1", "==", "value2"))
        query_b.root.results = IndexFactory.create_index(IndexTypes.IN_MEMORY_DICT_RANGE)
        query_b.root.results.append_entry("field1", "value2", "did", "fid", (0, 2))
        gt_result = {"field1": {}}
        self.intersection.children.append(query_b.root)
        self.intersection.execute(self.client)
        self.assertDictEqual(defaultdict_to_dict(self.intersection.results._index), gt_result)

    def test_execute_no_overlapping_field(self):
        query_b = Query.for_job("job_id").select(("field2", "==", "value2"))
        query_b.root.results = IndexFactory.create_index(IndexTypes.IN_MEMORY_DICT_RANGE)
        query_b.root.results.append_entry("field2", "value2", "did", "fid", (0, 2))
        gt_result = {}
        self.intersection.children.append(query_b.root)
        self.intersection.execute(self.client)
        self.assertDictEqual(defaultdict_to_dict(self.intersection.results._index), gt_result)

    def test_execute_no_overlapping_ranges(self):
        query_b = Query.for_job("job_id").select(("field1", "==", "value1"))
        query_b.root.results = IndexFactory.create_index(IndexTypes.IN_MEMORY_DICT_RANGE)
        query_b.root.results.append_entry("field1", "value1", "did", "fid", (3, 4))
        gt_result = {"field1": {"value1": {"did": {"fid": []}}}}
        self.intersection.children.append(query_b.root)
        self.intersection.execute(self.client)
        self.assertDictEqual(defaultdict_to_dict(self.intersection.results._index), gt_result)

    def test_execute_overlapping_ranges(self):
        query_b = Query.for_job("job_id").select(("field1", "==", "value1"))
        query_b.root.results = IndexFactory.create_index(IndexTypes.IN_MEMORY_DICT_RANGE)
        query_b.root.results.append_entry("field1", "value1", "did", "fid", (1, 4))
        gt_result = {"field1": {"value1": {"did": {"fid": [(1, 2)]}}}}
        self.intersection.children.append(query_b.root)
        self.intersection.execute(self.client)
        self.assertDictEqual(defaultdict_to_dict(self.intersection.results._index), gt_result)

    def test_execute_with_incorrect_children(self):
        with self.assertRaises(AssertionError):
            self.intersection.execute(None)

    def test_repr(self):
        self.assertEqual(str(self.intersection), "intersection<>()")


if __name__ == "__main__":
    unittest.main()
