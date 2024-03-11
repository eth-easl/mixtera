import unittest

from mixtera.core.datacollection import MixteraDataCollection
from mixtera.core.datacollection.index.index_collection import IndexFactory, IndexTypes
from mixtera.core.query.operators.union import Union
from mixtera.core.query.query import Query


class TestUnion(unittest.TestCase):
    def setUp(self):
        self.mdc = MixteraDataCollection.from_directory(".")
        self.query_a = Query.from_datacollection(self.mdc).select(("field1", "==", "value1"))
        self.query_a.root.results = IndexFactory.create_index(IndexTypes.IN_MEMORY_DICT_RANGE)

        self.query_a.root.results.append_entry("field1", "value1", "did", "fid", (0, 2))
        self.union = Union(self.query_a)

    def test_init(self):
        self.assertEqual(len(self.union.children), 1)
        self.assertEqual(self.union.children[0], self.query_a.root)

    def test_execute(self):
        query_b = Query.from_datacollection(self.mdc).select(("field1", "==", "value2"))
        query_b.root.results = IndexFactory.create_index(IndexTypes.IN_MEMORY_DICT_RANGE)
        query_b.root.results.append_entry("field1", "value2", "did", "fid", (0, 2))
        gt_result = {"field1": {"value1": {"did": {"fid": (0, 2)}}, "value2": {"did": {"fid": (0, 2)}}}}
        self.union.children.append(query_b.root)
        self.union.execute()
        self.assertTrue(self.union.results._index, gt_result)

    def test_repr(self):
        self.assertEqual(str(self.union), "union<>()")
