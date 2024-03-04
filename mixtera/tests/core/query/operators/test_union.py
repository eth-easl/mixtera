import unittest

from mixtera.core.datacollection import MixteraDataCollection
from mixtera.core.query.operators.union import Union
from mixtera.core.query.query import Query


class TestUnion(unittest.TestCase):
    def setUp(self):
        self.mdc = MixteraDataCollection.from_directory(".")
        self.query_a = Query.from_datacollection(self.mdc).select(("field1", "==", "value1"))
        self.query_a.root.results = ["result1", "result2", "result3"]
        self.union = Union(self.query_a)

    def test_init(self):
        self.assertEqual(len(self.union.children), 1)
        self.assertEqual(self.union.children[0], self.query_a.root)

    def test_execute(self):
        query_b = Query.from_datacollection(self.mdc).select(("field1", "==", "value2"))
        query_b.root.results = ["result3", "result4", "result5"]
        self.union.children.append(query_b.root)
        self.union.execute()
        self.assertEqual(len(self.union.results), 6)
        self.assertIn("result1", self.union.results)
        self.assertIn("result2", self.union.results)
        self.assertIn("result3", self.union.results)
        self.assertIn("result4", self.union.results)
        self.assertIn("result5", self.union.results)

    def test_repr(self):
        self.assertEqual(repr(self.union), "union<>()")
