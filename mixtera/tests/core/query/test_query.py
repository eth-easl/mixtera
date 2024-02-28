import unittest
from unittest.mock import MagicMock, patch

from mixtera.core.datacollection import MixteraDataCollection
from mixtera.core.query import Operator, Query, QueryPlan


class MockOperator(Operator):
    def __init__(self, name, len_results: int = 1):
        super().__init__()
        self.name = name
        self.len_results = len_results

    def insert(self, root):
        return self if root is None else root

    def display(self, level):
        print(" " * level + self.name)

    def apply(self):
        self.results = [{1: {1: [(1, 2)]}}] * self.len_results


Query.register(MockOperator)


class TestQueryPlan(unittest.TestCase):
    def setUp(self):
        self.query_plan = QueryPlan()

    def test_init(self):
        self.assertIsNone(self.query_plan.root)

    def test_add(self):
        operator = MockOperator("test_operator")
        self.query_plan.add(operator)
        self.assertEqual(self.query_plan.root, operator)

    def test_display(self):
        operator = MockOperator("test_operator")
        self.query_plan.add(operator)
        with unittest.mock.patch("builtins.print") as mock_print:
            self.query_plan.display()
            mock_print.assert_called_once_with("test_operator")


class TestQuery(unittest.TestCase):
    def setUp(self):
        self.mdc = MixteraDataCollection.from_directory(".")
        self.query = Query(self.mdc)

    def test_init(self):
        self.assertEqual(self.query.mdc, self.mdc)
        self.assertIsInstance(self.query.query_plan, QueryPlan)

    def test_register(self):
        class TestOperator(Operator):
            def apply(self) -> None:
                self.results = ["test"]

        Query.register(TestOperator)
        self.assertTrue(hasattr(Query, "testoperator"))

    def test_from_datacollection(self):
        query = Query.from_datacollection(self.mdc)
        self.assertEqual(query.mdc, self.mdc)
        self.assertIsInstance(query.query_plan, QueryPlan)

    def test_root(self):
        operator = MockOperator("test_operator")
        self.query.query_plan.add(operator)
        self.assertEqual(self.query.root, operator)

    def test_display(self):
        operator = MockOperator("test_operator")
        self.query.query_plan.add(operator)
        with unittest.mock.patch("builtins.print") as mock_print:
            self.query.display()
            mock_print.assert_called_once_with("test_operator")

    @patch("mixtera.core.datacollection.local.LocalDataCollection._get_dataset_type_by_id")
    @patch("mixtera.core.datacollection.local.LocalDataCollection._get_file_path_by_id")
    def test_execute_chunksize_one(self, mock_get_file_path_by_id: MagicMock, mock_get_dataset_type_by_id: MagicMock):
        mock_get_file_path_by_id.return_value = "test_file_path"
        mock_get_dataset_type_by_id.return_value = "test_dataset_type"

        query = Query(self.mdc).mockoperator("test")
        query_result = query.execute(chunk_size=1)
        res = list(query_result)
        gt_meta = {
            "dataset_type": {1: "test_dataset_type"},
            "file_path": {1: "test_file_path"},
        }
        print(gt_meta)
        self.assertEqual(res, [[{1: {1: [(1, 2)]}}]])
        self.assertEqual(query_result.dataset_type, gt_meta["dataset_type"])
        self.assertEqual(query_result.file_path, gt_meta["file_path"])

    @patch("mixtera.core.datacollection.local.LocalDataCollection._get_file_path_by_id")
    @patch("mixtera.core.datacollection.local.LocalDataCollection._get_dataset_type_by_id")
    def test_execute_chunksize_two(self, mock_get_file_path_by_id: MagicMock, mock_get_dataset_type_by_id: MagicMock):
        mock_get_file_path_by_id.return_value = "test_file_path"
        mock_get_dataset_type_by_id.return_value = "test_dataset_type"
        query = Query(self.mdc).mockoperator("test", len_results=2)
        res = query.execute(chunk_size=2)
        res = list(res)
        self.assertEqual(res, [[{1: {1: [(1, 2)]}}, {1: {1: [(1, 2)]}}]])
