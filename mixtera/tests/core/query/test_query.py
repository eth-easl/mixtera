import unittest
from unittest.mock import MagicMock, patch

from mixtera.core.datacollection import MixteraDataCollection
from mixtera.core.datacollection.index.index_collection import IndexFactory, IndexTypes
from mixtera.core.query import Operator, Query, QueryPlan


class MockOperator(Operator):
    def __init__(self, name, len_results: int = 1):
        super().__init__()
        self.name = name
        self.len_results = len_results

    def display(self, level):
        print("-" * level + self.name)

    def execute(self):
        self.results = IndexFactory.create_index(IndexTypes.IN_MEMORY_DICT_RANGE)
        self.results.append_entry("field", "value", "did", "fid", (0, 2))


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
            def execute(self) -> None:
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

    @patch("mixtera.core.datacollection.local.LocalDataCollection._get_dataset_func_by_id")
    @patch("mixtera.core.datacollection.local.LocalDataCollection._get_dataset_type_by_id")
    @patch("mixtera.core.datacollection.local.LocalDataCollection._get_file_path_by_id")
    def test_execute_chunksize_one(
        self,
        mock_get_file_path_by_id: MagicMock,
        mock_get_dataset_type_by_id: MagicMock,
        mock_get_dataset_func_by_id: MagicMock,
    ):
        mock_get_file_path_by_id.return_value = "test_file_path"
        mock_get_dataset_type_by_id.return_value = "test_dataset_type"
        mock_get_dataset_func_by_id.return_value = lambda x: x

        query = Query.from_datacollection(self.mdc).mockoperator("test")
        query_result = query.execute(chunk_size=1)
        res = list(query_result)
        res = [x._index for x in res]
        gt_meta = {
            "dataset_type": {"did": "test_dataset_type"},
            "file_path": {"fid": "test_file_path"},
        }

        self.assertEqual(
            res, [{"field": {"value": {"did": {"fid": [(0, 1)]}}}}, {"field": {"value": {"did": {"fid": [(1, 2)]}}}}]
        )

        self.assertEqual(query_result.dataset_type, gt_meta["dataset_type"])
        self.assertEqual(query_result.file_path, gt_meta["file_path"])

    @patch("mixtera.core.datacollection.local.LocalDataCollection._get_dataset_func_by_id")
    @patch("mixtera.core.datacollection.local.LocalDataCollection._get_dataset_type_by_id")
    @patch("mixtera.core.datacollection.local.LocalDataCollection._get_file_path_by_id")
    def test_execute_chunksize_two(
        self,
        mock_get_file_path_by_id: MagicMock,
        mock_get_dataset_type_by_id: MagicMock,
        mock_get_dataset_func_by_id: MagicMock,
    ):
        mock_get_file_path_by_id.return_value = "test_file_path"
        mock_get_dataset_type_by_id.return_value = "test_dataset_type"
        mock_get_dataset_func_by_id.return_value = lambda x: x

        query = Query(self.mdc).mockoperator("test", len_results=2)
        res = query.execute(chunk_size=2)
        res = list(res)
        res = [x._index for x in res]
        self.assertEqual(res, [{"field": {"value": {"did": {"fid": [(0, 2)]}}}}])
