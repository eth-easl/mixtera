import unittest
from unittest.mock import MagicMock, patch

from mixtera.core.client import MixteraClient
from mixtera.core.datacollection.index.index_collection import IndexFactory, IndexTypes
from mixtera.core.query import Operator, Query, QueryPlan
from mixtera.utils import defaultdict_to_dict


class MockOperator(Operator):
    def __init__(self, name, len_results: int = 1):
        super().__init__()
        self.name = name
        self.len_results = len_results

    def display(self, level):
        print("-" * level + self.name)

    def execute(self, mdc):
        del mdc
        self.results = IndexFactory.create_index(IndexTypes.IN_MEMORY_DICT_RANGE)
        self.results.append_entry("field", "value", "did", "fid", (0, 2))


class ComplexMockOperator(Operator):
    def __init__(self, name, len_results: int = 1):
        super().__init__()
        self.name = name
        self.len_results = len_results

    def display(self, level):
        print("-" * level + self.name)

    def execute(self, mdc):
        del mdc
        self.results = IndexFactory.create_index(IndexTypes.IN_MEMORY_DICT_RANGE)
        self.results._index = {
            "language": {
                "french": {
                    0: {
                        0: [(0, 50), (100, 150), (200, 300)],
                        1: [(100, 350)],
                        2: [],  # This should never be the case, but good to test
                    },
                    1: {0: [(25, 50), (60, 100)], 1: [(0, 50)]},
                },
                "english": {
                    0: {0: [(25, 75), (140, 210), (300, 400)], 1: [(50, 150)], 2: [(10, 20)]},
                    1: {0: [(50, 60), (90, 110), (130, 150)]},
                    2: {0: [(0, 100), (150, 200)]},
                },
            },
            "topic": {
                "law": {
                    0: {
                        0: [(0, 50), (125, 180)],
                        1: [(100, 150), (200, 250)],
                    },
                    1: {0: [(0, 40), (50, 75), (200, 250)], 1: [(20, 30)]},
                    2: {0: [(80, 100)]},
                },
                "medicine": {
                    0: {
                        0: [(25, 75), (80, 100), (120, 200)],
                        1: [(50, 150), (160, 170), (200, 210)],
                    },
                    1: {0: [(50, 60), (90, 110), (130, 150)], 1: [(30, 100), (150, 200)]},
                },
            },
        }


Query.register(MockOperator)
Query.register(ComplexMockOperator)


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
        self.client = MixteraClient.from_directory(".")
        self.query = Query("job_id")

    def test_init(self):
        self.assertIsInstance(self.query.query_plan, QueryPlan)

    def test_register(self):
        class TestOperator(Operator):
            def execute(self, mdc) -> None:
                del mdc
                self.results = ["test"]

        Query.register(TestOperator)
        self.assertTrue(hasattr(Query, "testoperator"))

    def test_for_training(self):
        query = Query.for_job("job_id")
        self.assertEqual(query.job_id, "job_id")

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

    @patch("mixtera.core.datacollection.MixteraDataCollection._get_dataset_func_by_id")
    @patch("mixtera.core.datacollection.MixteraDataCollection._get_dataset_type_by_id")
    @patch("mixtera.core.datacollection.MixteraDataCollection._get_file_path_by_id")
    def test_execute_chunksize_one(
        self,
        mock_get_file_path_by_id: MagicMock,
        mock_get_dataset_type_by_id: MagicMock,
        mock_get_dataset_func_by_id: MagicMock,
    ):
        mock_get_file_path_by_id.return_value = "test_file_path"
        mock_get_dataset_type_by_id.return_value = "test_dataset_type"
        mock_get_dataset_func_by_id.return_value = lambda x: x

        query = Query("job_id").mockoperator("test")
        assert self.client.execute_query(query, 1)
        query_result = query.results
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

    @patch("mixtera.core.datacollection.MixteraDataCollection._get_dataset_func_by_id")
    @patch("mixtera.core.datacollection.MixteraDataCollection._get_dataset_type_by_id")
    @patch("mixtera.core.datacollection.MixteraDataCollection._get_file_path_by_id")
    def test_execute_chunksize_two(
        self,
        mock_get_file_path_by_id: MagicMock,
        mock_get_dataset_type_by_id: MagicMock,
        mock_get_dataset_func_by_id: MagicMock,
    ):
        mock_get_file_path_by_id.return_value = "test_file_path"
        mock_get_dataset_type_by_id.return_value = "test_dataset_type"
        mock_get_dataset_func_by_id.return_value = lambda x: x

        query = Query.for_job("job_id").mockoperator("test", len_results=2)
        assert self.client.execute_query(query, 2)
        res = list(query.results)
        res = [x._index for x in res]
        self.assertEqual(res, [{"field": {"value": {"did": {"fid": [(0, 2)]}}}}])

    @patch("mixtera.core.datacollection.MixteraDataCollection._get_dataset_func_by_id")
    @patch("mixtera.core.datacollection.MixteraDataCollection._get_dataset_type_by_id")
    @patch("mixtera.core.datacollection.MixteraDataCollection._get_file_path_by_id")
    def test_create_chunker_index(
        self,
        mock_get_file_path_by_id: MagicMock,
        mock_get_dataset_type_by_id: MagicMock,
        mock_get_dataset_func_by_id: MagicMock,
    ):
        mock_get_file_path_by_id.return_value = "test_file_path"
        mock_get_dataset_type_by_id.return_value = "test_dataset_type"
        mock_get_dataset_func_by_id.return_value = lambda x: x

        reference_result = {
            "language:french;topic:law": {
                0: {0: [[0, 25]], 1: [[210, 250]]},
                1: {0: [[25, 40], [60, 75]], 1: [[20, 30]]},
            },
            "language:french,english;topic:law,medicine": {0: {0: [[25, 50], [140, 150]], 1: [[100, 150]]}},
            "language:english;topic:medicine": {
                0: {0: [[50, 75], [180, 200]], 1: [[50, 100]]},
                1: {0: [[100, 110], [130, 150]]},
            },
            "topic:medicine": {0: {0: [[80, 100]]}, 1: {1: [[50, 100], [150, 200]]}},
            "language:french": {
                0: {0: [[100, 120], [210, 300]], 1: [[150, 160], [170, 200], [250, 350]]},
                1: {0: [[40, 50], [75, 90]], 1: [[0, 20]]},
            },
            "language:french;topic:medicine": {0: {0: [[120, 125]], 1: [[160, 170]]}, 1: {1: [[30, 50]]}},
            "language:french;topic:law,medicine": {0: {0: [[125, 140]], 1: [[200, 210]]}},
            "language:english;topic:law,medicine": {0: {0: [[150, 180]]}, 1: {0: [[50, 60]]}},
            "language:french,english": {0: {0: [[200, 210]]}},
            "language:english": {0: {0: [[300, 400]], 2: [[10, 20]]}, 2: {0: [[0, 80], [150, 200]]}},
            "topic:law": {1: {0: [[0, 25], [200, 250]]}},
            "language:french,english;topic:medicine": {1: {0: [[90, 100]]}},
            "language:english;topic:law": {2: {0: [[80, 100]]}},
        }

        query = Query.for_job("job_id").complexmockoperator("test")
        assert self.client.execute_query(query, 1)
        inverted_index = query.results._invert_result(query.results.results._index)
        chunk_index = query.results._create_chunker_index(inverted_index)
        self.assertDictEqual(defaultdict_to_dict(chunk_index), reference_result)
