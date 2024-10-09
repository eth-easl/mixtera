import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import portion as P
from mixtera.core.client import MixteraClient
from mixtera.core.client.mixtera_client import QueryExecutionArgs
from mixtera.core.datacollection.index.index_collection import IndexFactory, IndexTypes
from mixtera.core.query import ArbitraryMixture, MixtureKey, Operator, Query, QueryPlan, StaticMixture
from mixtera.core.query.mixture import InferringMixture
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


class SimpleMockOperator(Operator):
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
                "english": {
                    0: {0: [(50, 150)]},
                },
                "french": {
                    0: {
                        0: [(0, 100), (150, 200)],
                        1: [(0, 100)],
                    }
                },
            },
        }


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


class FlexibleChunkingTestMockOperator(Operator):
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
                "english": {
                    0: {0: [(0, 10)]},
                },
                "french": {
                    0: {
                        0: [(0, 5)],
                    }
                },
                "german": {
                    0: {
                        0: [(5, 10)],
                    }
                },
            },
        }


Query.register(MockOperator)
Query.register(SimpleMockOperator)
Query.register(ComplexMockOperator)
Query.register(FlexibleChunkingTestMockOperator)


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
        self.temp_dir = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        self.directory = Path(self.temp_dir.name)
        self.client = MixteraClient.from_directory(self.directory)
        self.query = Query("job_id")

    def tearDown(self):
        self.temp_dir.cleanup()

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
        args = QueryExecutionArgs(mixture=ArbitraryMixture(1))
        assert self.client.execute_query(query, args)
        query_result = query.results
        gt_meta = {
            "dataset_type": {"did": "test_dataset_type"},
            "file_path": {"fid": "test_file_path"},
        }

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
        args = QueryExecutionArgs(mixture=ArbitraryMixture(2))
        assert self.client.execute_query(query, args)
        chunks = list(iter(query.results))
        chunks = [chunk._result_index for chunk in chunks]
        self.assertEqual(chunks, [{MixtureKey({"field": ["value"]}): {"did": {"fid": [(0, 2)]}}}])

    @patch("mixtera.core.datacollection.MixteraDataCollection._get_dataset_func_by_id")
    @patch("mixtera.core.datacollection.MixteraDataCollection._get_dataset_type_by_id")
    @patch("mixtera.core.datacollection.MixteraDataCollection._get_file_path_by_id")
    def test_create_inverted_index_simple(
        self,
        mock_get_file_path_by_id: MagicMock,
        mock_get_dataset_type_by_id: MagicMock,
        mock_get_dataset_func_by_id: MagicMock,
    ):
        mock_get_file_path_by_id.return_value = "test_file_path"
        mock_get_dataset_type_by_id.return_value = "test_dataset_type"
        mock_get_dataset_func_by_id.return_value = lambda x: x

        reference_result = {
            0: {
                0: {
                    # File 0 in dataset 0 has 3 overlapping has 3 overlapping intervals from a properties perspective:
                    # [0, 50), [50, 100), [100, 150), [150, 200). We can see that:
                    # 1. [100, 150) is both english and french
                    # 2. [0, 50) and [150, 200) are only french
                    # 3. [100, 150) is only english
                    P.closedopen(0, 50) | P.closedopen(150, 200): {"language": ["french"]},
                    P.closedopen(50, 100): {"language": ["english", "french"]},
                    P.closedopen(100, 150): {"language": ["english"]},
                },
                1: {
                    # File 1 in dataset 0 only exists for the language:french combination, hence interval [0, 100) is
                    # only assigned to this property
                    P.closedopen(0, 100): {"language": ["french"]}
                },
            }
        }

        query = Query.for_job("job_id").simplemockoperator("test")
        args = QueryExecutionArgs(mixture=ArbitraryMixture(1))

        assert self.client.execute_query(query, args)
        inverted_index = query.results._invert_result(query.results.results)

        # True result vs reference
        for document_id, doc_entries in inverted_index.items():
            for file_id, file_entries in doc_entries.items():
                for intervals, properties in file_entries.items():
                    self.assertTrue(
                        document_id in reference_result
                        and file_id in reference_result[document_id]
                        and intervals in reference_result[document_id][file_id]
                    )
                    self.assertDictEqual(properties, reference_result[document_id][file_id][intervals])

        # Reference vs True result
        for document_id, doc_entries in reference_result.items():
            for file_id, file_entries in doc_entries.items():
                for intervals, properties in file_entries.items():
                    self.assertTrue(
                        document_id in inverted_index
                        and file_id in inverted_index[document_id]
                        and intervals in inverted_index[document_id][file_id]
                    )
                    self.assertDictEqual(properties, inverted_index[document_id][file_id][intervals].values()[0])

    @patch("mixtera.core.datacollection.MixteraDataCollection._get_dataset_func_by_id")
    @patch("mixtera.core.datacollection.MixteraDataCollection._get_dataset_type_by_id")
    @patch("mixtera.core.datacollection.MixteraDataCollection._get_file_path_by_id")
    def test_create_inverted_index(
        self,
        mock_get_file_path_by_id: MagicMock,
        mock_get_dataset_type_by_id: MagicMock,
        mock_get_dataset_func_by_id: MagicMock,
    ):
        mock_get_file_path_by_id.return_value = "test_file_path"
        mock_get_dataset_type_by_id.return_value = "test_dataset_type"
        mock_get_dataset_func_by_id.return_value = lambda x: x

        reference_result = {
            0: {
                0: {
                    P.closedopen(0, 25): {"topic": ["law"], "language": ["french"]},
                    P.closedopen(25, 50)
                    | P.closedopen(140, 150): {"topic": ["law", "medicine"], "language": ["french", "english"]},
                    P.closedopen(50, 75) | P.closedopen(180, 200): {"topic": ["medicine"], "language": ["english"]},
                    P.closedopen(80, 100): {"topic": ["medicine"]},
                    P.closedopen(100, 120) | P.closedopen(210, 300): {"language": ["french"]},
                    P.closedopen(120, 125): {"topic": ["medicine"], "language": ["french"]},
                    P.closedopen(125, 140): {"topic": ["law", "medicine"], "language": ["french"]},
                    P.closedopen(150, 180): {"topic": ["law", "medicine"], "language": ["english"]},
                    P.closedopen(200, 210): {"language": ["french", "english"]},
                    P.closedopen(300, 400): {"language": ["english"]},
                },
                1: {
                    P.closedopen(50, 100): {"topic": ["medicine"], "language": ["english"]},
                    P.closedopen(100, 150): {"topic": ["law", "medicine"], "language": ["french", "english"]},
                    P.closedopen(150, 160) | P.closedopen(170, 200) | P.closedopen(250, 350): {"language": ["french"]},
                    P.closedopen(160, 170): {"topic": ["medicine"], "language": ["french"]},
                    P.closedopen(200, 210): {"topic": ["law", "medicine"], "language": ["french"]},
                    P.closedopen(210, 250): {"topic": ["law"], "language": ["french"]},
                },
                2: {P.closedopen(10, 20): {"language": ["english"]}},
            },
            1: {
                0: {
                    P.closedopen(0, 25) | P.closedopen(200, 250): {"topic": ["law"]},
                    P.closedopen(25, 40) | P.closedopen(60, 75): {"topic": ["law"], "language": ["french"]},
                    P.closedopen(40, 50) | P.closedopen(75, 90): {"language": ["french"]},
                    P.closedopen(50, 60): {"topic": ["law", "medicine"], "language": ["english"]},
                    P.closedopen(90, 100): {"topic": ["medicine"], "language": ["french", "english"]},
                    P.closedopen(100, 110) | P.closedopen(130, 150): {"topic": ["medicine"], "language": ["english"]},
                },
                1: {
                    P.closedopen(0, 20): {"language": ["french"]},
                    P.closedopen(20, 30): {"topic": ["law"], "language": ["french"]},
                    P.closedopen(30, 50): {"topic": ["medicine"], "language": ["french"]},
                    P.closedopen(50, 100) | P.closedopen(150, 200): {"topic": ["medicine"]},
                },
            },
            2: {
                0: {
                    P.closedopen(0, 80) | P.closedopen(150, 200): {"language": ["english"]},
                    P.closedopen(80, 100): {"topic": ["law"], "language": ["english"]},
                }
            },
        }

        query = Query.for_job("job_id").complexmockoperator("test")
        args = QueryExecutionArgs(mixture=ArbitraryMixture(1))
        assert self.client.execute_query(query, args)
        inverted_index = query.results._invert_result(query.results.results)

        # True result vs reference
        for document_id, doc_entries in inverted_index.items():
            for file_id, file_entries in doc_entries.items():
                for intervals, properties in file_entries.items():
                    self.assertTrue(
                        document_id in reference_result
                        and file_id in reference_result[document_id]
                        and intervals in reference_result[document_id][file_id]
                    )
                    self.assertDictEqual(properties, reference_result[document_id][file_id][intervals])

        # Reference vs True result
        for document_id, doc_entries in reference_result.items():
            for file_id, file_entries in doc_entries.items():
                for intervals, properties in file_entries.items():
                    self.assertTrue(
                        document_id in inverted_index
                        and file_id in inverted_index[document_id]
                        and intervals in inverted_index[document_id][file_id]
                    )
                    self.assertDictEqual(properties, inverted_index[document_id][file_id][intervals].values()[0])

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

        # This assumes keys are generated using a single (the first) value of a property
        reference_result = {
            MixtureKey({"language": ["french"], "topic": ["law"]}): {
                0: {0: [[0, 25]], 1: [[210, 250]]},
                1: {0: [[25, 40], [60, 75]], 1: [[20, 30]]},
            },
            MixtureKey({"language": ["french", "english"], "topic": ["law", "medicine"]}): {
                0: {0: [[25, 50], [140, 150]], 1: [[100, 150]]},
            },
            MixtureKey({"language": ["english"], "topic": ["medicine"]}): {
                0: {0: [[50, 75], [180, 200]], 1: [[50, 100]]},
                1: {0: [[100, 110], [130, 150]]},
            },
            MixtureKey({"topic": ["medicine"]}): {0: {0: [[80, 100]]}, 1: {1: [[50, 100], [150, 200]]}},
            MixtureKey({"language": ["french"]}): {
                0: {0: [[100, 120], [210, 300]], 1: [[150, 160], [170, 200], [250, 350]]},
                1: {0: [[40, 50], [75, 90]], 1: [[0, 20]]},
            },
            MixtureKey({"language": ["french"], "topic": ["medicine"]}): {
                0: {0: [[120, 125]], 1: [[160, 170]]},
                1: {1: [[30, 50]]},
            },
            MixtureKey({"language": ["french"], "topic": ["law", "medicine"]}): {0: {0: [[125, 140]], 1: [[200, 210]]}},
            MixtureKey({"language": ["english"], "topic": ["law", "medicine"]}): {
                0: {0: [[150, 180]]},
                1: {0: [[50, 60]]},
            },
            MixtureKey({"language": ["french", "english"]}): {0: {0: [[200, 210]]}},
            MixtureKey({"language": ["english"]}): {0: {0: [[300, 400]], 2: [[10, 20]]}, 2: {0: [[0, 80], [150, 200]]}},
            MixtureKey({"topic": ["law"]}): {1: {0: [[0, 25], [200, 250]]}},
            MixtureKey({"language": ["french", "english"], "topic": ["medicine"]}): {1: {0: [[90, 100]]}},
            MixtureKey({"language": ["english"], "topic": ["law"]}): {2: {0: [[80, 100]]}},
        }

        query = Query.for_job("job_id").complexmockoperator("test")
        args = QueryExecutionArgs(mixture=ArbitraryMixture(1))
        assert self.client.execute_query(query, args)
        inverted_index = query.results._invert_result(query.results.results)
        chunk_index = query.results._create_chunker_index(inverted_index)
        self.assertDictEqual(defaultdict_to_dict(chunk_index), reference_result)

    @patch("mixtera.core.datacollection.MixteraDataCollection._get_dataset_func_by_id")
    @patch("mixtera.core.datacollection.MixteraDataCollection._get_dataset_type_by_id")
    @patch("mixtera.core.datacollection.MixteraDataCollection._get_file_path_by_id")
    def test_create_chunking_with_simple_static_mixture(
        self,
        mock_get_file_path_by_id: MagicMock,
        mock_get_dataset_type_by_id: MagicMock,
        mock_get_dataset_func_by_id: MagicMock,
    ):
        mock_get_file_path_by_id.return_value = "test_file_path"
        mock_get_dataset_type_by_id.return_value = "test_dataset_type"
        mock_get_dataset_func_by_id.return_value = lambda x: x

        reference_chunks = [
            {
                MixtureKey({"language": ["french"]}): {0: {0: [(0, 12)]}},
                MixtureKey({"language": ["english"]}): {0: {0: [(100, 104)]}},
            },
            {
                MixtureKey({"language": ["french"]}): {0: {0: [(12, 24)]}},
                MixtureKey({"language": ["english"]}): {0: {0: [(104, 108)]}},
            },
            {
                MixtureKey({"language": ["french"]}): {0: {0: [(24, 36)]}},
                MixtureKey({"language": ["english"]}): {0: {0: [(108, 112)]}},
            },
            {
                MixtureKey({"language": ["french"]}): {0: {0: [(36, 48)]}},
                MixtureKey({"language": ["english"]}): {0: {0: [(112, 116)]}},
            },
            {
                MixtureKey({"language": ["french"]}): {0: {0: [(48, 50), (150, 160)]}},
                MixtureKey({"language": ["english"]}): {0: {0: [(116, 120)]}},
            },
            {
                MixtureKey({"language": ["french"]}): {0: {0: [(160, 172)]}},
                MixtureKey({"language": ["english"]}): {0: {0: [(120, 124)]}},
            },
            {
                MixtureKey({"language": ["french"]}): {0: {0: [(172, 184)]}},
                MixtureKey({"language": ["english"]}): {0: {0: [(124, 128)]}},
            },
            {
                MixtureKey({"language": ["french"]}): {0: {0: [(184, 196)]}},
                MixtureKey({"language": ["english"]}): {0: {0: [(128, 132)]}},
            },
            {
                MixtureKey({"language": ["french"]}): {0: {0: [(196, 200)], 1: [(0, 8)]}},
                MixtureKey({"language": ["english"]}): {0: {0: [(132, 136)]}},
            },
            {
                MixtureKey({"language": ["french"]}): {0: {1: [(8, 20)]}},
                MixtureKey({"language": ["english"]}): {0: {0: [(136, 140)]}},
            },
            {
                MixtureKey({"language": ["french"]}): {0: {1: [(20, 32)]}},
                MixtureKey({"language": ["english"]}): {0: {0: [(140, 144)]}},
            },
            {
                MixtureKey({"language": ["french"]}): {0: {1: [(32, 44)]}},
                MixtureKey({"language": ["english"]}): {0: {0: [(144, 148)]}},
            },
            {
                MixtureKey({"language": ["french"]}): {0: {1: [(44, 56)]}},
                MixtureKey({"language": ["english"]}): {0: {0: [(148, 150), (50, 52)]}},
            },
            {
                MixtureKey({"language": ["french"]}): {0: {1: [(56, 68)]}},
                MixtureKey({"language": ["english"]}): {0: {0: [(52, 56)]}},
            },
            {
                MixtureKey({"language": ["french"]}): {0: {1: [(68, 80)]}},
                MixtureKey({"language": ["english"]}): {0: {0: [(56, 60)]}},
            },
            {
                MixtureKey({"language": ["french"]}): {0: {1: [(80, 92)]}},
                MixtureKey({"language": ["english"]}): {0: {0: [(60, 64)]}},
            },
            {
                MixtureKey({"language": ["french"]}): {0: {1: [(92, 100)], 0: [(64, 68)]}},
                MixtureKey({"language": ["english"]}): {0: {0: [(68, 72)]}},
            },
            {
                MixtureKey({"language": ["french"]}): {0: {0: [(72, 84)]}},
                MixtureKey({"language": ["english"]}): {0: {0: [(84, 88)]}},
            },
        ]

        reference_chunker_index = {
            MixtureKey({"language": ["french"]}): {0: {0: [[0, 50], [150, 200]], 1: [[0, 100]]}},
            MixtureKey({"language": ["english", "french"]}): {0: {0: [[50, 100]]}},
            MixtureKey({"language": ["english"]}): {0: {0: [[100, 150]]}},
        }

        mixture_concentration = {
            MixtureKey({"language": ["french"]}): 0.75,  # 12 instances per batch
            MixtureKey({"language": ["english"]}): 0.25,  # 4 instances per batch
        }

        query = Query.for_job("job_id").simplemockoperator("test")
        mixture = StaticMixture(16, mixture_concentration)
        args = QueryExecutionArgs(mixture=mixture)

        assert self.client.execute_query(query, args)
        chunks = list(iter(query.results))

        # Check the structure of the chunker index
        inverted_index = query.results._invert_result(query.results.results)
        chunker_index = defaultdict_to_dict(query.results._create_chunker_index(inverted_index))
        self.assertDictEqual(defaultdict_to_dict(chunker_index), reference_chunker_index)

        # Check the equality of the chunks
        for i, chunk in enumerate(chunks):
            self.assertDictEqual(reference_chunks[i], chunk._result_index)

    @patch("mixtera.core.datacollection.MixteraDataCollection._get_dataset_func_by_id")
    @patch("mixtera.core.datacollection.MixteraDataCollection._get_dataset_type_by_id")
    @patch("mixtera.core.datacollection.MixteraDataCollection._get_file_path_by_id")
    def test_create_chunking_with_simple_inferring_mixture(
        self,
        mock_get_file_path_by_id: MagicMock,
        mock_get_dataset_type_by_id: MagicMock,
        mock_get_dataset_func_by_id: MagicMock,
    ):
        mock_get_file_path_by_id.return_value = "test_file_path"
        mock_get_dataset_type_by_id.return_value = "test_dataset_type"
        mock_get_dataset_func_by_id.return_value = lambda x: x

        reference_chunks = [
            {
                MixtureKey({"language": ["french"]}): {0: {0: [(0, 20)]}},
                MixtureKey({"language": ["english", "french"]}): {0: {0: [(20, 25)]}},
                MixtureKey({"language": ["english"]}): {0: {0: [(100, 105)]}},
            },
            {
                MixtureKey({"language": ["french"]}): {0: {0: [(25, 45)]}},
                MixtureKey({"language": ["english", "french"]}): {0: {0: [(45, 50)]}},
                MixtureKey({"language": ["english"]}): {0: {0: [(105, 110)]}},
            },
            {
                MixtureKey({"language": ["french"]}): {0: {0: [(150, 170)]}},
                MixtureKey({"language": ["english", "french"]}): {0: {0: [(170, 175)]}},
                MixtureKey({"language": ["english"]}): {0: {0: [(110, 115)]}},
            },
            {
                MixtureKey({"language": ["french"]}): {0: {0: [(175, 195)]}},
                MixtureKey({"language": ["english", "french"]}): {0: {0: [(195, 200)]}},
                MixtureKey({"language": ["english"]}): {0: {0: [(115, 120)]}},
            },
            {
                MixtureKey({"language": ["french"]}): {0: {1: [(0, 20)]}},
                MixtureKey({"language": ["english", "french"]}): {0: {1: [(20, 25)]}},
                MixtureKey({"language": ["english"]}): {0: {0: [(120, 125)]}},
            },
            {
                MixtureKey({"language": ["french"]}): {0: {1: [(25, 45)]}},
                MixtureKey({"language": ["english", "french"]}): {0: {1: [(45, 50)]}},
                MixtureKey({"language": ["english"]}): {0: {0: [(125, 130)]}},
            },
            {
                MixtureKey({"language": ["french"]}): {0: {1: [(50, 70)]}},
                MixtureKey({"language": ["english", "french"]}): {0: {1: [(70, 75)]}},
                MixtureKey({"language": ["english"]}): {0: {0: [(130, 135)]}},
            },
            {
                MixtureKey({"language": ["french"]}): {0: {1: [(75, 95)]}},
                MixtureKey({"language": ["english", "french"]}): {0: {1: [(95, 100)]}},
                MixtureKey({"language": ["english"]}): {0: {0: [(135, 140)]}},
            },
            {
                MixtureKey({"language": ["french"]}): {0: {0: [(50, 70)]}},
                MixtureKey({"language": ["english", "french"]}): {0: {0: [(140, 145)]}},
                MixtureKey({"language": ["english"]}): {0: {0: [(145, 150)]}},
            },
            {
                MixtureKey({"language": ["french"]}): {0: {0: [(70, 90)]}},
                MixtureKey({"language": ["english", "french"]}): {0: {0: [(90, 95)]}},
                MixtureKey({"language": ["english"]}): {0: {0: [(95, 100)]}},
            },
        ]

        reference_chunker_index = {
            MixtureKey({"language": ["french"]}): {0: {0: [[0, 50], [150, 200]], 1: [[0, 100]]}},
            MixtureKey({"language": ["english"]}): {0: {0: [[100, 150]]}},
            MixtureKey({"language": ["english", "french"]}): {0: {0: [[50, 100]]}},
        }

        # We have 100 English and 200 French lines in the chunker index (see reference above)
        # This means we have 2/3 French data, and 1/3 English data per chunk
        # Hence in the chunks above we always have 20 lines french 5 lines english and 5 lines french+english.

        query = Query.for_job("job_id").simplemockoperator("test")
        mixture = InferringMixture(30)
        args = QueryExecutionArgs(mixture=mixture)

        assert self.client.execute_query(query, args)
        assert mixture._mixture == {
            MixtureKey({"language": ["french"]}): 20,
            MixtureKey({"language": ["english"]}): 5,
            MixtureKey({"language": ["english", "french"]}): 5,
        }

        chunks = list(iter(query.results))

        # Check the structure of the chunker index
        inverted_index = query.results._invert_result(query.results.results)
        chunker_index = defaultdict_to_dict(query.results._create_chunker_index(inverted_index))
        self.assertDictEqual(defaultdict_to_dict(chunker_index), reference_chunker_index)

        # Check the equality of the chunks
        for i, chunk in enumerate(chunks):
            self.assertDictEqual(reference_chunks[i], chunk._result_index)

    @patch("mixtera.core.datacollection.MixteraDataCollection._get_dataset_func_by_id")
    @patch("mixtera.core.datacollection.MixteraDataCollection._get_dataset_type_by_id")
    @patch("mixtera.core.datacollection.MixteraDataCollection._get_file_path_by_id")
    def test_create_chunking_with_simple_dynamic_mixture(
        self,
        mock_get_file_path_by_id: MagicMock,
        mock_get_dataset_type_by_id: MagicMock,
        mock_get_dataset_func_by_id: MagicMock,
    ):
        mock_get_file_path_by_id.return_value = "test_file_path"
        mock_get_dataset_type_by_id.return_value = "test_dataset_type"
        mock_get_dataset_func_by_id.return_value = lambda x: x

        reference_chunks = [
            {
                MixtureKey({"language": ["french"]}): {0: {0: [(0, 12)]}},
                MixtureKey({"language": ["english"]}): {0: {0: [(100, 104)]}},
            },
            {
                MixtureKey({"language": ["french"]}): {0: {0: [(12, 24)]}},
                MixtureKey({"language": ["english"]}): {0: {0: [(104, 108)]}},
            },
            {
                MixtureKey({"language": ["french"]}): {0: {0: [(24, 36)]}},
                MixtureKey({"language": ["english"]}): {0: {0: [(108, 112)]}},
            },
            {
                MixtureKey({"language": ["french"]}): {0: {0: [(36, 48)]}},
                MixtureKey({"language": ["english"]}): {0: {0: [(112, 116)]}},
            },
            {
                MixtureKey({"language": ["french"]}): {0: {0: [(48, 50), (150, 160)]}},
                MixtureKey({"language": ["english"]}): {0: {0: [(116, 120)]}},
            },
            {
                MixtureKey({"language": ["french"]}): {0: {0: [(160, 172)]}},
                MixtureKey({"language": ["english"]}): {0: {0: [(120, 124)]}},
            },
            {
                MixtureKey({"language": ["french"]}): {0: {0: [(172, 184)]}},
                MixtureKey({"language": ["english"]}): {0: {0: [(124, 128)]}},
            },
            {
                MixtureKey({"language": ["french"]}): {0: {0: [(184, 196)]}},
                MixtureKey({"language": ["english"]}): {0: {0: [(128, 132)]}},
            },
            {
                MixtureKey({"language": ["french"]}): {0: {0: [(196, 200)], 1: [(0, 8)]}},
                MixtureKey({"language": ["english"]}): {0: {0: [(132, 136)]}},
            },
            {
                MixtureKey({"language": ["french"]}): {0: {1: [(8, 20)]}},
                MixtureKey({"language": ["english"]}): {0: {0: [(136, 140)]}},
            },
            {
                MixtureKey({"language": ["french"]}): {0: {1: [(20, 28)]}},
                MixtureKey({"language": ["english"]}): {0: {0: [(140, 148)]}},
            },
            {
                MixtureKey({"language": ["french"]}): {0: {1: [(28, 36)]}},
                MixtureKey({"language": ["english"]}): {0: {0: [(148, 150), (50, 56)]}},
            },
            {
                MixtureKey({"language": ["french"]}): {0: {1: [(36, 44)]}},
                MixtureKey({"language": ["english"]}): {0: {0: [(56, 64)]}},
            },
            {
                MixtureKey({"language": ["french"]}): {0: {1: [(44, 52)]}},
                MixtureKey({"language": ["english"]}): {0: {0: [(64, 72)]}},
            },
            {
                MixtureKey({"language": ["french"]}): {0: {1: [(52, 60)]}},
                MixtureKey({"language": ["english"]}): {0: {0: [(72, 80)]}},
            },
            {
                MixtureKey({"language": ["french"]}): {0: {1: [(60, 68)]}},
                MixtureKey({"language": ["english"]}): {0: {0: [(80, 88)]}},
            },
            {
                MixtureKey({"language": ["french"]}): {0: {1: [(68, 76)]}},
                MixtureKey({"language": ["english"]}): {0: {0: [(88, 96)]}},
            },
        ]

        mixture_concentration_1 = {
            MixtureKey({"language": ["french"]}): 0.75,  # 12 instances per batch
            MixtureKey({"language": ["english"]}): 0.25,  # 4 instances per batch
        }

        mixture_concentration_2 = {
            MixtureKey({"language": ["french"]}): 0.5,  # 8 and 8 instances per batch
            MixtureKey({"language": ["english"]}): 0.5,  # 8 and 8 instances per batch
        }

        query = Query.for_job("job_id").simplemockoperator("test")

        mixture_1 = StaticMixture(16, mixture_concentration_1)
        mixture_2 = StaticMixture(16, mixture_concentration_2)
        args = QueryExecutionArgs(mixture=mixture_1)

        assert self.client.execute_query(query, args)
        result_iterator = iter(query.results)

        chunks = [next(result_iterator) for _ in range(10)]
        query.results.update_mixture(mixture_2)
        chunks.extend([next(result_iterator) for _ in range(7)])
        self.assertRaises(StopIteration, next, result_iterator)

        # Check the equality of the chunks
        for i, chunk in enumerate(chunks):
            self.assertDictEqual(reference_chunks[i], chunk._result_index)

    @patch("mixtera.core.datacollection.MixteraDataCollection._get_dataset_func_by_id")
    @patch("mixtera.core.datacollection.MixteraDataCollection._get_dataset_type_by_id")
    @patch("mixtera.core.datacollection.MixteraDataCollection._get_file_path_by_id")
    def test_create_complex_chunking_with_mixture(
        self,
        mock_get_file_path_by_id: MagicMock,
        mock_get_dataset_type_by_id: MagicMock,
        mock_get_dataset_func_by_id: MagicMock,
    ):
        mock_get_file_path_by_id.return_value = "test_file_path"
        mock_get_dataset_type_by_id.return_value = "test_dataset_type"
        mock_get_dataset_func_by_id.return_value = lambda x: x

        reference_chunks = [
            {
                MixtureKey({"language": ["french"], "topic": ["law"]}): {0: {0: [(0, 6)]}},
                MixtureKey({"language": ["english"], "topic": ["medicine"]}): {0: {0: [(50, 54)]}},
            },
            {
                MixtureKey({"language": ["french"], "topic": ["law"]}): {0: {0: [(6, 12)]}},
                MixtureKey({"language": ["english"], "topic": ["medicine"]}): {0: {0: [(54, 58)]}},
            },
            {
                MixtureKey({"language": ["french"], "topic": ["law"]}): {0: {0: [(12, 18)]}},
                MixtureKey({"language": ["english"], "topic": ["medicine"]}): {0: {0: [(58, 62)]}},
            },
            {
                MixtureKey({"language": ["french"], "topic": ["law"]}): {0: {0: [(18, 24)]}},
                MixtureKey({"language": ["english"], "topic": ["medicine"]}): {0: {0: [(62, 66)]}},
            },
            {
                MixtureKey({"language": ["french"], "topic": ["law"]}): {0: {0: [(24, 25)], 1: [(210, 215)]}},
                MixtureKey({"language": ["english"], "topic": ["medicine"]}): {0: {0: [(66, 70)]}},
            },
            {
                MixtureKey({"language": ["french"], "topic": ["law"]}): {0: {1: [(215, 221)]}},
                MixtureKey({"language": ["english"], "topic": ["medicine"]}): {0: {0: [(70, 74)]}},
            },
            {
                MixtureKey({"language": ["french"], "topic": ["law"]}): {0: {1: [(221, 227)]}},
                MixtureKey({"language": ["english"], "topic": ["medicine"]}): {0: {0: [(74, 75), (180, 183)]}},
            },
            {
                MixtureKey({"language": ["french"], "topic": ["law"]}): {0: {1: [(227, 233)]}},
                MixtureKey({"language": ["english"], "topic": ["medicine"]}): {0: {0: [(183, 187)]}},
            },
            {
                MixtureKey({"language": ["french"], "topic": ["law"]}): {0: {1: [(233, 239)]}},
                MixtureKey({"language": ["english"], "topic": ["medicine"]}): {0: {0: [(187, 191)]}},
            },
            {
                MixtureKey({"language": ["french"], "topic": ["law"]}): {0: {1: [(239, 245)]}},
                MixtureKey({"language": ["english"], "topic": ["medicine"]}): {0: {0: [(191, 195)]}},
            },
            {
                MixtureKey({"language": ["french"], "topic": ["law"]}): {0: {1: [(245, 250)]}, 1: {0: [(25, 26)]}},
                MixtureKey({"language": ["english"], "topic": ["medicine"]}): {0: {0: [(195, 199)]}},
            },
            {
                MixtureKey({"language": ["french"], "topic": ["law"]}): {1: {0: [(26, 32)]}},
                MixtureKey({"language": ["english"], "topic": ["medicine"]}): {0: {0: [(199, 200)], 1: [(50, 53)]}},
            },
            {
                MixtureKey({"language": ["french"], "topic": ["law"]}): {1: {0: [(32, 38)]}},
                MixtureKey({"language": ["english"], "topic": ["medicine"]}): {0: {1: [(53, 57)]}},
            },
            {
                MixtureKey({"language": ["french"], "topic": ["law"]}): {1: {0: [(38, 40), (60, 64)]}},
                MixtureKey({"language": ["english"], "topic": ["medicine"]}): {0: {1: [(57, 61)]}},
            },
            {
                MixtureKey({"language": ["french"], "topic": ["law"]}): {1: {0: [(64, 70)]}},
                MixtureKey({"language": ["english"], "topic": ["medicine"]}): {0: {1: [(61, 65)]}},
            },
            {
                MixtureKey({"language": ["french"], "topic": ["law"]}): {1: {0: [(70, 75)], 1: [(20, 21)]}},
                MixtureKey({"language": ["english"], "topic": ["medicine"]}): {0: {1: [(65, 69)]}},
            },
            {
                MixtureKey({"language": ["french"], "topic": ["law"]}): {1: {1: [(21, 27)]}},
                MixtureKey({"language": ["english"], "topic": ["medicine"]}): {0: {1: [(69, 73)]}},
            },
            {
                MixtureKey({"language": ["french"], "topic": ["law"]}): {1: {1: [(27, 30)]}, 0: {0: [(125, 128)]}},
                MixtureKey({"language": ["english"], "topic": ["medicine"]}): {0: {1: [(73, 77)]}},
            },
            {
                MixtureKey({"language": ["french"], "topic": ["law"]}): {0: {0: [(128, 134)]}},
                MixtureKey({"language": ["english"], "topic": ["medicine"]}): {0: {1: [(77, 81)]}},
            },
            {
                MixtureKey({"language": ["french"], "topic": ["law"]}): {0: {0: [(134, 140)]}},
                MixtureKey({"language": ["english"], "topic": ["medicine"]}): {0: {1: [(81, 85)]}},
            },
            {
                MixtureKey({"language": ["french"], "topic": ["law"]}): {0: {1: [(200, 206)]}},
                MixtureKey({"language": ["english"], "topic": ["medicine"]}): {0: {1: [(85, 89)]}},
            },
            {
                MixtureKey({"language": ["french"], "topic": ["law"]}): {0: {1: [(206, 210)], 0: [(25, 27)]}},
                MixtureKey({"language": ["english"], "topic": ["medicine"]}): {0: {1: [(89, 93)]}},
            },
            {
                MixtureKey({"language": ["french"], "topic": ["law"]}): {0: {0: [(27, 33)]}},
                MixtureKey({"language": ["english"], "topic": ["medicine"]}): {0: {1: [(93, 97)]}},
            },
            {
                MixtureKey({"language": ["french"], "topic": ["law"]}): {0: {0: [(33, 39)]}},
                MixtureKey({"language": ["english"], "topic": ["medicine"]}): {
                    0: {1: [(97, 100)]},
                    1: {0: [(100, 101)]},
                },
            },
            {
                MixtureKey({"language": ["french"], "topic": ["law"]}): {0: {0: [(39, 45)]}},
                MixtureKey({"language": ["english"], "topic": ["medicine"]}): {1: {0: [(101, 105)]}},
            },
            {
                MixtureKey({"language": ["french"], "topic": ["law"]}): {0: {0: [(45, 50), (140, 141)]}},
                MixtureKey({"language": ["english"], "topic": ["medicine"]}): {1: {0: [(105, 109)]}},
            },
            {
                MixtureKey({"language": ["french"], "topic": ["law"]}): {0: {0: [(141, 147)]}},
                MixtureKey({"language": ["english"], "topic": ["medicine"]}): {1: {0: [(109, 110), (130, 133)]}},
            },
            {
                MixtureKey({"language": ["french"], "topic": ["law"]}): {0: {0: [(147, 150)], 1: [(100, 103)]}},
                MixtureKey({"language": ["english"], "topic": ["medicine"]}): {1: {0: [(133, 137)]}},
            },
            {
                MixtureKey({"language": ["french"], "topic": ["law"]}): {0: {1: [(103, 109)]}},
                MixtureKey({"language": ["english"], "topic": ["medicine"]}): {1: {0: [(137, 141)]}},
            },
            {
                MixtureKey({"language": ["french"], "topic": ["law"]}): {0: {1: [(109, 115)]}},
                MixtureKey({"language": ["english"], "topic": ["medicine"]}): {1: {0: [(141, 145)]}},
            },
            {
                MixtureKey({"language": ["french"], "topic": ["law"]}): {0: {1: [(115, 121)]}},
                MixtureKey({"language": ["english"], "topic": ["medicine"]}): {1: {0: [(145, 149)]}},
            },
            {
                MixtureKey({"language": ["french"], "topic": ["law"]}): {0: {1: [(121, 127)]}},
                MixtureKey({"language": ["english"], "topic": ["medicine"]}): {1: {0: [(149, 150), (90, 93)]}},
            },
            {
                MixtureKey({"language": ["french"], "topic": ["law"]}): {0: {1: [(127, 133)]}},
                MixtureKey({"language": ["english"], "topic": ["medicine"]}): {1: {0: [(93, 97)]}},
            },
            {
                MixtureKey({"language": ["french"], "topic": ["law"]}): {0: {1: [(133, 139)]}},
                MixtureKey({"language": ["english"], "topic": ["medicine"]}): {
                    1: {0: [(97, 100)]},
                    0: {0: [(150, 151)]},
                },
            },
            {
                MixtureKey({"language": ["french"], "topic": ["law"]}): {0: {1: [(139, 145)]}},
                MixtureKey({"language": ["english"], "topic": ["medicine"]}): {0: {0: [(151, 155)]}},
            },
        ]
        mixture_concentration = {
            MixtureKey({"language": ["french"], "topic": ["law"]}): 0.6,  # 6 instances per batch
            MixtureKey({"language": ["english"], "topic": ["medicine"]}): 0.4,  # 4 instances per batch
        }

        query = Query.for_job("job_id").complexmockoperator("test")

        mixture = StaticMixture(10, mixture_concentration)
        args = QueryExecutionArgs(mixture=mixture)
        assert self.client.execute_query(query, args)
        chunks = list(iter(query.results))

        for i, chunk in enumerate(chunks):
            self.assertDictEqual(reference_chunks[i], chunk._result_index)

    @patch("mixtera.core.datacollection.MixteraDataCollection._get_dataset_func_by_id")
    @patch("mixtera.core.datacollection.MixteraDataCollection._get_dataset_type_by_id")
    @patch("mixtera.core.datacollection.MixteraDataCollection._get_file_path_by_id")
    def test_create_chunking_without_mixture(
        self,
        mock_get_file_path_by_id: MagicMock,
        mock_get_dataset_type_by_id: MagicMock,
        mock_get_dataset_func_by_id: MagicMock,
    ):
        mock_get_file_path_by_id.return_value = "test_file_path"
        mock_get_dataset_type_by_id.return_value = "test_dataset_type"
        mock_get_dataset_func_by_id.return_value = lambda x: x

        reference_chunks = [
            {MixtureKey({"language": ["english"]}): {0: {0: [(300, 307)]}}},
            {MixtureKey({"language": ["english"], "topic": ["medicine"]}): {0: {0: [(50, 57)]}}},
            {MixtureKey({"language": ["french"], "topic": ["medicine"]}): {0: {0: [(120, 125)], 1: [(160, 162)]}}},
            {MixtureKey({"language": ["french"], "topic": ["law"]}): {0: {0: [(0, 7)]}}},
            {MixtureKey({"language": ["french"], "topic": ["law", "medicine"]}): {0: {0: [(125, 132)]}}},
            {MixtureKey({"topic": ["law"]}): {1: {0: [(0, 7)]}}},
            {MixtureKey({"language": ["english"], "topic": ["law"]}): {2: {0: [(80, 87)]}}},
            {MixtureKey({"language": ["english"], "topic": ["law", "medicine"]}): {0: {0: [(150, 157)]}}},
            {MixtureKey({"language": ["french"]}): {0: {0: [(100, 107)]}}},
            {MixtureKey({"language": ["french", "english"], "topic": ["law", "medicine"]}): {0: {0: [(25, 32)]}}},
            {MixtureKey({"topic": ["medicine"]}): {0: {0: [(80, 87)]}}},
            {MixtureKey({"language": ["french", "english"]}): {0: {0: [(200, 207)]}}},
            {MixtureKey({"language": ["english", "french"], "topic": ["medicine"]}): {1: {0: [(90, 97)]}}},
            {MixtureKey({"language": ["english"]}): {0: {0: [(307, 314)]}}},
            {MixtureKey({"language": ["english"], "topic": ["medicine"]}): {0: {0: [(57, 64)]}}},
            {MixtureKey({"language": ["french"], "topic": ["medicine"]}): {0: {1: [(162, 169)]}}},
            {MixtureKey({"language": ["french"], "topic": ["law"]}): {0: {0: [(7, 14)]}}},
            {MixtureKey({"language": ["french"], "topic": ["law", "medicine"]}): {0: {0: [(132, 139)]}}},
            {MixtureKey({"topic": ["law"]}): {1: {0: [(7, 14)]}}},
            {MixtureKey({"language": ["english"], "topic": ["law"]}): {2: {0: [(87, 94)]}}},
            {MixtureKey({"language": ["english"], "topic": ["law", "medicine"]}): {0: {0: [(157, 164)]}}},
            {MixtureKey({"language": ["french"]}): {0: {0: [(107, 114)]}}},
            {MixtureKey({"language": ["french", "english"], "topic": ["law", "medicine"]}): {0: {0: [(32, 39)]}}},
            {MixtureKey({"topic": ["medicine"]}): {0: {0: [(87, 94)]}}},
            {MixtureKey({"language": ["french", "english"]}): {0: {0: [(207, 210)]}}},
            {MixtureKey({"language": ["english", "french"], "topic": ["medicine"]}): {1: {0: [(97, 100)]}}},
            {MixtureKey({"language": ["english"]}): {0: {0: [(314, 321)]}}},
            {MixtureKey({"language": ["english"], "topic": ["medicine"]}): {0: {0: [(64, 71)]}}},
            {MixtureKey({"language": ["french"], "topic": ["medicine"]}): {0: {1: [(169, 170)]}, 1: {1: [(30, 36)]}}},
            {MixtureKey({"language": ["french"], "topic": ["law"]}): {0: {0: [(14, 21)]}}},
            {
                MixtureKey({"language": ["french"], "topic": ["law", "medicine"]}): {
                    0: {0: [(139, 140)], 1: [(200, 206)]}
                }
            },
            {MixtureKey({"topic": ["law"]}): {1: {0: [(14, 21)]}}},
            {MixtureKey({"language": ["english"], "topic": ["law"]}): {2: {0: [(94, 100)]}}},
            {MixtureKey({"language": ["english"], "topic": ["law", "medicine"]}): {0: {0: [(164, 171)]}}},
            {MixtureKey({"language": ["french"]}): {0: {0: [(114, 120), (210, 211)]}}},
            {MixtureKey({"language": ["french", "english"], "topic": ["law", "medicine"]}): {0: {0: [(39, 46)]}}},
            {MixtureKey({"topic": ["medicine"]}): {0: {0: [(94, 100)]}, 1: {1: [(50, 51)]}}},
            {MixtureKey({"language": ["english"]}): {0: {0: [(321, 328)]}}},
            {MixtureKey({"language": ["english"], "topic": ["medicine"]}): {0: {0: [(71, 75), (180, 183)]}}},
            {MixtureKey({"language": ["french"], "topic": ["medicine"]}): {1: {1: [(36, 43)]}}},
            {MixtureKey({"language": ["french"], "topic": ["law"]}): {0: {0: [(21, 25)], 1: [(210, 213)]}}},
            {MixtureKey({"language": ["french"], "topic": ["law", "medicine"]}): {0: {1: [(206, 210)]}}},
            {MixtureKey({"topic": ["law"]}): {1: {0: [(21, 25), (200, 203)]}}},
            {MixtureKey({"language": ["english"], "topic": ["law", "medicine"]}): {0: {0: [(171, 178)]}}},
            {MixtureKey({"language": ["french"]}): {0: {0: [(211, 218)]}}},
            {
                MixtureKey({"language": ["french", "english"], "topic": ["law", "medicine"]}): {
                    0: {0: [(46, 50), (140, 143)]}
                }
            },
            {MixtureKey({"topic": ["medicine"]}): {1: {1: [(51, 58)]}}},
            {MixtureKey({"language": ["english"]}): {0: {0: [(328, 335)]}}},
            {MixtureKey({"language": ["english"], "topic": ["medicine"]}): {0: {0: [(183, 190)]}}},
            {MixtureKey({"language": ["french"], "topic": ["medicine"]}): {1: {1: [(43, 50)]}}},
            {MixtureKey({"language": ["french"], "topic": ["law"]}): {0: {1: [(213, 220)]}}},
            {MixtureKey({"topic": ["law"]}): {1: {0: [(203, 210)]}}},
            {MixtureKey({"language": ["french"]}): {0: {0: [(218, 225)]}}},
            {MixtureKey({"language": ["french", "english"], "topic": ["law", "medicine"]}): {0: {0: [(143, 150)]}}},
            {MixtureKey({"topic": ["medicine"]}): {1: {1: [(58, 65)]}}},
            {MixtureKey({"language": ["english"]}): {0: {0: [(335, 342)]}}},
            {MixtureKey({"language": ["english"], "topic": ["medicine"]}): {0: {0: [(190, 197)]}}},
            {MixtureKey({"language": ["french"], "topic": ["law"]}): {0: {1: [(220, 227)]}}},
            {MixtureKey({"language": ["french"]}): {0: {0: [(225, 232)]}}},
            {MixtureKey({"language": ["french", "english"], "topic": ["law", "medicine"]}): {0: {1: [(100, 107)]}}},
            {MixtureKey({"topic": ["medicine"]}): {1: {1: [(65, 72)]}}},
            {MixtureKey({"language": ["english"]}): {0: {0: [(342, 349)]}}},
            {MixtureKey({"language": ["english"], "topic": ["medicine"]}): {0: {0: [(197, 200)], 1: [(50, 54)]}}},
            {MixtureKey({"language": ["french"]}): {0: {0: [(232, 239)]}}},
            {MixtureKey({"language": ["french", "english"], "topic": ["law", "medicine"]}): {0: {1: [(107, 114)]}}},
            {MixtureKey({"topic": ["medicine"]}): {1: {1: [(72, 79)]}}},
            {MixtureKey({"language": ["english"]}): {0: {0: [(349, 356)]}}},
            {MixtureKey({"language": ["english"], "topic": ["medicine"]}): {0: {1: [(54, 61)]}}},
            {MixtureKey({"language": ["french"]}): {0: {0: [(239, 246)]}}},
            {MixtureKey({"language": ["french", "english"], "topic": ["law", "medicine"]}): {0: {1: [(114, 121)]}}},
            {MixtureKey({"topic": ["medicine"]}): {1: {1: [(79, 86)]}}},
            {MixtureKey({"language": ["english"]}): {0: {0: [(356, 363)]}}},
            {MixtureKey({"language": ["english"], "topic": ["medicine"]}): {0: {1: [(61, 68)]}}},
            {MixtureKey({"language": ["french"]}): {0: {0: [(246, 253)]}}},
            {MixtureKey({"language": ["french", "english"], "topic": ["law", "medicine"]}): {0: {1: [(121, 128)]}}},
            {MixtureKey({"topic": ["medicine"]}): {1: {1: [(86, 93)]}}},
            {MixtureKey({"language": ["english"]}): {0: {0: [(363, 370)]}}},
            {MixtureKey({"language": ["english"], "topic": ["medicine"]}): {0: {1: [(68, 75)]}}},
            {MixtureKey({"language": ["french"]}): {0: {0: [(253, 260)]}}},
            {MixtureKey({"language": ["french", "english"], "topic": ["law", "medicine"]}): {0: {1: [(128, 135)]}}},
            {MixtureKey({"topic": ["medicine"]}): {1: {1: [(93, 100)]}}},
            {MixtureKey({"language": ["english"]}): {0: {0: [(370, 377)]}}},
            {MixtureKey({"language": ["english"], "topic": ["medicine"]}): {0: {1: [(75, 82)]}}},
            {MixtureKey({"language": ["french"]}): {0: {0: [(260, 267)]}}},
            {MixtureKey({"language": ["french", "english"], "topic": ["law", "medicine"]}): {0: {1: [(135, 142)]}}},
            {MixtureKey({"topic": ["medicine"]}): {1: {1: [(150, 157)]}}},
            {MixtureKey({"language": ["english"]}): {0: {0: [(377, 384)]}}},
            {MixtureKey({"language": ["english"], "topic": ["medicine"]}): {0: {1: [(82, 89)]}}},
            {MixtureKey({"language": ["french"]}): {0: {0: [(267, 274)]}}},
            {MixtureKey({"language": ["french", "english"], "topic": ["law", "medicine"]}): {0: {1: [(142, 149)]}}},
            {MixtureKey({"topic": ["medicine"]}): {1: {1: [(157, 164)]}}},
            {MixtureKey({"language": ["english"]}): {0: {0: [(384, 391)]}}},
            {MixtureKey({"language": ["english"], "topic": ["medicine"]}): {0: {1: [(89, 96)]}}},
            {MixtureKey({"language": ["french"]}): {0: {0: [(274, 281)]}}},
            {MixtureKey({"language": ["french", "english"], "topic": ["law", "medicine"]}): {0: {1: [(149, 150)]}}},
            {MixtureKey({"topic": ["medicine"]}): {1: {1: [(164, 171)]}}},
            {MixtureKey({"language": ["english"]}): {0: {0: [(391, 398)]}}},
            {MixtureKey({"language": ["english"], "topic": ["medicine"]}): {0: {1: [(96, 100)]}, 1: {0: [(100, 103)]}}},
            {MixtureKey({"language": ["french"]}): {0: {0: [(281, 288)]}}},
            {MixtureKey({"topic": ["medicine"]}): {1: {1: [(171, 178)]}}},
            {MixtureKey({"language": ["english"]}): {0: {0: [(398, 400)], 2: [(10, 15)]}}},
            {MixtureKey({"language": ["english"], "topic": ["medicine"]}): {1: {0: [(103, 110)]}}},
            {MixtureKey({"language": ["french"]}): {0: {0: [(288, 295)]}}},
            {MixtureKey({"language": ["english"]}): {0: {2: [(15, 20)]}, 2: {0: [(0, 2)]}}},
            {MixtureKey({"language": ["english"], "topic": ["medicine"]}): {1: {0: [(130, 137)]}}},
            {MixtureKey({"language": ["french"]}): {0: {0: [(295, 300)], 1: [(150, 152)]}}},
            {MixtureKey({"language": ["english"]}): {2: {0: [(2, 9)]}}},
            {MixtureKey({"language": ["english"], "topic": ["medicine"]}): {1: {0: [(137, 144)]}}},
            {MixtureKey({"language": ["french"]}): {0: {1: [(152, 159)]}}},
            {MixtureKey({"language": ["english"]}): {2: {0: [(9, 16)]}}},
            {MixtureKey({"language": ["english"], "topic": ["medicine"]}): {1: {0: [(144, 150)]}}},
            {MixtureKey({"language": ["french"]}): {0: {1: [(159, 160), (170, 176)]}}},
            {MixtureKey({"language": ["english"]}): {2: {0: [(16, 23)]}}},
            {MixtureKey({"language": ["french"]}): {0: {1: [(176, 183)]}}},
            {MixtureKey({"language": ["english"]}): {2: {0: [(23, 30)]}}},
            {MixtureKey({"language": ["french"], "topic": ["law"]}): {0: {1: [(227, 234)]}}},
            {MixtureKey({"language": ["french"]}): {0: {1: [(183, 190)]}}},
            {MixtureKey({"language": ["english"]}): {2: {0: [(30, 37)]}}},
            {MixtureKey({"language": ["french"], "topic": ["law"]}): {0: {1: [(234, 241)]}}},
            {MixtureKey({"language": ["french"]}): {0: {1: [(190, 197)]}}},
            {MixtureKey({"language": ["english"]}): {2: {0: [(37, 44)]}}},
            {MixtureKey({"language": ["french"], "topic": ["law"]}): {0: {1: [(241, 248)]}}},
            {MixtureKey({"language": ["french"]}): {0: {1: [(197, 200), (250, 254)]}}},
            {MixtureKey({"language": ["english"]}): {2: {0: [(44, 51)]}}},
            {MixtureKey({"language": ["french"], "topic": ["law"]}): {0: {1: [(248, 250)]}, 1: {0: [(25, 30)]}}},
            {MixtureKey({"language": ["french"]}): {0: {1: [(254, 261)]}}},
            {MixtureKey({"language": ["english"]}): {2: {0: [(51, 58)]}}},
            {MixtureKey({"language": ["french"], "topic": ["law"]}): {1: {0: [(30, 37)]}}},
            {MixtureKey({"language": ["french"]}): {0: {1: [(261, 268)]}}},
            {MixtureKey({"language": ["english"]}): {2: {0: [(58, 65)]}}},
            {MixtureKey({"language": ["french"], "topic": ["law"]}): {1: {0: [(37, 40), (60, 64)]}}},
            {MixtureKey({"language": ["french"]}): {0: {1: [(268, 275)]}}},
            {MixtureKey({"language": ["english"]}): {2: {0: [(65, 72)]}}},
            {MixtureKey({"language": ["french"], "topic": ["law"]}): {1: {0: [(64, 71)]}}},
            {MixtureKey({"language": ["french"]}): {0: {1: [(275, 282)]}}},
            {MixtureKey({"language": ["english"]}): {2: {0: [(72, 79)]}}},
            {MixtureKey({"language": ["french"], "topic": ["law"]}): {1: {0: [(71, 75)], 1: [(20, 23)]}}},
            {MixtureKey({"language": ["french"]}): {0: {1: [(282, 289)]}}},
            {MixtureKey({"language": ["english"]}): {2: {0: [(79, 80), (150, 156)]}}},
            {MixtureKey({"language": ["french"], "topic": ["law"]}): {1: {1: [(23, 30)]}}},
            {MixtureKey({"language": ["french"]}): {0: {1: [(289, 296)]}}},
            {MixtureKey({"language": ["english"]}): {2: {0: [(156, 163)]}}},
            {MixtureKey({"language": ["french"]}): {0: {1: [(296, 303)]}}},
            {MixtureKey({"language": ["english"]}): {2: {0: [(163, 170)]}}},
            {MixtureKey({"topic": ["law"]}): {1: {0: [(210, 217)]}}},
            {MixtureKey({"language": ["french"]}): {0: {1: [(303, 310)]}}},
            {MixtureKey({"language": ["english"]}): {2: {0: [(170, 177)]}}},
            {MixtureKey({"topic": ["law"]}): {1: {0: [(217, 224)]}}},
            {MixtureKey({"language": ["french"]}): {0: {1: [(310, 317)]}}},
            {MixtureKey({"language": ["english"]}): {2: {0: [(177, 184)]}}},
            {MixtureKey({"topic": ["law"]}): {1: {0: [(224, 231)]}}},
            {MixtureKey({"language": ["french"]}): {0: {1: [(317, 324)]}}},
            {MixtureKey({"language": ["english"]}): {2: {0: [(184, 191)]}}},
            {MixtureKey({"topic": ["law"]}): {1: {0: [(231, 238)]}}},
            {MixtureKey({"language": ["french"]}): {0: {1: [(324, 331)]}}},
            {MixtureKey({"language": ["english"]}): {2: {0: [(191, 198)]}}},
            {MixtureKey({"topic": ["law"]}): {1: {0: [(238, 245)]}}},
            {MixtureKey({"language": ["french"]}): {0: {1: [(331, 338)]}}},
            {MixtureKey({"language": ["english"]}): {2: {0: [(198, 200)]}}},
            {MixtureKey({"topic": ["law"]}): {1: {0: [(245, 250)]}}},
            {MixtureKey({"language": ["french"]}): {0: {1: [(338, 345)]}}},
            {MixtureKey({"language": ["french"]}): {0: {1: [(345, 350)]}, 1: {0: [(40, 42)]}}},
            {MixtureKey({"language": ["french"]}): {1: {0: [(42, 49)]}}},
            {MixtureKey({"language": ["french"]}): {1: {0: [(49, 50), (75, 81)]}}},
            {MixtureKey({"language": ["french"]}): {1: {0: [(81, 88)]}}},
            {MixtureKey({"language": ["french"]}): {1: {0: [(88, 90)], 1: [(0, 5)]}}},
            {MixtureKey({"language": ["french"]}): {1: {1: [(5, 12)]}}},
            {MixtureKey({"language": ["french"]}): {1: {1: [(12, 19)]}}},
            {MixtureKey({"language": ["french"]}): {1: {1: [(19, 20)]}}},
            {MixtureKey({"topic": ["medicine"]}): {1: {1: [(178, 185)]}}},
            {MixtureKey({"topic": ["medicine"]}): {1: {1: [(185, 192)]}}},
            {MixtureKey({"topic": ["medicine"]}): {1: {1: [(192, 199)]}}},
            {MixtureKey({"topic": ["medicine"]}): {1: {1: [(199, 200)]}}},
            {
                MixtureKey({"language": ["english"], "topic": ["law", "medicine"]}): {
                    0: {0: [(178, 180)]},
                    1: {0: [(50, 55)]},
                }
            },
            {MixtureKey({"language": ["english"], "topic": ["law", "medicine"]}): {1: {0: [(55, 60)]}}},
        ]

        query = Query.for_job("job_id").complexmockoperator("test")
        args = QueryExecutionArgs(mixture=ArbitraryMixture(7))

        assert self.client.execute_query(query, args)
        chunks = list(iter(query.results))

        def _subchunk_counter(chunk, key):
            count = 0
            for _0, document_entry in chunk._result_index[key].items():
                for _1, ranges in document_entry.items():
                    for base_range in ranges:
                        count += base_range[1] - base_range[0]
            return count

        for i, chunk in enumerate(chunks):
            self.assertDictEqual(reference_chunks[i], chunk._result_index)

        expected_chunk_count = 175
        expected_error_count = 11

        real_error_count = 0
        for chunk in chunks:
            chunk_count = 0
            for k, _ in chunk._result_index.items():
                chunk_count += _subchunk_counter(chunk, k)

            if chunk_count != 7:
                real_error_count += 1

        self.assertEqual(expected_chunk_count, len(chunks))
        self.assertEqual(expected_error_count, real_error_count)

    @patch("mixtera.core.datacollection.MixteraDataCollection._get_dataset_func_by_id")
    @patch("mixtera.core.datacollection.MixteraDataCollection._get_dataset_type_by_id")
    @patch("mixtera.core.datacollection.MixteraDataCollection._get_file_path_by_id")
    def test_create_chunking_with_simple_dynamic_mixture_single_property(
        self,
        mock_get_file_path_by_id: MagicMock,
        mock_get_dataset_type_by_id: MagicMock,
        mock_get_dataset_func_by_id: MagicMock,
    ):
        mock_get_file_path_by_id.return_value = "test_file_path"
        mock_get_dataset_type_by_id.return_value = "test_dataset_type"
        mock_get_dataset_func_by_id.return_value = lambda x: x

        # This assumes keys are generated using a single (the first) value of a property
        reference_chunks = [
            {
                MixtureKey({"language": ["french"]}): {0: {0: [(100, 112)]}},
                MixtureKey({"language": ["english"]}): {0: {0: [(300, 304)]}},
            },
            {
                MixtureKey({"language": ["french"]}): {0: {0: [(112, 120), (210, 214)]}},
                MixtureKey({"language": ["english"]}): {0: {0: [(304, 308)]}},
            },
            {
                MixtureKey({"language": ["french"]}): {0: {0: [(214, 226)]}},
                MixtureKey({"language": ["english"]}): {0: {0: [(308, 312)]}},
            },
            {
                MixtureKey({"language": ["french"]}): {0: {0: [(226, 238)]}},
                MixtureKey({"language": ["english"]}): {0: {0: [(312, 316)]}},
            },
            {
                MixtureKey({"language": ["french"]}): {0: {0: [(238, 250)]}},
                MixtureKey({"language": ["english"]}): {0: {0: [(316, 320)]}},
            },
            {
                MixtureKey({"language": ["french"]}): {0: {0: [(250, 262)]}},
                MixtureKey({"language": ["english"]}): {0: {0: [(320, 324)]}},
            },
            {
                MixtureKey({"language": ["french"]}): {0: {0: [(262, 274)]}},
                MixtureKey({"language": ["english"]}): {0: {0: [(324, 328)]}},
            },
            {
                MixtureKey({"language": ["french"]}): {0: {0: [(274, 286)]}},
                MixtureKey({"language": ["english"]}): {0: {0: [(328, 332)]}},
            },
            {
                MixtureKey({"language": ["french"]}): {0: {0: [(286, 298)]}},
                MixtureKey({"language": ["english"]}): {0: {0: [(332, 336)]}},
            },
            {
                MixtureKey({"language": ["french"]}): {0: {0: [(298, 300)], 1: [(150, 160)]}},
                MixtureKey({"language": ["english"]}): {0: {0: [(336, 340)]}},
            },
            {
                MixtureKey({"language": ["french"]}): {0: {1: [(170, 182)]}},
                MixtureKey({"language": ["english"]}): {0: {0: [(340, 344)]}},
            },
            {
                MixtureKey({"language": ["french"]}): {0: {1: [(182, 194)]}},
                MixtureKey({"language": ["english"]}): {0: {0: [(344, 348)]}},
            },
            {
                MixtureKey({"language": ["french"]}): {0: {1: [(194, 200), (250, 256)]}},
                MixtureKey({"language": ["english"]}): {0: {0: [(348, 352)]}},
            },
            {
                MixtureKey({"language": ["french"]}): {0: {1: [(256, 268)]}},
                MixtureKey({"language": ["english"]}): {0: {0: [(352, 356)]}},
            },
            {
                MixtureKey({"language": ["french"]}): {0: {1: [(268, 280)]}},
                MixtureKey({"language": ["english"]}): {0: {0: [(356, 360)]}},
            },
            {
                MixtureKey({"language": ["french"]}): {0: {1: [(280, 292)]}},
                MixtureKey({"language": ["english"]}): {0: {0: [(360, 364)]}},
            },
            {
                MixtureKey({"language": ["french"]}): {0: {1: [(292, 304)]}},
                MixtureKey({"language": ["english"]}): {0: {0: [(364, 368)]}},
            },
            {
                MixtureKey({"language": ["french"]}): {0: {1: [(304, 316)]}},
                MixtureKey({"language": ["english"]}): {0: {0: [(368, 372)]}},
            },
            {
                MixtureKey({"language": ["french"]}): {0: {1: [(316, 328)]}},
                MixtureKey({"language": ["english"]}): {0: {0: [(372, 376)]}},
            },
            {
                MixtureKey({"language": ["french"]}): {0: {1: [(328, 340)]}},
                MixtureKey({"language": ["english"]}): {0: {0: [(376, 380)]}},
            },
            {
                MixtureKey({"language": ["french"]}): {0: {1: [(340, 350)]}, 1: {0: [(40, 42)]}},
                MixtureKey({"language": ["english"]}): {0: {0: [(380, 384)]}},
            },
            {
                MixtureKey({"language": ["french"]}): {1: {0: [(42, 50), (75, 79)]}},
                MixtureKey({"language": ["english"]}): {0: {0: [(384, 388)]}},
            },
            {
                MixtureKey({"language": ["french"]}): {1: {0: [(79, 90)], 1: [(0, 1)]}},
                MixtureKey({"language": ["english"]}): {0: {0: [(388, 392)]}},
            },
            {
                MixtureKey({"language": ["french"]}): {1: {1: [(1, 13)]}},
                MixtureKey({"language": ["english"]}): {0: {0: [(392, 396)]}},
            },
            {
                MixtureKey({"language": ["french"]}): {1: {1: [(13, 20)]}, 0: {0: [(120, 125)]}},
                MixtureKey({"language": ["english"]}): {0: {0: [(396, 400)]}},
            },
            {
                MixtureKey({"language": ["french"]}): {0: {1: [(160, 170)]}, 1: {1: [(30, 32)]}},
                MixtureKey({"language": ["english"]}): {0: {2: [(10, 14)]}},
            },
            {
                MixtureKey({"language": ["french"]}): {1: {1: [(32, 44)]}},
                MixtureKey({"language": ["english"]}): {0: {2: [(14, 18)]}},
            },
            {
                MixtureKey({"language": ["french"]}): {1: {1: [(44, 50)]}, 0: {0: [(0, 6)]}},
                MixtureKey({"language": ["english"]}): {0: {2: [(18, 20)]}, 2: {0: [(0, 2)]}},
            },
            {
                MixtureKey({"language": ["french"]}): {0: {0: [(6, 18)]}},
                MixtureKey({"language": ["english"]}): {2: {0: [(2, 6)]}},
            },
            {
                MixtureKey({"language": ["french"]}): {0: {0: [(18, 25)], 1: [(210, 215)]}},
                MixtureKey({"language": ["english"]}): {2: {0: [(6, 10)]}},
            },
            {
                MixtureKey({"language": ["french"]}): {0: {1: [(215, 227)]}},
                MixtureKey({"language": ["english"]}): {2: {0: [(10, 14)]}},
            },
            {
                MixtureKey({"language": ["french"]}): {0: {1: [(227, 239)]}},
                MixtureKey({"language": ["english"]}): {2: {0: [(14, 18)]}},
            },
            {
                MixtureKey({"language": ["french"]}): {0: {1: [(239, 250)]}, 1: {0: [(25, 26)]}},
                MixtureKey({"language": ["english"]}): {2: {0: [(18, 22)]}},
            },
            {
                MixtureKey({"language": ["french"]}): {1: {0: [(26, 38)]}},
                MixtureKey({"language": ["english"]}): {2: {0: [(22, 26)]}},
            },
            {
                MixtureKey({"language": ["french"]}): {1: {0: [(38, 40), (60, 70)]}},
                MixtureKey({"language": ["english"]}): {2: {0: [(26, 30)]}},
            },
            {
                MixtureKey({"language": ["french"]}): {1: {0: [(70, 75)], 1: [(20, 27)]}},
                MixtureKey({"language": ["english"]}): {2: {0: [(30, 34)]}},
            },
            {
                MixtureKey({"language": ["french"]}): {1: {1: [(27, 30)], 0: [(90, 99)]}},
                MixtureKey({"language": ["english"]}): {2: {0: [(34, 38)]}},
            },
            {
                MixtureKey({"language": ["french"]}): {1: {0: [(99, 100)]}, 0: {0: [(125, 136)]}},
                MixtureKey({"language": ["english"]}): {2: {0: [(38, 42)]}},
            },
            {
                MixtureKey({"language": ["french"]}): {0: {0: [(136, 140)], 1: [(200, 208)]}},
                MixtureKey({"language": ["english"]}): {2: {0: [(42, 46)]}},
            },
            {
                MixtureKey({"language": ["french"]}): {0: {1: [(208, 210)], 0: [(200, 210)]}},
                MixtureKey({"language": ["english"]}): {2: {0: [(46, 50)]}},
            },
            {
                MixtureKey({"language": ["french"]}): {0: {0: [(25, 37)]}},
                MixtureKey({"language": ["english"]}): {2: {0: [(50, 54)]}},
            },
            {
                MixtureKey({"language": ["french"]}): {0: {0: [(37, 49)]}},
                MixtureKey({"language": ["english"]}): {2: {0: [(54, 58)]}},
            },
            {
                MixtureKey({"language": ["french"]}): {0: {0: [(49, 50), (140, 150)], 1: [(100, 101)]}},
                MixtureKey({"language": ["english"]}): {2: {0: [(58, 62)]}},
            },
            {
                MixtureKey({"language": ["french"]}): {0: {1: [(101, 113)]}},
                MixtureKey({"language": ["english"]}): {2: {0: [(62, 66)]}},
            },
            {
                MixtureKey({"language": ["french"]}): {0: {1: [(113, 125)]}},
                MixtureKey({"language": ["english"]}): {2: {0: [(66, 70)]}},
            },
            {
                MixtureKey({"language": ["french"]}): {0: {1: [(125, 137)]}},
                MixtureKey({"language": ["english"]}): {2: {0: [(70, 74)]}},
            },
            {
                MixtureKey({"language": ["french"]}): {0: {1: [(137, 149)]}},
                MixtureKey({"language": ["english"]}): {2: {0: [(74, 78)]}},
            },
        ]

        mixture_concentration = {
            MixtureKey({"language": ["french"]}): 0.75,  # 12 instances per batch
            MixtureKey({"language": ["english"]}): 0.25,  # 4 instances per batch
        }
        mixture = StaticMixture(16, mixture_concentration)

        query = Query.for_job("job_id").complexmockoperator("test")
        args = QueryExecutionArgs(mixture=mixture)
        assert self.client.execute_query(query, args)
        chunks = list(iter(query.results))

        for i, chunk in enumerate(chunks):
            self.assertDictEqual(reference_chunks[i], chunk._result_index)

    @patch("mixtera.core.datacollection.MixteraDataCollection._get_dataset_func_by_id")
    @patch("mixtera.core.datacollection.MixteraDataCollection._get_dataset_type_by_id")
    @patch("mixtera.core.datacollection.MixteraDataCollection._get_file_path_by_id")
    def test_flexible_chunking(
        self,
        mock_get_file_path_by_id: MagicMock,
        mock_get_dataset_type_by_id: MagicMock,
        mock_get_dataset_func_by_id: MagicMock,
    ):
        mock_get_file_path_by_id.return_value = "test_file_path"
        mock_get_dataset_type_by_id.return_value = "test_dataset_type"
        mock_get_dataset_func_by_id.return_value = lambda x: x

        # Note the order here does not really matter and is just a result of the ordering when generating the chunk
        # Note that we cannot fuse those intervals here because they come from different property combinations
        reference_chunks = [
            {
                MixtureKey({"language": ["english"]}): {0: {0: [(5, 10), (0, 5)]}},
            },
        ]

        reference_chunker_index = {
            MixtureKey({"language": ["english", "french"]}): {0: {0: [[0, 5]]}},
            MixtureKey({"language": ["english", "german"]}): {0: {0: [[5, 10]]}},
        }

        mixture_concentration = {
            MixtureKey({"language": ["english"]}): 1,
        }

        query = Query.for_job("job_id").flexiblechunkingtestmockoperator("test")
        mixture = StaticMixture(10, mixture_concentration)
        args = QueryExecutionArgs(mixture=mixture)

        assert self.client.execute_query(query, args)
        chunks = list(iter(query.results))

        # Check the structure of the chunker index
        inverted_index = query.results._invert_result(query.results.results)
        chunker_index = defaultdict_to_dict(query.results._create_chunker_index(inverted_index))
        self.assertDictEqual(defaultdict_to_dict(chunker_index), reference_chunker_index)

        # Check the equality of the chunks
        for i, chunk in enumerate(chunks):
            self.assertDictEqual(reference_chunks[i], chunk._result_index)

        self.assertEqual(len(chunks), 1)


if __name__ == "__main__":
    unittest.main()
