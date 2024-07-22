import unittest
from unittest.mock import MagicMock, patch

import portion as P
from mixtera.core.client import MixteraClient
from mixtera.core.client.mixtera_client import QueryExecutionArgs
from mixtera.core.datacollection.index.index_collection import IndexFactory, IndexTypes
from mixtera.core.query import ArbitraryMixture, Operator, Query, QueryPlan, StaticMixture
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


Query.register(MockOperator)
Query.register(SimpleMockOperator)
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
        self.assertEqual(chunks, [{"field:value": {"did": {"fid": [(0, 2)]}}}])

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
            "language:french;topic:law": {
                0: {0: [[0, 25], [25, 50], [140, 150], [125, 140]], 1: [[100, 150], [200, 210], [210, 250]]},
                1: {0: [[25, 40], [60, 75]], 1: [[20, 30]]},
            },
            "language:english;topic:medicine": {
                0: {0: [[50, 75], [180, 200]], 1: [[50, 100]]},
                1: {0: [[100, 110], [130, 150]]},
            },
            "topic:medicine": {0: {0: [[80, 100]]}, 1: {1: [[50, 100], [150, 200]]}},
            "language:french": {
                0: {0: [[100, 120], [210, 300], [200, 210]], 1: [[150, 160], [170, 200], [250, 350]]},
                1: {0: [[40, 50], [75, 90]], 1: [[0, 20]]},
            },
            "language:french;topic:medicine": {
                0: {0: [[120, 125]], 1: [[160, 170]]},
                1: {0: [[90, 100]], 1: [[30, 50]]},
            },
            "language:english;topic:law": {0: {0: [[150, 180]]}, 1: {0: [[50, 60]]}, 2: {0: [[80, 100]]}},
            "language:english": {0: {0: [[300, 400]], 2: [[10, 20]]}, 2: {0: [[0, 80], [150, 200]]}},
            "topic:law": {1: {0: [[0, 25], [200, 250]]}},
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
            {"language:french": {0: {0: [(0, 12)]}}, "language:english": {0: {0: [(50, 54)]}}},
            {"language:french": {0: {0: [(12, 24)]}}, "language:english": {0: {0: [(54, 58)]}}},
            {"language:french": {0: {0: [(24, 36)]}}, "language:english": {0: {0: [(58, 62)]}}},
            {"language:french": {0: {0: [(36, 48)]}}, "language:english": {0: {0: [(62, 66)]}}},
            {"language:french": {0: {0: [(48, 50), (150, 160)]}}, "language:english": {0: {0: [(66, 70)]}}},
            {"language:french": {0: {0: [(160, 172)]}}, "language:english": {0: {0: [(70, 74)]}}},
            {"language:french": {0: {0: [(172, 184)]}}, "language:english": {0: {0: [(74, 78)]}}},
            {"language:french": {0: {0: [(184, 196)]}}, "language:english": {0: {0: [(78, 82)]}}},
            {"language:french": {0: {0: [(196, 200)], 1: [(0, 8)]}}, "language:english": {0: {0: [(82, 86)]}}},
            {"language:french": {0: {1: [(8, 20)]}}, "language:english": {0: {0: [(86, 90)]}}},
            {"language:french": {0: {1: [(20, 32)]}}, "language:english": {0: {0: [(90, 94)]}}},
            {"language:french": {0: {1: [(32, 44)]}}, "language:english": {0: {0: [(94, 98)]}}},
            {"language:french": {0: {1: [(44, 56)]}}, "language:english": {0: {0: [(98, 100), (100, 102)]}}},
            {"language:french": {0: {1: [(56, 68)]}}, "language:english": {0: {0: [(102, 106)]}}},
            {"language:french": {0: {1: [(68, 80)]}}, "language:english": {0: {0: [(106, 110)]}}},
            {"language:french": {0: {1: [(80, 92)]}}, "language:english": {0: {0: [(110, 114)]}}},
            {"language:french": {0: {1: [(92, 100)]}}, "language:english": {0: {0: [(114, 118)]}}},
        ]

        reference_chunker_index = {
            "language:french": {0: {0: [[0, 50], [150, 200]], 1: [[0, 100]]}},
            "language:english": {0: {0: [[50, 100], [100, 150]]}},
        }

        mixture_concentration = {
            "language:french": 0.75,  # 12 instances per batch
            "language:english": 0.25,  # 4 instances per batch
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
            self.assertDictEqual(reference_chunks[i], chunk)

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
            # Mixture should contain 12 'language:french' instances and 4 'language:english' instances
            {"language:french": {0: {0: [(0, 12)]}}, "language:english": {0: {0: [(50, 54)]}}},
            {"language:french": {0: {0: [(12, 24)]}}, "language:english": {0: {0: [(54, 58)]}}},
            {"language:french": {0: {0: [(24, 36)]}}, "language:english": {0: {0: [(58, 62)]}}},
            {"language:french": {0: {0: [(36, 48)]}}, "language:english": {0: {0: [(62, 66)]}}},
            {"language:french": {0: {0: [(48, 50), (150, 160)]}}, "language:english": {0: {0: [(66, 70)]}}},
            {"language:french": {0: {0: [(160, 172)]}}, "language:english": {0: {0: [(70, 74)]}}},
            {"language:french": {0: {0: [(172, 184)]}}, "language:english": {0: {0: [(74, 78)]}}},
            {"language:french": {0: {0: [(184, 196)]}}, "language:english": {0: {0: [(78, 82)]}}},
            {"language:french": {0: {0: [(196, 200)], 1: [(0, 8)]}}, "language:english": {0: {0: [(82, 86)]}}},
            {"language:french": {0: {1: [(8, 20)]}}, "language:english": {0: {0: [(86, 90)]}}},
            # Mixture should contain 8 'language:french' instances and 8 'language:english' instances
            {"language:french": {0: {1: [(20, 28)]}}, "language:english": {0: {0: [(90, 98)]}}},
            {"language:french": {0: {1: [(28, 36)]}}, "language:english": {0: {0: [(98, 100), (100, 106)]}}},
            {"language:french": {0: {1: [(36, 44)]}}, "language:english": {0: {0: [(106, 114)]}}},
            {"language:french": {0: {1: [(44, 52)]}}, "language:english": {0: {0: [(114, 122)]}}},
            # Mixture should contain 10 'language:french' instances and 10 'language:english' instances
            {"language:french": {0: {1: [(52, 62)]}}, "language:english": {0: {0: [(122, 132)]}}},
            {"language:french": {0: {1: [(62, 72)]}}, "language:english": {0: {0: [(132, 142)]}}},
            {"language:french": {0: {1: [(72, 82)]}}, "language:english": {0: {0: [(142, 150)]}}},
        ]

        mixture_concentration_1 = {
            "language:french": 0.75,  # 12 instances per batch
            "language:english": 0.25,  # 4 instances per batch
        }

        mixture_concentration_2 = {
            "language:french": 0.5,  # 8 and 10 instances per batch
            "language:english": 0.5,  # 8 and 10 instances per batch
        }

        query = Query.for_job("job_id").simplemockoperator("test")

        mixture_1 = StaticMixture(16, mixture_concentration_1)
        mixture_2 = StaticMixture(16, mixture_concentration_2)
        mixture_3 = StaticMixture(20, mixture_concentration_2)
        args = QueryExecutionArgs(mixture=mixture_1)

        assert self.client.execute_query(query, args)
        result_iterator = iter(query.results)
        chunks = [next(result_iterator) for _ in range(10)]
        query.results.update_mixture(mixture_2)
        chunks.extend([next(result_iterator) for _ in range(4)])
        query.results.update_mixture(mixture_3)
        chunks.extend([next(result_iterator) for _ in range(3)])
        self.assertRaises(StopIteration, next, result_iterator)

        # Check the equality of the chunks
        for i, chunk in enumerate(chunks):
            self.assertDictEqual(reference_chunks[i], chunk)

    @patch("mixtera.core.datacollection.MixteraDataCollection._get_dataset_func_by_id")
    @patch("mixtera.core.datacollection.MixteraDataCollection._get_dataset_type_by_id")
    @patch("mixtera.core.datacollection.MixteraDataCollection._get_file_path_by_id")
    def test_create_chunking_with_mixture(
        self,
        mock_get_file_path_by_id: MagicMock,
        mock_get_dataset_type_by_id: MagicMock,
        mock_get_dataset_func_by_id: MagicMock,
    ):
        mock_get_file_path_by_id.return_value = "test_file_path"
        mock_get_dataset_type_by_id.return_value = "test_dataset_type"
        mock_get_dataset_func_by_id.return_value = lambda x: x

        reference_chunks = [
            {"language:french;topic:law": {0: {0: [(0, 6)]}}, "language:english;topic:medicine": {0: {0: [(50, 54)]}}},
            {"language:french;topic:law": {0: {0: [(6, 12)]}}, "language:english;topic:medicine": {0: {0: [(54, 58)]}}},
            {
                "language:french;topic:law": {0: {0: [(12, 18)]}},
                "language:english;topic:medicine": {0: {0: [(58, 62)]}},
            },
            {
                "language:french;topic:law": {0: {0: [(18, 24)]}},
                "language:english;topic:medicine": {0: {0: [(62, 66)]}},
            },
            {
                "language:french;topic:law": {0: {0: [(24, 25), (25, 30)]}},
                "language:english;topic:medicine": {0: {0: [(66, 70)]}},
            },
            {
                "language:french;topic:law": {0: {0: [(30, 36)]}},
                "language:english;topic:medicine": {0: {0: [(70, 74)]}},
            },
            {
                "language:french;topic:law": {0: {0: [(36, 42)]}},
                "language:english;topic:medicine": {0: {0: [(74, 75), (180, 183)]}},
            },
            {
                "language:french;topic:law": {0: {0: [(42, 48)]}},
                "language:english;topic:medicine": {0: {0: [(183, 187)]}},
            },
            {
                "language:french;topic:law": {0: {0: [(48, 50), (140, 144)]}},
                "language:english;topic:medicine": {0: {0: [(187, 191)]}},
            },
            {
                "language:french;topic:law": {0: {0: [(144, 150)]}},
                "language:english;topic:medicine": {0: {0: [(191, 195)]}},
            },
            {
                "language:french;topic:law": {0: {0: [(125, 131)]}},
                "language:english;topic:medicine": {0: {0: [(195, 199)]}},
            },
            {
                "language:french;topic:law": {0: {0: [(131, 137)]}},
                "language:english;topic:medicine": {0: {0: [(199, 200)], 1: [(50, 53)]}},
            },
            {
                "language:french;topic:law": {0: {0: [(137, 140)], 1: [(100, 103)]}},
                "language:english;topic:medicine": {0: {1: [(53, 57)]}},
            },
            {
                "language:french;topic:law": {0: {1: [(103, 109)]}},
                "language:english;topic:medicine": {0: {1: [(57, 61)]}},
            },
            {
                "language:french;topic:law": {0: {1: [(109, 115)]}},
                "language:english;topic:medicine": {0: {1: [(61, 65)]}},
            },
            {
                "language:french;topic:law": {0: {1: [(115, 121)]}},
                "language:english;topic:medicine": {0: {1: [(65, 69)]}},
            },
            {
                "language:french;topic:law": {0: {1: [(121, 127)]}},
                "language:english;topic:medicine": {0: {1: [(69, 73)]}},
            },
            {
                "language:french;topic:law": {0: {1: [(127, 133)]}},
                "language:english;topic:medicine": {0: {1: [(73, 77)]}},
            },
            {
                "language:french;topic:law": {0: {1: [(133, 139)]}},
                "language:english;topic:medicine": {0: {1: [(77, 81)]}},
            },
            {
                "language:french;topic:law": {0: {1: [(139, 145)]}},
                "language:english;topic:medicine": {0: {1: [(81, 85)]}},
            },
            {
                "language:french;topic:law": {0: {1: [(145, 150), (200, 201)]}},
                "language:english;topic:medicine": {0: {1: [(85, 89)]}},
            },
            {
                "language:french;topic:law": {0: {1: [(201, 207)]}},
                "language:english;topic:medicine": {0: {1: [(89, 93)]}},
            },
            {
                "language:french;topic:law": {0: {1: [(207, 210), (210, 213)]}},
                "language:english;topic:medicine": {0: {1: [(93, 97)]}},
            },
            {
                "language:french;topic:law": {0: {1: [(213, 219)]}},
                "language:english;topic:medicine": {0: {1: [(97, 100)]}, 1: {0: [(100, 101)]}},
            },
            {
                "language:french;topic:law": {0: {1: [(219, 225)]}},
                "language:english;topic:medicine": {1: {0: [(101, 105)]}},
            },
            {
                "language:french;topic:law": {0: {1: [(225, 231)]}},
                "language:english;topic:medicine": {1: {0: [(105, 109)]}},
            },
            {
                "language:french;topic:law": {0: {1: [(231, 237)]}},
                "language:english;topic:medicine": {1: {0: [(109, 110), (130, 133)]}},
            },
            {
                "language:french;topic:law": {0: {1: [(237, 243)]}},
                "language:english;topic:medicine": {1: {0: [(133, 137)]}},
            },
            {
                "language:french;topic:law": {0: {1: [(243, 249)]}},
                "language:english;topic:medicine": {1: {0: [(137, 141)]}},
            },
            {
                "language:french;topic:law": {0: {1: [(249, 250)]}, 1: {0: [(25, 30)]}},
                "language:english;topic:medicine": {1: {0: [(141, 145)]}},
            },
            {
                "language:french;topic:law": {1: {0: [(30, 36)]}},
                "language:english;topic:medicine": {1: {0: [(145, 149)]}},
            },
            {
                "language:french;topic:law": {1: {0: [(36, 40), (60, 62)]}},
                "language:english;topic:medicine": {1: {0: [(149, 150)]}},
            },
        ]

        mixture_concentration = {
            "language:french;topic:law": 0.6,  # 6 instances per batch
            "language:english;topic:medicine": 0.4,  # 4 instances per batch
        }

        query = Query.for_job("job_id").complexmockoperator("test")

        mixture = StaticMixture(10, mixture_concentration)
        args = QueryExecutionArgs(mixture=mixture)
        assert self.client.execute_query(query, args)
        chunks = list(iter(query.results))

        def _subchunk_counter(chunk, key):
            count = 0
            for _0, document_entry in chunk[key].items():
                for _1, ranges in document_entry.items():
                    for base_range in ranges:
                        count += base_range[1] - base_range[0]
            return count

        expected_chunk_count = 32
        expected_error_count_s1 = 0
        expected_error_count_s2 = 1

        real_error_count_s1 = 0
        real_error_count_s2 = 0

        for chunk in chunks:
            subchunk_1_count = _subchunk_counter(chunk, "language:french;topic:law")
            subchunk_2_count = _subchunk_counter(chunk, "language:english;topic:medicine")

            if subchunk_1_count != 6:
                real_error_count_s1 += 1

            if subchunk_2_count != 4:
                real_error_count_s2 += 1

        self.assertEqual(expected_chunk_count, len(chunks))
        self.assertEqual(expected_error_count_s1, real_error_count_s1)
        self.assertEqual(expected_error_count_s2, real_error_count_s2)
        for i, chunk in enumerate(chunks):
            self.assertDictEqual(reference_chunks[i], chunk)

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
            {"language:french;topic:law": {0: {0: [(0, 7)]}}},
            {"language:french;topic:law": {0: {0: [(7, 14)]}}},
            {"language:french;topic:law": {0: {0: [(14, 21)]}}},
            {"language:french;topic:law": {0: {0: [(21, 25), (25, 28)]}}},
            {"language:french;topic:law": {0: {0: [(28, 35)]}}},
            {"language:french;topic:law": {0: {0: [(35, 42)]}}},
            {"language:french;topic:law": {0: {0: [(42, 49)]}}},
            {"language:french;topic:law": {0: {0: [(49, 50), (140, 146)]}}},
            {"language:french;topic:law": {0: {0: [(146, 150), (125, 128)]}}},
            {"language:french;topic:law": {0: {0: [(128, 135)]}}},
            {"language:french;topic:law": {0: {0: [(135, 140)], 1: [(100, 102)]}}},
            {"language:french;topic:law": {0: {1: [(102, 109)]}}},
            {"language:french;topic:law": {0: {1: [(109, 116)]}}},
            {"language:french;topic:law": {0: {1: [(116, 123)]}}},
            {"language:french;topic:law": {0: {1: [(123, 130)]}}},
            {"language:french;topic:law": {0: {1: [(130, 137)]}}},
            {"language:french;topic:law": {0: {1: [(137, 144)]}}},
            {"language:french;topic:law": {0: {1: [(144, 150), (200, 201)]}}},
            {"language:french;topic:law": {0: {1: [(201, 208)]}}},
            {"language:french;topic:law": {0: {1: [(208, 210), (210, 215)]}}},
            {"language:french;topic:law": {0: {1: [(215, 222)]}}},
            {"language:french;topic:law": {0: {1: [(222, 229)]}}},
            {"language:french;topic:law": {0: {1: [(229, 236)]}}},
            {"language:french;topic:law": {0: {1: [(236, 243)]}}},
            {"language:french;topic:law": {0: {1: [(243, 250)]}}},
            {"language:french;topic:law": {1: {0: [(25, 32)]}}},
            {"language:french;topic:law": {1: {0: [(32, 39)]}}},
            {"language:french;topic:law": {1: {0: [(39, 40), (60, 66)]}}},
            {"language:french;topic:law": {1: {0: [(66, 73)]}}},
            {"language:french;topic:law": {1: {0: [(73, 75)], 1: [(20, 25)]}}},
            {"language:french;topic:law": {1: {1: [(25, 30)]}}},
            {"language:english;topic:medicine": {0: {0: [(50, 57)]}}},
            {"language:english;topic:medicine": {0: {0: [(57, 64)]}}},
            {"language:english;topic:medicine": {0: {0: [(64, 71)]}}},
            {"language:english;topic:medicine": {0: {0: [(71, 75), (180, 183)]}}},
            {"language:english;topic:medicine": {0: {0: [(183, 190)]}}},
            {"language:english;topic:medicine": {0: {0: [(190, 197)]}}},
            {"language:english;topic:medicine": {0: {0: [(197, 200)], 1: [(50, 54)]}}},
            {"language:english;topic:medicine": {0: {1: [(54, 61)]}}},
            {"language:english;topic:medicine": {0: {1: [(61, 68)]}}},
            {"language:english;topic:medicine": {0: {1: [(68, 75)]}}},
            {"language:english;topic:medicine": {0: {1: [(75, 82)]}}},
            {"language:english;topic:medicine": {0: {1: [(82, 89)]}}},
            {"language:english;topic:medicine": {0: {1: [(89, 96)]}}},
            {"language:english;topic:medicine": {0: {1: [(96, 100)]}, 1: {0: [(100, 103)]}}},
            {"language:english;topic:medicine": {1: {0: [(103, 110)]}}},
            {"language:english;topic:medicine": {1: {0: [(130, 137)]}}},
            {"language:english;topic:medicine": {1: {0: [(137, 144)]}}},
            {"language:english;topic:medicine": {1: {0: [(144, 150)]}}},
            {"topic:medicine": {0: {0: [(80, 87)]}}},
            {"topic:medicine": {0: {0: [(87, 94)]}}},
            {"topic:medicine": {0: {0: [(94, 100)]}, 1: {1: [(50, 51)]}}},
            {"topic:medicine": {1: {1: [(51, 58)]}}},
            {"topic:medicine": {1: {1: [(58, 65)]}}},
            {"topic:medicine": {1: {1: [(65, 72)]}}},
            {"topic:medicine": {1: {1: [(72, 79)]}}},
            {"topic:medicine": {1: {1: [(79, 86)]}}},
            {"topic:medicine": {1: {1: [(86, 93)]}}},
            {"topic:medicine": {1: {1: [(93, 100)]}}},
            {"topic:medicine": {1: {1: [(150, 157)]}}},
            {"topic:medicine": {1: {1: [(157, 164)]}}},
            {"topic:medicine": {1: {1: [(164, 171)]}}},
            {"topic:medicine": {1: {1: [(171, 178)]}}},
            {"topic:medicine": {1: {1: [(178, 185)]}}},
            {"topic:medicine": {1: {1: [(185, 192)]}}},
            {"topic:medicine": {1: {1: [(192, 199)]}}},
            {"topic:medicine": {1: {1: [(199, 200)]}}},
            {"language:french": {0: {0: [(100, 107)]}}},
            {"language:french": {0: {0: [(107, 114)]}}},
            {"language:french": {0: {0: [(114, 120), (210, 211)]}}},
            {"language:french": {0: {0: [(211, 218)]}}},
            {"language:french": {0: {0: [(218, 225)]}}},
            {"language:french": {0: {0: [(225, 232)]}}},
            {"language:french": {0: {0: [(232, 239)]}}},
            {"language:french": {0: {0: [(239, 246)]}}},
            {"language:french": {0: {0: [(246, 253)]}}},
            {"language:french": {0: {0: [(253, 260)]}}},
            {"language:french": {0: {0: [(260, 267)]}}},
            {"language:french": {0: {0: [(267, 274)]}}},
            {"language:french": {0: {0: [(274, 281)]}}},
            {"language:french": {0: {0: [(281, 288)]}}},
            {"language:french": {0: {0: [(288, 295)]}}},
            {"language:french": {0: {0: [(295, 300), (200, 202)]}}},
            {"language:french": {0: {0: [(202, 209)]}}},
            {"language:french": {0: {0: [(209, 210)], 1: [(150, 156)]}}},
            {"language:french": {0: {1: [(156, 160), (170, 173)]}}},
            {"language:french": {0: {1: [(173, 180)]}}},
            {"language:french": {0: {1: [(180, 187)]}}},
            {"language:french": {0: {1: [(187, 194)]}}},
            {"language:french": {0: {1: [(194, 200), (250, 251)]}}},
            {"language:french": {0: {1: [(251, 258)]}}},
            {"language:french": {0: {1: [(258, 265)]}}},
            {"language:french": {0: {1: [(265, 272)]}}},
            {"language:french": {0: {1: [(272, 279)]}}},
            {"language:french": {0: {1: [(279, 286)]}}},
            {"language:french": {0: {1: [(286, 293)]}}},
            {"language:french": {0: {1: [(293, 300)]}}},
            {"language:french": {0: {1: [(300, 307)]}}},
            {"language:french": {0: {1: [(307, 314)]}}},
            {"language:french": {0: {1: [(314, 321)]}}},
            {"language:french": {0: {1: [(321, 328)]}}},
            {"language:french": {0: {1: [(328, 335)]}}},
            {"language:french": {0: {1: [(335, 342)]}}},
            {"language:french": {0: {1: [(342, 349)]}}},
            {"language:french": {0: {1: [(349, 350)]}, 1: {0: [(40, 46)]}}},
            {"language:french": {1: {0: [(46, 50), (75, 78)]}}},
            {"language:french": {1: {0: [(78, 85)]}}},
            {"language:french": {1: {0: [(85, 90)], 1: [(0, 2)]}}},
            {"language:french": {1: {1: [(2, 9)]}}},
            {"language:french": {1: {1: [(9, 16)]}}},
            {"language:french": {1: {1: [(16, 20)]}}},
            {"language:french;topic:medicine": {0: {0: [(120, 125)], 1: [(160, 162)]}}},
            {"language:french;topic:medicine": {0: {1: [(162, 169)]}}},
            {"language:french;topic:medicine": {0: {1: [(169, 170)]}, 1: {0: [(90, 96)]}}},
            {"language:french;topic:medicine": {1: {0: [(96, 100)], 1: [(30, 33)]}}},
            {"language:french;topic:medicine": {1: {1: [(33, 40)]}}},
            {"language:french;topic:medicine": {1: {1: [(40, 47)]}}},
            {"language:french;topic:medicine": {1: {1: [(47, 50)]}}},
            {"language:english;topic:law": {0: {0: [(150, 157)]}}},
            {"language:english;topic:law": {0: {0: [(157, 164)]}}},
            {"language:english;topic:law": {0: {0: [(164, 171)]}}},
            {"language:english;topic:law": {0: {0: [(171, 178)]}}},
            {"language:english;topic:law": {0: {0: [(178, 180)]}, 1: {0: [(50, 55)]}}},
            {"language:english;topic:law": {1: {0: [(55, 60)]}, 2: {0: [(80, 82)]}}},
            {"language:english;topic:law": {2: {0: [(82, 89)]}}},
            {"language:english;topic:law": {2: {0: [(89, 96)]}}},
            {"language:english;topic:law": {2: {0: [(96, 100)]}}},
            {"language:english": {0: {0: [(300, 307)]}}},
            {"language:english": {0: {0: [(307, 314)]}}},
            {"language:english": {0: {0: [(314, 321)]}}},
            {"language:english": {0: {0: [(321, 328)]}}},
            {"language:english": {0: {0: [(328, 335)]}}},
            {"language:english": {0: {0: [(335, 342)]}}},
            {"language:english": {0: {0: [(342, 349)]}}},
            {"language:english": {0: {0: [(349, 356)]}}},
            {"language:english": {0: {0: [(356, 363)]}}},
            {"language:english": {0: {0: [(363, 370)]}}},
            {"language:english": {0: {0: [(370, 377)]}}},
            {"language:english": {0: {0: [(377, 384)]}}},
            {"language:english": {0: {0: [(384, 391)]}}},
            {"language:english": {0: {0: [(391, 398)]}}},
            {"language:english": {0: {0: [(398, 400)], 2: [(10, 15)]}}},
            {"language:english": {0: {2: [(15, 20)]}, 2: {0: [(0, 2)]}}},
            {"language:english": {2: {0: [(2, 9)]}}},
            {"language:english": {2: {0: [(9, 16)]}}},
            {"language:english": {2: {0: [(16, 23)]}}},
            {"language:english": {2: {0: [(23, 30)]}}},
            {"language:english": {2: {0: [(30, 37)]}}},
            {"language:english": {2: {0: [(37, 44)]}}},
            {"language:english": {2: {0: [(44, 51)]}}},
            {"language:english": {2: {0: [(51, 58)]}}},
            {"language:english": {2: {0: [(58, 65)]}}},
            {"language:english": {2: {0: [(65, 72)]}}},
            {"language:english": {2: {0: [(72, 79)]}}},
            {"language:english": {2: {0: [(79, 80), (150, 156)]}}},
            {"language:english": {2: {0: [(156, 163)]}}},
            {"language:english": {2: {0: [(163, 170)]}}},
            {"language:english": {2: {0: [(170, 177)]}}},
            {"language:english": {2: {0: [(177, 184)]}}},
            {"language:english": {2: {0: [(184, 191)]}}},
            {"language:english": {2: {0: [(191, 198)]}}},
            {"language:english": {2: {0: [(198, 200)]}}},
            {"topic:law": {1: {0: [(0, 7)]}}},
            {"topic:law": {1: {0: [(7, 14)]}}},
            {"topic:law": {1: {0: [(14, 21)]}}},
            {"topic:law": {1: {0: [(21, 25), (200, 203)]}}},
            {"topic:law": {1: {0: [(203, 210)]}}},
            {"topic:law": {1: {0: [(210, 217)]}}},
            {"topic:law": {1: {0: [(217, 224)]}}},
            {"topic:law": {1: {0: [(224, 231)]}}},
            {"topic:law": {1: {0: [(231, 238)]}}},
            {"topic:law": {1: {0: [(238, 245)]}}},
            {"topic:law": {1: {0: [(245, 250)]}}},
        ]

        query = Query.for_job("job_id").complexmockoperator("test")
        args = QueryExecutionArgs(mixture=ArbitraryMixture(7))

        assert self.client.execute_query(query, args)
        chunks = list(iter(query.results))

        def _subchunk_counter(chunk, key):
            count = 0
            for _0, document_entry in chunk[key].items():
                for _1, ranges in document_entry.items():
                    for base_range in ranges:
                        count += base_range[1] - base_range[0]
            return count

        expected_chunk_count = 173
        expected_error_count = 8

        real_error_count = 0
        for chunk in chunks:
            chunk_count = 0
            for k, _ in chunk.items():
                chunk_count += _subchunk_counter(chunk, k)

            if chunk_count != 7:
                real_error_count += 1

        self.assertEqual(expected_chunk_count, len(chunks))
        self.assertEqual(expected_error_count, real_error_count)
        for i, chunk in enumerate(chunks):
            self.assertDictEqual(reference_chunks[i], chunk)
