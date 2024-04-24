import unittest
from unittest.mock import MagicMock, patch

import portion as P
from mixtera.core.client import MixteraClient
from mixtera.core.datacollection.index.index_collection import IndexFactory, IndexTypes
from mixtera.core.query import NoopMixture, Operator, Query, QueryPlan
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
        # res = list(query_result)
        # res = [x._index for x in res]
        gt_meta = {
            "dataset_type": {"did": "test_dataset_type"},
            "file_path": {"fid": "test_file_path"},
        }

        # self.assertEqual(
        #     res, [{"field": {"value": {"did": {"fid": [(0, 1)]}}}}, {"field": {"value": {"did": {"fid": [(1, 2)]}}}}]
        # )

        self.assertEqual(query_result.dataset_type, gt_meta["dataset_type"])
        self.assertEqual(query_result.file_path, gt_meta["file_path"])

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
        assert self.client.execute_query(query, 1)
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

        # This assumes keys are generated using all potential property values
        # reference_result = {
        #     "language:french;topic:law": {
        #         0: {0: [[0, 25]], 1: [[210, 250]]},
        #         1: {0: [[25, 40], [60, 75]], 1: [[20, 30]]},
        #     },
        #     "language:french,english;topic:law,medicine": {0: {0: [[25, 50], [140, 150]], 1: [[100, 150]]}},
        #     "language:english;topic:medicine": {
        #         0: {0: [[50, 75], [180, 200]], 1: [[50, 100]]},
        #         1: {0: [[100, 110], [130, 150]]},
        #     },
        #     "topic:medicine": {0: {0: [[80, 100]]}, 1: {1: [[50, 100], [150, 200]]}},
        #     "language:french": {
        #         0: {0: [[100, 120], [210, 300]], 1: [[150, 160], [170, 200], [250, 350]]},
        #         1: {0: [[40, 50], [75, 90]], 1: [[0, 20]]},
        #     },
        #     "language:french;topic:medicine": {0: {0: [[120, 125]], 1: [[160, 170]]}, 1: {1: [[30, 50]]}},
        #     "language:french;topic:law,medicine": {0: {0: [[125, 140]], 1: [[200, 210]]}},
        #     "language:english;topic:law,medicine": {0: {0: [[150, 180]]}, 1: {0: [[50, 60]]}},
        #     "language:french,english": {0: {0: [[200, 210]]}},
        #     "language:english": {0: {0: [[300, 400]], 2: [[10, 20]]}, 2: {0: [[0, 80], [150, 200]]}},
        #     "topic:law": {1: {0: [[0, 25], [200, 250]]}},
        #     "language:french,english;topic:medicine": {1: {0: [[90, 100]]}},
        #     "language:english;topic:law": {2: {0: [[80, 100]]}},
        # }

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
        assert self.client.execute_query(query, 1)
        inverted_index = query.results._invert_result(query.results.results)
        chunk_index = query.results._create_chunker_index(inverted_index)
        print(defaultdict_to_dict(chunk_index))
        self.assertDictEqual(defaultdict_to_dict(chunk_index), reference_result)

    @patch("mixtera.core.datacollection.MixteraDataCollection._get_dataset_func_by_id")
    @patch("mixtera.core.datacollection.MixteraDataCollection._get_dataset_type_by_id")
    @patch("mixtera.core.datacollection.MixteraDataCollection._get_file_path_by_id")
    def test_create_chunking(
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

        mixture = NoopMixture(10, mixture_concentration)
        assert self.client.execute_query(query, mixture=mixture)
        chunks = query.results._chunks

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
        for i in range(len(chunks)):
            self.assertDictEqual(reference_chunks[i], chunks[i])
