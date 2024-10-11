import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
from mixtera.core.client import MixteraClient
from mixtera.core.query import ArbitraryMixture, Query
from mixtera.core.query.mixture import InferringMixture, MixtureKey, StaticMixture
from mixtera.core.query.query_result import QueryResult
from mixtera.utils.utils import defaultdict_to_dict


class TestQueryResult(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        self.directory = Path(self.temp_dir.name)
        self.client = MixteraClient.from_directory(self.directory)
        self.query = Query("job_id")

    def tearDown(self):
        self.temp_dir.cleanup()

    def create_simple_df(self):
        return pl.DataFrame(
            {
                "dataset_id": [0, 0, 0, 0, 0],
                "file_id": [0, 0, 0, 0, 1],
                "interval_start": [0, 50, 100, 150, 0],
                "interval_end": [50, 100, 150, 200, 100],
                "language": [["french"], ["english", "french"], ["english"], ["french"], ["french"]],
            }
        )

    def create_complex_df(self):
        return pl.DataFrame(
            {
                "dataset_id": [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    2,
                    2,
                    2,
                ],
                "file_id": [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    2,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    1,
                    1,
                    1,
                    0,
                    0,
                    0,
                ],
                "interval_start": [
                    0,
                    25,
                    50,
                    80,
                    100,
                    120,
                    125,
                    140,
                    150,
                    180,
                    200,
                    210,
                    300,
                    50,
                    100,
                    150,
                    160,
                    170,
                    200,
                    210,
                    250,
                    10,
                    0,
                    25,
                    40,
                    50,
                    60,
                    75,
                    90,
                    100,
                    130,
                    200,
                    0,
                    20,
                    30,
                    50,
                    150,
                    0,
                    80,
                    150,
                ],
                "interval_end": [
                    25,
                    50,
                    75,
                    100,
                    120,
                    125,
                    140,
                    150,
                    180,
                    200,
                    210,
                    300,
                    400,
                    100,
                    150,
                    160,
                    170,
                    200,
                    210,
                    250,
                    350,
                    20,
                    25,
                    40,
                    50,
                    60,
                    75,
                    90,
                    100,
                    110,
                    150,
                    250,
                    20,
                    30,
                    50,
                    100,
                    200,
                    80,
                    100,
                    200,
                ],
                "topic": [
                    ["law"],
                    ["law", "medicine"],
                    ["medicine"],
                    ["medicine"],
                    None,
                    ["medicine"],
                    ["law", "medicine"],
                    ["law", "medicine"],
                    ["law", "medicine"],
                    ["medicine"],
                    None,
                    None,
                    None,
                    ["medicine"],
                    ["law", "medicine"],
                    None,
                    ["medicine"],
                    None,
                    ["law", "medicine"],
                    ["law"],
                    None,
                    None,
                    ["law"],
                    ["law"],
                    None,
                    ["law", "medicine"],
                    ["law"],
                    None,
                    ["medicine"],
                    ["medicine"],
                    ["medicine"],
                    ["law"],
                    None,
                    ["law"],
                    ["medicine"],
                    ["medicine"],
                    ["medicine"],
                    None,
                    ["law"],
                    None,
                ],
                "language": [
                    ["french"],
                    ["french", "english"],
                    ["english"],
                    None,
                    ["french"],
                    ["french"],
                    ["french"],
                    ["french", "english"],
                    ["english"],
                    ["english"],
                    ["french", "english"],
                    ["french"],
                    ["english"],
                    ["english"],
                    ["french", "english"],
                    ["french"],
                    ["french"],
                    ["french"],
                    ["french"],
                    ["french"],
                    ["french"],
                    ["english"],
                    None,
                    ["french"],
                    ["french"],
                    ["english"],
                    ["french"],
                    ["french"],
                    ["french", "english"],
                    ["english"],
                    ["english"],
                    None,
                    ["french"],
                    ["french"],
                    ["french"],
                    None,
                    None,
                    ["english"],
                    ["english"],
                    ["english"],
                ],
            }
        )

    def create_flexible_chunking_test_df(self):
        return pl.DataFrame(
            {
                "dataset_id": [
                    0,
                    0,
                ],
                "file_id": [0, 0],
                "interval_start": [0, 5],
                "interval_end": [5, 10],
                "language": [
                    ["english", "french"],
                    ["english", "german"],
                ],
            }
        )

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

        query_result = QueryResult(self.client._mdc, self.create_simple_df(), ArbitraryMixture(1))
        gt_meta = {
            "dataset_type": {0: "test_dataset_type"},
            "file_path": {0: "test_file_path", 1: "test_file_path"},
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

        query_result = QueryResult(self.client._mdc, self.create_simple_df(), ArbitraryMixture(2))
        chunks = list(iter(query_result))
        chunks = [chunk._result_index for chunk in chunks]

        for chunk in chunks:
            for _, d1 in chunk.items():
                assert len(d1.keys()) == 1
                assert 0 in d1
                for fid, ranges in d1[0].items():
                    assert fid == 0 or fid == 1
                    for r_start, r_end in ranges:
                        assert r_end - r_start == 2

    @patch("mixtera.core.datacollection.MixteraDataCollection._get_dataset_func_by_id")
    @patch("mixtera.core.datacollection.MixteraDataCollection._get_dataset_type_by_id")
    @patch("mixtera.core.datacollection.MixteraDataCollection._get_file_path_by_id")
    def test_create_chunker_index_simple(
        self,
        mock_get_file_path_by_id: MagicMock,
        mock_get_dataset_type_by_id: MagicMock,
        mock_get_dataset_func_by_id: MagicMock,
    ):
        mock_get_file_path_by_id.return_value = "test_file_path"
        mock_get_dataset_type_by_id.return_value = "test_dataset_type"
        mock_get_dataset_func_by_id.return_value = lambda x: x

        query_result = QueryResult(self.client._mdc, self.create_simple_df(), ArbitraryMixture(1))
        chunker_index = query_result._chunker_index

        expected_chunker_index = {
            MixtureKey({"language": ["french"]}): {0: {0: [[0, 50], [150, 200]], 1: [[0, 100]]}},
            MixtureKey({"language": ["english", "french"]}): {0: {0: [[50, 100]]}},
            MixtureKey({"language": ["english"]}): {0: {0: [[100, 150]]}},
        }

        self.assertEqual(chunker_index, expected_chunker_index)

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
                # Note the order here! the earlier ranges come first due to our sorting
                MixtureKey({"language": ["english"]}): {0: {0: [(50, 52), (148, 150)]}},
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

        mixture = StaticMixture(16, mixture_concentration)
        query_result = QueryResult(self.client._mdc, self.create_simple_df(), mixture)

        # Check the structure of the chunker index
        chunker_index = defaultdict_to_dict(query_result._chunker_index)
        self.assertDictEqual(chunker_index, reference_chunker_index)

        # Check the equality of the chunks
        chunks = list(iter(query_result))
        for i, chunk in enumerate(chunks):
            self.assertDictEqual(reference_chunks[i], chunk._result_index)

        # Ensure we have the expected number of chunks
        self.assertEqual(len(chunks), len(reference_chunks))

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

        mixture = InferringMixture(30)
        query_result = QueryResult(self.client._mdc, self.create_simple_df(), mixture)

        assert mixture._mixture == {
            MixtureKey({"language": ["french"]}): 20,
            MixtureKey({"language": ["english"]}): 5,
            MixtureKey({"language": ["english", "french"]}): 5,
        }

        chunks = list(iter(query_result))

        # Check the structure of the chunker index
        chunker_index = defaultdict_to_dict(query_result._chunker_index)
        self.assertDictEqual(chunker_index, reference_chunker_index)

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
                MixtureKey({"language": ["english"]}): {0: {0: [(50, 56), (148, 150)]}},
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

        mixture_1 = StaticMixture(16, mixture_concentration_1)
        mixture_2 = StaticMixture(16, mixture_concentration_2)

        query_result = QueryResult(self.client._mdc, self.create_simple_df(), mixture_1)
        result_iterator = iter(query_result)

        chunks = [next(result_iterator) for _ in range(10)]
        query_result.update_mixture(mixture_2)
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
                MixtureKey({"language": ["english"], "topic": ["medicine"]}): {1: {0: [(90, 93), (149, 150)]}},
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
        mixture = StaticMixture(10, mixture_concentration)
        query_result = QueryResult(self.client._mdc, self.create_complex_df(), mixture)

        chunks = list(iter(query_result))

        # Check the equality of the chunks
        for i, chunk in enumerate(chunks):
            self.assertDictEqual(reference_chunks[i], chunk._result_index)
