import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
from mixtera.core.client import MixteraClient
from mixtera.core.query import Query
from mixtera.core.query.mixture import ArbitraryMixture, InferringMixture, MixtureKey, StaticMixture
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
        ).to_arrow()

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
        ).to_arrow()

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
                "another_property": ["MIT", "CC"],
            }
        ).to_arrow()

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
                    assert fid in {0, 1}
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
            MixtureKey({"language": ["english"]}): {0: {0: [(100, 150)]}},
            MixtureKey({"language": ["english", "french"]}): {0: {0: [(50, 100)]}},
            MixtureKey({"language": ["french"]}): {0: {0: [(0, 50), (150, 200)], 1: [(0, 100)]}},
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
                MixtureKey({"language": ["french"]}): {0: {1: [(92, 100)], 0: [(68, 72)]}},
                MixtureKey({"language": ["english"]}): {0: {0: [(64, 68)]}},
            },
            {
                MixtureKey({"language": ["french"]}): {0: {0: [(76, 88)]}},
                MixtureKey({"language": ["english"]}): {0: {0: [(72, 76)]}},
            },
        ]

        reference_chunker_index = {
            MixtureKey({"language": ["french"]}): {0: {0: [(0, 50), (150, 200)], 1: [(0, 100)]}},
            MixtureKey({"language": ["english", "french"]}): {0: {0: [(50, 100)]}},
            MixtureKey({"language": ["english"]}): {0: {0: [(100, 150)]}},
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
                MixtureKey({"language": ["english"]}): {0: {0: [(100, 105)]}},
                MixtureKey({"language": ["english", "french"]}): {0: {0: [(105, 110)]}},
                MixtureKey({"language": ["french"]}): {0: {0: [(0, 20)]}},
            },
            {
                MixtureKey({"language": ["english"]}): {0: {0: [(110, 115)]}},
                MixtureKey({"language": ["english", "french"]}): {0: {0: [(115, 120)]}},
                MixtureKey({"language": ["french"]}): {0: {0: [(20, 40)]}},
            },
            {
                MixtureKey({"language": ["english"]}): {0: {0: [(120, 125)]}},
                MixtureKey({"language": ["english", "french"]}): {0: {0: [(125, 130)]}},
                MixtureKey({"language": ["french"]}): {0: {0: [(40, 50), (150, 160)]}},
            },
            {
                MixtureKey({"language": ["english"]}): {0: {0: [(130, 135)]}},
                MixtureKey({"language": ["english", "french"]}): {0: {0: [(135, 140)]}},
                MixtureKey({"language": ["french"]}): {0: {0: [(160, 180)]}},
            },
            {
                MixtureKey({"language": ["english"]}): {0: {0: [(140, 145)]}},
                MixtureKey({"language": ["english", "french"]}): {0: {0: [(145, 150)]}},
                MixtureKey({"language": ["french"]}): {0: {0: [(180, 200)]}},
            },
            {
                MixtureKey({"language": ["english"]}): {0: {0: [(50, 55)]}},
                MixtureKey({"language": ["english", "french"]}): {0: {1: [(20, 25)]}},
                MixtureKey({"language": ["french"]}): {0: {1: [(0, 20)]}},
            },
            {
                MixtureKey({"language": ["english"]}): {0: {0: [(55, 60)]}},
                MixtureKey({"language": ["english", "french"]}): {0: {1: [(45, 50)]}},
                MixtureKey({"language": ["french"]}): {0: {1: [(25, 45)]}},
            },
            {
                MixtureKey({"language": ["english"]}): {0: {0: [(60, 65)]}},
                MixtureKey({"language": ["english", "french"]}): {0: {1: [(70, 75)]}},
                MixtureKey({"language": ["french"]}): {0: {1: [(50, 70)]}},
            },
            {
                MixtureKey({"language": ["english"]}): {0: {0: [(65, 70)]}},
                MixtureKey({"language": ["english", "french"]}): {0: {1: [(95, 100)]}},
                MixtureKey({"language": ["french"]}): {0: {1: [(75, 95)]}},
            },
            {
                MixtureKey({"language": ["english"]}): {0: {0: [(70, 75)]}},
                MixtureKey({"language": ["english", "french"]}): {0: {0: [(95, 100)]}},
                MixtureKey({"language": ["french"]}): {0: {0: [(75, 95)]}},
            },
        ]

        reference_chunker_index = {
            MixtureKey({"language": ["english"]}): {0: {0: [(100, 150)]}},
            MixtureKey({"language": ["english", "french"]}): {0: {0: [(50, 100)]}},
            MixtureKey({"language": ["french"]}): {0: {0: [(0, 50), (150, 200)], 1: [(0, 100)]}},
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
    def test_create_complex_chunking_with_static_mixture(
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
                MixtureKey({"language": ["english"], "topic": ["medicine"]}): {
                    1: {0: [(149, 150)]},
                    0: {0: [(150, 153)]},
                },
            },
            {
                MixtureKey({"language": ["french"], "topic": ["law"]}): {0: {1: [(127, 133)]}},
                MixtureKey({"language": ["english"], "topic": ["medicine"]}): {0: {0: [(153, 157)]}},
            },
            {
                MixtureKey({"language": ["french"], "topic": ["law"]}): {0: {1: [(133, 139)]}},
                MixtureKey({"language": ["english"], "topic": ["medicine"]}): {
                    0: {0: [(157, 161)]},
                },
            },
            {
                MixtureKey({"language": ["french"], "topic": ["law"]}): {0: {1: [(139, 145)]}},
                MixtureKey({"language": ["english"], "topic": ["medicine"]}): {0: {0: [(161, 165)]}},
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

    @patch("mixtera.core.datacollection.MixteraDataCollection._get_dataset_func_by_id")
    @patch("mixtera.core.datacollection.MixteraDataCollection._get_dataset_type_by_id")
    @patch("mixtera.core.datacollection.MixteraDataCollection._get_file_path_by_id")
    def test_create_chunking_with_arbitrary_mixture(
        self,
        mock_get_file_path_by_id: MagicMock,
        mock_get_dataset_type_by_id: MagicMock,
        mock_get_dataset_func_by_id: MagicMock,
    ):
        mock_get_file_path_by_id.return_value = "test_file_path"
        mock_get_dataset_type_by_id.return_value = "test_dataset_type"
        mock_get_dataset_func_by_id.return_value = lambda x: x

        reference_chunks = [
            {MixtureKey({"language": ["french"], "topic": ["law", "medicine"]}): {0: {0: [(125, 132)]}}},
            {MixtureKey({"language": ["english", "french"], "topic": ["medicine"]}): {1: {0: [(90, 97)]}}},
            {MixtureKey({"language": ["english"], "topic": ["law", "medicine"]}): {0: {0: [(150, 157)]}}},
            {MixtureKey({"language": ["french"], "topic": ["medicine"]}): {0: {0: [(120, 125)], 1: [(160, 162)]}}},
            {MixtureKey({"language": ["english"], "topic": ["medicine"]}): {0: {0: [(50, 57)]}}},
            {MixtureKey({"topic": ["medicine"]}): {0: {0: [(80, 87)]}}},
            {MixtureKey({"language": ["french"]}): {0: {0: [(100, 107)]}}},
            {MixtureKey({"language": ["english"], "topic": ["law"]}): {2: {0: [(80, 87)]}}},
            {MixtureKey({"language": ["english"]}): {0: {0: [(300, 307)]}}},
            {MixtureKey({"language": ["french"], "topic": ["law"]}): {0: {0: [(0, 7)]}}},
            {MixtureKey({"language": ["english", "french"]}): {0: {0: [(200, 207)]}}},
            {MixtureKey({"topic": ["law"]}): {1: {0: [(0, 7)]}}},
            {MixtureKey({"language": ["english", "french"], "topic": ["law", "medicine"]}): {0: {0: [(25, 32)]}}},
            {MixtureKey({"language": ["french"], "topic": ["law", "medicine"]}): {0: {0: [(132, 139)]}}},
            {MixtureKey({"language": ["english", "french"], "topic": ["medicine"]}): {1: {0: [(97, 100)]}}},
            {MixtureKey({"language": ["english"], "topic": ["law", "medicine"]}): {0: {0: [(157, 164)]}}},
            {MixtureKey({"language": ["french"], "topic": ["medicine"]}): {0: {1: [(162, 169)]}}},
            {MixtureKey({"language": ["english"], "topic": ["medicine"]}): {0: {0: [(57, 64)]}}},
            {MixtureKey({"topic": ["medicine"]}): {0: {0: [(87, 94)]}}},
            {MixtureKey({"language": ["french"]}): {0: {0: [(107, 114)]}}},
            {MixtureKey({"language": ["english"], "topic": ["law"]}): {2: {0: [(87, 94)]}}},
            {MixtureKey({"language": ["english"]}): {0: {0: [(307, 314)]}}},
            {MixtureKey({"language": ["french"], "topic": ["law"]}): {0: {0: [(7, 14)]}}},
            {MixtureKey({"language": ["english", "french"]}): {0: {0: [(207, 210)]}}},
            {MixtureKey({"topic": ["law"]}): {1: {0: [(7, 14)]}}},
            {MixtureKey({"language": ["english", "french"], "topic": ["law", "medicine"]}): {0: {0: [(32, 39)]}}},
            {
                MixtureKey({"language": ["french"], "topic": ["law", "medicine"]}): {
                    0: {0: [(139, 140)], 1: [(200, 206)]}
                }
            },
            {MixtureKey({"language": ["english"], "topic": ["law", "medicine"]}): {0: {0: [(164, 171)]}}},
            {MixtureKey({"language": ["french"], "topic": ["medicine"]}): {0: {1: [(169, 170)]}, 1: {1: [(30, 36)]}}},
            {MixtureKey({"language": ["english"], "topic": ["medicine"]}): {0: {0: [(64, 71)]}}},
            {MixtureKey({"topic": ["medicine"]}): {0: {0: [(94, 100)]}, 1: {1: [(50, 51)]}}},
            {MixtureKey({"language": ["french"]}): {0: {0: [(114, 120), (210, 211)]}}},
            {MixtureKey({"language": ["english"], "topic": ["law"]}): {2: {0: [(94, 100)]}}},
            {MixtureKey({"language": ["english"]}): {0: {0: [(314, 321)]}}},
            {MixtureKey({"language": ["french"], "topic": ["law"]}): {0: {0: [(14, 21)]}}},
            {MixtureKey({"topic": ["law"]}): {1: {0: [(14, 21)]}}},
            {MixtureKey({"language": ["english", "french"], "topic": ["law", "medicine"]}): {0: {0: [(39, 46)]}}},
            {MixtureKey({"language": ["french"], "topic": ["law", "medicine"]}): {0: {1: [(206, 210)]}}},
            {MixtureKey({"language": ["french"], "topic": ["medicine"]}): {1: {1: [(36, 43)]}}},
            {MixtureKey({"language": ["english"], "topic": ["medicine"]}): {0: {0: [(71, 75), (180, 183)]}}},
            {MixtureKey({"topic": ["medicine"]}): {1: {1: [(51, 58)]}}},
            {MixtureKey({"language": ["french"]}): {0: {0: [(211, 218)]}}},
            {MixtureKey({"language": ["english"]}): {0: {0: [(321, 328)]}}},
            {MixtureKey({"language": ["french"], "topic": ["law"]}): {0: {0: [(21, 25)], 1: [(210, 213)]}}},
            {
                MixtureKey({"language": ["english", "french"], "topic": ["law", "medicine"]}): {
                    0: {0: [(46, 50), (140, 143)]}
                }
            },
            {MixtureKey({"language": ["french"], "topic": ["medicine"]}): {1: {1: [(43, 50)]}}},
            {MixtureKey({"language": ["english"], "topic": ["medicine"]}): {0: {0: [(183, 190)]}}},
            {MixtureKey({"topic": ["medicine"]}): {1: {1: [(58, 65)]}}},
            {MixtureKey({"language": ["french"]}): {0: {0: [(218, 225)]}}},
            {MixtureKey({"language": ["french"], "topic": ["law"]}): {0: {1: [(213, 220)]}}},
            {MixtureKey({"language": ["english", "french"], "topic": ["law", "medicine"]}): {0: {0: [(143, 150)]}}},
            {MixtureKey({"language": ["english"], "topic": ["law", "medicine"]}): {0: {0: [(171, 178)]}}},
            {MixtureKey({"language": ["english"], "topic": ["medicine"]}): {0: {0: [(190, 197)]}}},
            {MixtureKey({"topic": ["medicine"]}): {1: {1: [(65, 72)]}}},
            {MixtureKey({"language": ["french"]}): {0: {0: [(225, 232)]}}},
            {MixtureKey({"language": ["french"], "topic": ["law"]}): {0: {1: [(220, 227)]}}},
            {MixtureKey({"language": ["english", "french"], "topic": ["law", "medicine"]}): {0: {1: [(100, 107)]}}},
            {
                MixtureKey({"language": ["english"], "topic": ["law", "medicine"]}): {
                    0: {0: [(178, 180)]},
                    1: {0: [(50, 55)]},
                }
            },
            {MixtureKey({"topic": ["medicine"]}): {1: {1: [(72, 79)]}}},
            {MixtureKey({"language": ["french"]}): {0: {0: [(232, 239)]}}},
            {MixtureKey({"language": ["french"], "topic": ["law"]}): {0: {1: [(227, 234)]}}},
            {MixtureKey({"language": ["english", "french"], "topic": ["law", "medicine"]}): {0: {1: [(107, 114)]}}},
            {MixtureKey({"language": ["english"], "topic": ["law", "medicine"]}): {1: {0: [(55, 60)]}}},
            {MixtureKey({"topic": ["medicine"]}): {1: {1: [(79, 86)]}}},
            {MixtureKey({"language": ["french"]}): {0: {0: [(239, 246)]}}},
            {MixtureKey({"language": ["french"], "topic": ["law"]}): {0: {1: [(234, 241)]}}},
            {MixtureKey({"language": ["english", "french"], "topic": ["law", "medicine"]}): {0: {1: [(114, 121)]}}},
            {MixtureKey({"topic": ["medicine"]}): {1: {1: [(86, 93)]}}},
            {MixtureKey({"language": ["french"]}): {0: {0: [(246, 253)]}}},
            {MixtureKey({"language": ["french"], "topic": ["law"]}): {0: {1: [(241, 248)]}}},
            {MixtureKey({"language": ["english", "french"], "topic": ["law", "medicine"]}): {0: {1: [(121, 128)]}}},
            {MixtureKey({"language": ["english"], "topic": ["medicine"]}): {0: {0: [(197, 200)], 1: [(50, 54)]}}},
            {MixtureKey({"topic": ["medicine"]}): {1: {1: [(93, 100)]}}},
            {MixtureKey({"language": ["french"]}): {0: {0: [(253, 260)]}}},
            {MixtureKey({"language": ["french"], "topic": ["law"]}): {0: {1: [(248, 250)]}, 1: {0: [(25, 30)]}}},
            {MixtureKey({"language": ["english", "french"], "topic": ["law", "medicine"]}): {0: {1: [(128, 135)]}}},
            {MixtureKey({"language": ["english"], "topic": ["medicine"]}): {0: {1: [(54, 61)]}}},
            {MixtureKey({"topic": ["medicine"]}): {1: {1: [(150, 157)]}}},
            {MixtureKey({"language": ["french"]}): {0: {0: [(260, 267)]}}},
            {MixtureKey({"language": ["french"], "topic": ["law"]}): {1: {0: [(30, 37)]}}},
            {MixtureKey({"language": ["english", "french"], "topic": ["law", "medicine"]}): {0: {1: [(135, 142)]}}},
            {MixtureKey({"language": ["english"], "topic": ["medicine"]}): {0: {1: [(61, 68)]}}},
            {MixtureKey({"topic": ["medicine"]}): {1: {1: [(157, 164)]}}},
            {MixtureKey({"language": ["french"]}): {0: {0: [(267, 274)]}}},
            {MixtureKey({"language": ["french"], "topic": ["law"]}): {1: {0: [(37, 40), (60, 64)]}}},
            {MixtureKey({"language": ["english", "french"], "topic": ["law", "medicine"]}): {0: {1: [(142, 149)]}}},
            {MixtureKey({"language": ["english"], "topic": ["medicine"]}): {0: {1: [(68, 75)]}}},
            {MixtureKey({"topic": ["medicine"]}): {1: {1: [(164, 171)]}}},
            {MixtureKey({"language": ["french"]}): {0: {0: [(274, 281)]}}},
            {MixtureKey({"language": ["french"], "topic": ["law"]}): {1: {0: [(64, 71)]}}},
            {MixtureKey({"language": ["english", "french"], "topic": ["law", "medicine"]}): {0: {1: [(149, 150)]}}},
            {MixtureKey({"language": ["english"], "topic": ["medicine"]}): {0: {1: [(75, 82)]}}},
            {MixtureKey({"topic": ["medicine"]}): {1: {1: [(171, 178)]}}},
            {MixtureKey({"language": ["french"]}): {0: {0: [(281, 288)]}}},
            {MixtureKey({"language": ["french"], "topic": ["law"]}): {1: {0: [(71, 75)], 1: [(20, 23)]}}},
            {MixtureKey({"language": ["english"], "topic": ["medicine"]}): {0: {1: [(82, 89)]}}},
            {MixtureKey({"topic": ["medicine"]}): {1: {1: [(178, 185)]}}},
            {MixtureKey({"language": ["french"]}): {0: {0: [(288, 295)]}}},
            {MixtureKey({"language": ["french"], "topic": ["law"]}): {1: {1: [(23, 30)]}}},
            {MixtureKey({"topic": ["medicine"]}): {1: {1: [(185, 192)]}}},
            {MixtureKey({"language": ["french"]}): {0: {0: [(295, 300)], 1: [(150, 152)]}}},
            {MixtureKey({"topic": ["medicine"]}): {1: {1: [(192, 199)]}}},
            {MixtureKey({"language": ["french"]}): {0: {1: [(152, 159)]}}},
            {MixtureKey({"topic": ["law"]}): {1: {0: [(21, 25), (200, 203)]}}},
            {MixtureKey({"topic": ["medicine"]}): {1: {1: [(199, 200)]}}},
            {MixtureKey({"language": ["french"]}): {0: {1: [(159, 160), (170, 176)]}}},
            {MixtureKey({"topic": ["law"]}): {1: {0: [(203, 210)]}}},
            {MixtureKey({"language": ["french"]}): {0: {1: [(176, 183)]}}},
            {MixtureKey({"topic": ["law"]}): {1: {0: [(210, 217)]}}},
            {MixtureKey({"topic": ["law"]}): {1: {0: [(217, 224)]}}},
            {MixtureKey({"topic": ["law"]}): {1: {0: [(224, 231)]}}},
            {MixtureKey({"topic": ["law"]}): {1: {0: [(231, 238)]}}},
            {MixtureKey({"topic": ["law"]}): {1: {0: [(238, 245)]}}},
            {MixtureKey({"topic": ["law"]}): {1: {0: [(245, 250)]}}},
            {MixtureKey({"language": ["english"], "topic": ["medicine"]}): {0: {1: [(89, 96)]}}},
            {MixtureKey({"language": ["english"], "topic": ["medicine"]}): {0: {1: [(96, 100)]}, 1: {0: [(100, 103)]}}},
            {MixtureKey({"language": ["english"], "topic": ["medicine"]}): {1: {0: [(103, 110)]}}},
            {MixtureKey({"language": ["english"], "topic": ["medicine"]}): {1: {0: [(130, 137)]}}},
            {MixtureKey({"language": ["english"], "topic": ["medicine"]}): {1: {0: [(137, 144)]}}},
            {MixtureKey({"language": ["english"], "topic": ["medicine"]}): {1: {0: [(144, 150)]}}},
            {MixtureKey({"language": ["french"]}): {0: {1: [(183, 190)]}}},
            {MixtureKey({"language": ["french"]}): {0: {1: [(190, 197)]}}},
            {MixtureKey({"language": ["french"]}): {0: {1: [(197, 200), (250, 254)]}}},
            {MixtureKey({"language": ["french"]}): {0: {1: [(254, 261)]}}},
            {MixtureKey({"language": ["french"]}): {0: {1: [(261, 268)]}}},
            {MixtureKey({"language": ["french"]}): {0: {1: [(268, 275)]}}},
            {MixtureKey({"language": ["french"]}): {0: {1: [(275, 282)]}}},
            {MixtureKey({"language": ["french"]}): {0: {1: [(282, 289)]}}},
            {MixtureKey({"language": ["french"]}): {0: {1: [(289, 296)]}}},
            {MixtureKey({"language": ["french"]}): {0: {1: [(296, 303)]}}},
            {MixtureKey({"language": ["french"]}): {0: {1: [(303, 310)]}}},
            {MixtureKey({"language": ["french"]}): {0: {1: [(310, 317)]}}},
            {MixtureKey({"language": ["french"]}): {0: {1: [(317, 324)]}}},
            {MixtureKey({"language": ["french"]}): {0: {1: [(324, 331)]}}},
            {MixtureKey({"language": ["french"]}): {0: {1: [(331, 338)]}}},
            {MixtureKey({"language": ["french"]}): {0: {1: [(338, 345)]}}},
            {MixtureKey({"language": ["french"]}): {0: {1: [(345, 350)]}, 1: {0: [(40, 42)]}}},
            {MixtureKey({"language": ["french"]}): {1: {0: [(42, 49)]}}},
            {MixtureKey({"language": ["french"]}): {1: {0: [(49, 50), (75, 81)]}}},
            {MixtureKey({"language": ["french"]}): {1: {0: [(81, 88)]}}},
            {MixtureKey({"language": ["french"]}): {1: {0: [(88, 90)], 1: [(0, 5)]}}},
            {MixtureKey({"language": ["french"]}): {1: {1: [(5, 12)]}}},
            {MixtureKey({"language": ["french"]}): {1: {1: [(12, 19)]}}},
            {MixtureKey({"language": ["french"]}): {1: {1: [(19, 20)]}}},
            {MixtureKey({"language": ["english"]}): {0: {0: [(328, 335)]}}},
            {MixtureKey({"language": ["english"]}): {0: {0: [(335, 342)]}}},
            {MixtureKey({"language": ["english"]}): {0: {0: [(342, 349)]}}},
            {MixtureKey({"language": ["english"]}): {0: {0: [(349, 356)]}}},
            {MixtureKey({"language": ["english"]}): {0: {0: [(356, 363)]}}},
            {MixtureKey({"language": ["english"]}): {0: {0: [(363, 370)]}}},
            {MixtureKey({"language": ["english"]}): {0: {0: [(370, 377)]}}},
            {MixtureKey({"language": ["english"]}): {0: {0: [(377, 384)]}}},
            {MixtureKey({"language": ["english"]}): {0: {0: [(384, 391)]}}},
            {MixtureKey({"language": ["english"]}): {0: {0: [(391, 398)]}}},
            {MixtureKey({"language": ["english"]}): {0: {0: [(398, 400)], 2: [(10, 15)]}}},
            {MixtureKey({"language": ["english"]}): {0: {2: [(15, 20)]}, 2: {0: [(0, 2)]}}},
            {MixtureKey({"language": ["english"]}): {2: {0: [(2, 9)]}}},
            {MixtureKey({"language": ["english"]}): {2: {0: [(9, 16)]}}},
            {MixtureKey({"language": ["english"]}): {2: {0: [(16, 23)]}}},
            {MixtureKey({"language": ["english"]}): {2: {0: [(23, 30)]}}},
            {MixtureKey({"language": ["english"]}): {2: {0: [(30, 37)]}}},
            {MixtureKey({"language": ["english"]}): {2: {0: [(37, 44)]}}},
            {MixtureKey({"language": ["english"]}): {2: {0: [(44, 51)]}}},
            {MixtureKey({"language": ["english"]}): {2: {0: [(51, 58)]}}},
            {MixtureKey({"language": ["english"]}): {2: {0: [(58, 65)]}}},
            {MixtureKey({"language": ["english"]}): {2: {0: [(65, 72)]}}},
            {MixtureKey({"language": ["english"]}): {2: {0: [(72, 79)]}}},
            {MixtureKey({"language": ["english"]}): {2: {0: [(79, 80), (150, 156)]}}},
            {MixtureKey({"language": ["english"]}): {2: {0: [(156, 163)]}}},
            {MixtureKey({"language": ["english"]}): {2: {0: [(163, 170)]}}},
            {MixtureKey({"language": ["english"]}): {2: {0: [(170, 177)]}}},
            {MixtureKey({"language": ["english"]}): {2: {0: [(177, 184)]}}},
            {MixtureKey({"language": ["english"]}): {2: {0: [(184, 191)]}}},
            {MixtureKey({"language": ["english"]}): {2: {0: [(191, 198)]}}},
            {MixtureKey({"language": ["english"]}): {2: {0: [(198, 200)]}}},
        ]

        query_result = QueryResult(self.client._mdc, self.create_complex_df(), ArbitraryMixture(7))
        chunks = list(iter(query_result))

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
    def test_flexible_chunking(
        self,
        mock_get_file_path_by_id: MagicMock,
        mock_get_dataset_type_by_id: MagicMock,
        mock_get_dataset_func_by_id: MagicMock,
    ):
        mock_get_file_path_by_id.return_value = "test_file_path"
        mock_get_dataset_type_by_id.return_value = "test_dataset_type"
        mock_get_dataset_func_by_id.return_value = lambda x: x

        # Note the order here does matter and we correctly return the ranges in order.
        reference_chunks = [
            {
                MixtureKey({"language": ["english"]}): {0: {0: [(0, 5), (5, 10)]}},
            },
        ]

        reference_chunker_index = {
            MixtureKey({"language": ["english", "french"], "another_property": ["MIT"]}): {0: {0: [(0, 5)]}},
            MixtureKey({"language": ["english", "german"], "another_property": ["CC"]}): {0: {0: [(5, 10)]}},
        }

        mixture_concentration = {
            MixtureKey({"language": ["english"]}): 1,
        }

        mixture = StaticMixture(10, mixture_concentration)
        query_result = QueryResult(self.client._mdc, self.create_flexible_chunking_test_df(), mixture)
        chunks = list(iter(query_result))

        # Check the structure of the chunker index
        chunker_index = defaultdict_to_dict(query_result._chunker_index)
        self.assertDictEqual(chunker_index, reference_chunker_index)

        # Check the equality of the chunks
        for i, chunk in enumerate(chunks):
            self.assertDictEqual(reference_chunks[i], chunk._result_index)

        self.assertEqual(len(chunks), 1)

    def create_uneven_distribution_test_df(self):
        data = {"dataset_id": [], "file_id": [], "interval_start": [], "interval_end": [], "language": []}
        # 50 JavaScript, 30 HTML, 20 Python samples

        for i in range(50):
            data["dataset_id"].append(0)
            data["file_id"].append(0)
            data["interval_start"].append(i)
            data["interval_end"].append(i + 1)
            data["language"].append(["JavaScript"])

        for i in range(50, 80):
            data["dataset_id"].append(0)
            data["file_id"].append(0)
            data["interval_start"].append(i)
            data["interval_end"].append(i + 1)
            data["language"].append(["HTML"])

        for i in range(80, 100):
            data["dataset_id"].append(0)
            data["file_id"].append(0)
            data["interval_start"].append(i)
            data["interval_end"].append(i + 1)
            data["language"].append(["Python"])
        df = pl.DataFrame(data)
        return df.to_arrow()

    @patch("mixtera.core.datacollection.MixteraDataCollection._get_dataset_func_by_id")
    @patch("mixtera.core.datacollection.MixteraDataCollection._get_dataset_type_by_id")
    @patch("mixtera.core.datacollection.MixteraDataCollection._get_file_path_by_id")
    def test_non_strict_mixture_with_component_exhaustion(
        self,
        mock_get_file_path_by_id: MagicMock,
        mock_get_dataset_type_by_id: MagicMock,
        mock_get_dataset_func_by_id: MagicMock,
    ):
        mock_get_file_path_by_id.return_value = "test_file_path"
        mock_get_dataset_type_by_id.return_value = "test_dataset_type"
        mock_get_dataset_func_by_id.return_value = lambda x: x

        df = self.create_uneven_distribution_test_df()

        # Define non-strict mixture for best-effort chunk generation
        mixture = StaticMixture(
            chunk_size=10,
            mixture={
                MixtureKey({"language": ["JavaScript"]}): 0.4,
                MixtureKey({"language": ["Python"]}): 0.4,
                MixtureKey({"language": ["HTML"]}): 0.2,
            },
            strict=False,
        )

        query_result = QueryResult(self.client._mdc, df, mixture)

        chunks = list(iter(query_result))

        total_samples = len(df)
        self.assertTrue(len(chunks) > 0)

        language_counts = {"JavaScript": 0, "Python": 0, "HTML": 0}

        expected_counts = [
            {"JavaScript": 4, "Python": 4, "HTML": 2},
            {"JavaScript": 4, "Python": 4, "HTML": 2},
            {"JavaScript": 4, "Python": 4, "HTML": 2},
            {"JavaScript": 4, "Python": 4, "HTML": 2},
            {"JavaScript": 4, "Python": 4, "HTML": 2},
            {"JavaScript": 7, "Python": 0, "HTML": 3},
            {"JavaScript": 7, "Python": 0, "HTML": 3},
            {"JavaScript": 7, "Python": 0, "HTML": 3},
            {"JavaScript": 7, "Python": 0, "HTML": 3},
            {"JavaScript": 2, "Python": 0, "HTML": 8},
        ]

        for i, chunk in enumerate(chunks):
            chunk_size = 0
            chunk_language_counts = {"JavaScript": 0, "Python": 0, "HTML": 0}
            for mixture_key, datasets in chunk._result_index.items():
                language = mixture_key.properties["language"][0]
                for files in datasets.values():
                    for ranges in files.values():
                        for start, end in ranges:
                            count = end - start
                            chunk_size += count
                            chunk_language_counts[language] += count
                            language_counts[language] += count

            # Check that chunk size matches mixture.chunk_size or is less (for the last chunk)
            self.assertTrue(chunk_size == mixture.chunk_size, f"Chunk {i}: Expected {mixture.chunk_size} samples, got {chunk_size}")

            # Check that the result matches expected counts
            for lang in ["JavaScript", "Python", "HTML"]:
                self.assertTrue(
                    chunk_language_counts[lang] == expected_counts[i][lang],
                    f"Chunk {i}: Expected {expected_counts[i][lang]} samples of {lang}, "
                    f"got {chunk_language_counts[lang]}",
                )

        # Check that total samples returned matches the dataset
        total_samples_collected = sum(language_counts.values())
        self.assertEqual(total_samples_collected, total_samples)

    @patch("mixtera.core.datacollection.MixteraDataCollection._get_dataset_func_by_id")
    @patch("mixtera.core.datacollection.MixteraDataCollection._get_dataset_type_by_id")
    @patch("mixtera.core.datacollection.MixteraDataCollection._get_file_path_by_id")
    def test_strict_mixture_with_component_exhaustion(
        self,
        mock_get_file_path_by_id: MagicMock,
        mock_get_dataset_type_by_id: MagicMock,
        mock_get_dataset_func_by_id: MagicMock,
    ):
        mock_get_file_path_by_id.return_value = "test_file_path"
        mock_get_dataset_type_by_id.return_value = "test_dataset_type"
        mock_get_dataset_func_by_id.return_value = lambda x: x

        df = self.create_uneven_distribution_test_df()

        mixture = StaticMixture(
            chunk_size=10,
            mixture={
                MixtureKey({"language": ["JavaScript"]}): 0.4,
                MixtureKey({"language": ["Python"]}): 0.4,
                MixtureKey({"language": ["HTML"]}): 0.2,
            },
            strict=True,
        )

        query_result = QueryResult(self.client._mdc, df, mixture)

        chunks = list(iter(query_result))

        expected_total_chunks = 5

        self.assertEqual(len(chunks), expected_total_chunks)

        language_counts = {"JavaScript": 0, "Python": 0, "HTML": 0}

        for i, chunk in enumerate(chunks):
            chunk_size = 0
            chunk_language_counts = {"JavaScript": 0, "Python": 0, "HTML": 0}
            for mixture_key, datasets in chunk._result_index.items():
                language = mixture_key.properties["language"][0]  # Get the language
                for files in datasets.values():
                    for ranges in files.values():
                        for start, end in ranges:
                            count = end - start
                            chunk_size += count
                            chunk_language_counts[language] += count
                            language_counts[language] += count

            # Check that chunk size matches mixture.chunk_size
            self.assertEqual(chunk_size, mixture.chunk_size)
            # Check that the proportions are as specified
            expected_counts = {"JavaScript": 4, "Python": 4, "HTML": 2}

            for lang in ["JavaScript", "Python", "HTML"]:
                self.assertTrue(
                    (chunk_language_counts[lang] == expected_counts[lang]),
                    f"Chunk {i}: Expected {expected_counts[lang]} samples of {lang}, got {chunk_language_counts[lang]}",
                )

        # After 'Python' is exhausted, the iteration should stop (no more chunks)
        total_samples_collected = sum(language_counts.values())
        expected_total_samples_collected = mixture.chunk_size * expected_total_chunks
        self.assertEqual(total_samples_collected, expected_total_samples_collected)
