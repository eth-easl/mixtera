import unittest
from unittest.mock import MagicMock

from mixtera.core.query import ResultChunk


class TestResultChunk(unittest.TestCase):
    def setUp(self):
        self.chunker_index = MagicMock()
        self.dataset_type_dict = {1: MagicMock()}
        self.file_path_dict = {1: "path/to/file"}
        self.parsing_func_dict = {1: MagicMock()}
        self.mixture = MagicMock()
        self.server_connection = MagicMock()
        self.chunk_size = 10

    def test_configure_result_streaming_with_per_window_mixture_and_invalid_window_size(self):
        result_chunk = ResultChunk(
            self.chunker_index,
            self.dataset_type_dict,
            self.file_path_dict,
            self.parsing_func_dict,
            self.chunk_size,
            None,
        )

        result_chunk._infer_mixture = MagicMock(return_value=self.mixture)

        result_chunk.configure_result_streaming(self.server_connection, 1, True, -1)

        self.assertEqual(result_chunk._window_size, 128)

        result_chunk._infer_mixture.assert_called_once()

    def test_configure_result_streaming_without_per_window_mixture_and_mixture_is_none(self):
        result_chunk = ResultChunk(
            self.chunker_index,
            self.dataset_type_dict,
            self.file_path_dict,
            self.parsing_func_dict,
            self.chunk_size,
            None,
        )

        result_chunk._infer_mixture = MagicMock(return_value=self.mixture)

        result_chunk.configure_result_streaming(self.server_connection, 2, False, 128)

        result_chunk._infer_mixture.assert_called_once()

    def test_infer_mixture(self):
        mock_result_index = {
            "property1": {0: {0: [(0, 10), (20, 30)]}},
            "property2": {0: {0: [(0, 5)]}, 1: {0: [(5, 15)]}},
        }

        expected_partition_masses = {"property1": 20, "property2": 15}

        result_chunk = ResultChunk(
            mock_result_index,
            self.dataset_type_dict,
            self.file_path_dict,
            self.parsing_func_dict,
            self.chunk_size,
            None,
        )

        mixture = result_chunk._infer_mixture()

        self.assertTrue(isinstance(mixture, dict))
        self.assertEqual(mixture, expected_partition_masses)

    def test_iterate_result_chunks_single_threaded(self):
        result_chunk = ResultChunk(
            self.chunker_index,
            self.dataset_type_dict,
            self.file_path_dict,
            self.parsing_func_dict,
            self.chunk_size,
            self.mixture,
        )
        result_chunk._degree_of_parallelism = 1

        mock_yield_source = ["chunk1", "chunk2", "chunk3"]
        result_chunk._iterate_single_threaded = MagicMock(return_value=mock_yield_source)

        results = list(result_chunk._iterate_result_chunks())

        result_chunk._iterate_single_threaded.assert_called_once()
        self.assertEqual(results, mock_yield_source)

    def test_iterate_result_chunks_multi_threaded(self):
        result_chunk = ResultChunk(
            self.chunker_index,
            self.dataset_type_dict,
            self.file_path_dict,
            self.parsing_func_dict,
            self.chunk_size,
            self.mixture,
        )
        result_chunk._degree_of_parallelism = 2

        mock_yield_source = ["chunk1", "chunk2", "chunk3"]
        result_chunk._iterate_multi_threaded = MagicMock(return_value=mock_yield_source)

        results = list(result_chunk._iterate_result_chunks())

        result_chunk._iterate_multi_threaded.assert_called_once()
        self.assertEqual(results, mock_yield_source)

    def test_iterate_result_chunks_with_invalid_degree_of_parallelism(self):
        result_chunk = ResultChunk(
            self.chunker_index,
            self.dataset_type_dict,
            self.file_path_dict,
            self.parsing_func_dict,
            self.chunk_size,
            self.mixture,
        )
        result_chunk._degree_of_parallelism = -1  # Invalid value

        mock_yield_source = ["chunk1", "chunk2", "chunk3"]
        result_chunk._iterate_single_threaded = MagicMock(return_value=mock_yield_source)

        results = list(result_chunk._iterate_result_chunks())

        result_chunk._iterate_single_threaded.assert_called_once()
        self.assertEqual(results, mock_yield_source)
        self.assertEqual(result_chunk._degree_of_parallelism, 1)

    def test_iterate_single_threaded_window_mixture(self):
        mock_element_counts = [("property1", 2), ("property2", 1)]
        mock_workloads = {"property1": [(1, 1, [(0, 2)]), (1, 2, [(4, 5)])], "property2": [(2, 1, [(10, 11)])]}
        mock_file_path_dict = {1: "file1", 2: "file2"}
        mock_dataset_type_dict = {1: MagicMock(), 2: MagicMock()}
        mock_parsing_func_dict = {1: MagicMock(return_value="parsed1"), 2: MagicMock(return_value="parsed2")}
        mock_server_connection = MagicMock()

        mock_dataset_type_dict[1].read_ranges_from_files.side_effect = [
            iter(["instance1", "instance2"]),
            iter(["instance3"]),
        ]
        mock_dataset_type_dict[2].read_ranges_from_files.return_value = iter(["instance4"])

        result_chunk = ResultChunk(
            MagicMock(),
            mock_dataset_type_dict,
            mock_file_path_dict,
            mock_parsing_func_dict,
            self.chunk_size,
            MagicMock(),
        )
        result_chunk._get_element_counts = MagicMock(return_value=mock_element_counts)
        result_chunk._prepare_workloads = MagicMock(return_value=mock_workloads)
        result_chunk._server_connection = mock_server_connection

        results = list(result_chunk._iterate_single_threaded_window_mixture())

        expected_results = ["instance1", "instance2", "instance4", "instance3"]
        self.assertEqual(results, expected_results)

        mock_dataset_type_dict[1].read_ranges_from_files.assert_has_calls(
            [
                unittest.mock.call({"file1": [(0, 2)]}, mock_parsing_func_dict[1], mock_server_connection),
                unittest.mock.call({"file2": [(4, 5)]}, mock_parsing_func_dict[1], mock_server_connection),
            ],
            any_order=True,
        )
        mock_dataset_type_dict[2].read_ranges_from_files.assert_called_once_with(
            {"file1": [(10, 11)]}, mock_parsing_func_dict[2], mock_server_connection
        )

    def test_iterate_single_threaded_window_mixture_complex(self):
        mock_element_counts = [("property1", 3), ("property2", 2), ("property3", 1)]
        mock_workloads = {
            "property1": [(1, 1, [(0, 2)]), (1, 2, [(4, 6)]), (1, 3, [(8, 9)])],
            "property2": [(2, 1, [(10, 12)]), (2, 2, [(14, 15)])],
            "property3": [(3, 1, [(16, 17)])],
        }
        mock_file_path_dict = {1: "file1", 2: "file2", 3: "file3"}
        mock_dataset_type_dict = {1: MagicMock(), 2: MagicMock(), 3: MagicMock()}
        mock_parsing_func_dict = {
            1: MagicMock(return_value="parsed1"),
            2: MagicMock(return_value="parsed2"),
            3: MagicMock(return_value="parsed3"),
        }
        mock_server_connection = MagicMock()

        mock_dataset_type_dict[1].read_ranges_from_files.side_effect = [
            iter(["instance1", "instance2"]),
            iter(["instance3", "instance4"]),
            iter(["instance5"]),
        ]
        mock_dataset_type_dict[2].read_ranges_from_files.side_effect = [
            iter(["instance6", "instance7"]),
            iter(["instance8"]),
        ]
        mock_dataset_type_dict[3].read_ranges_from_files.return_value = iter(["instance9"])

        result_chunk = ResultChunk(
            MagicMock(),
            mock_dataset_type_dict,
            mock_file_path_dict,
            mock_parsing_func_dict,
            self.chunk_size,
            MagicMock(),
        )
        result_chunk._get_element_counts = MagicMock(return_value=mock_element_counts)
        result_chunk._prepare_workloads = MagicMock(return_value=mock_workloads)
        result_chunk._server_connection = mock_server_connection

        results = list(result_chunk._iterate_single_threaded_window_mixture())

        expected_results = [
            "instance1",
            "instance2",
            "instance3",
            "instance6",
            "instance7",
            "instance9",
            "instance4",
            "instance5",
            "instance8",
        ]
        self.assertEqual(results, expected_results)

        mock_dataset_type_dict[1].read_ranges_from_files.assert_has_calls(
            [
                unittest.mock.call({"file1": [(0, 2)]}, mock_parsing_func_dict[1], mock_server_connection),
                unittest.mock.call({"file2": [(4, 6)]}, mock_parsing_func_dict[1], mock_server_connection),
                unittest.mock.call({"file3": [(8, 9)]}, mock_parsing_func_dict[1], mock_server_connection),
            ],
            any_order=True,
        )
        mock_dataset_type_dict[2].read_ranges_from_files.assert_has_calls(
            [
                unittest.mock.call({"file1": [(10, 12)]}, mock_parsing_func_dict[2], mock_server_connection),
                unittest.mock.call({"file2": [(14, 15)]}, mock_parsing_func_dict[2], mock_server_connection),
            ],
            any_order=True,
        )
        mock_dataset_type_dict[3].read_ranges_from_files.assert_called_once_with(
            {"file1": [(16, 17)]}, mock_parsing_func_dict[3], mock_server_connection
        )

    def test_iterate_single_threaded_overall_mixture(self):
        mock_result_index = {
            "property1": {1: {1: [(0, 2)], 2: [(6, 10)]}, 2: {1: [(7, 8)]}},
            "property2": {1: {2: [(15, 18)]}},
        }
        mock_file_path_dict = {1: "file1", 2: "file2"}
        mock_dataset_type_dict = {1: MagicMock(), 2: MagicMock()}
        mock_parsing_func_dict = {1: MagicMock(return_value="parsed1"), 2: MagicMock(return_value="parsed2")}
        mock_server_connection = MagicMock()

        mock_dataset_type_dict[1].read_ranges_from_files.side_effect = [
            iter(["instance1", "instance2"]),
            iter(["instance3"]),
            iter(["instance4"]),
        ]
        mock_dataset_type_dict[2].read_ranges_from_files.return_value = iter(["instance5"])

        result_chunk = ResultChunk(
            MagicMock(),
            mock_dataset_type_dict,
            mock_file_path_dict,
            mock_parsing_func_dict,
            self.chunk_size,
            MagicMock(),
        )
        result_chunk._result_index = mock_result_index
        result_chunk._server_connection = mock_server_connection

        results = list(result_chunk._iterate_single_threaded_overall_mixture())

        expected_results = ["instance1", "instance2", "instance3", "instance4", "instance5"]
        self.assertCountEqual(results, expected_results)

        mock_dataset_type_dict[1].read_ranges_from_files.assert_has_calls(
            [
                unittest.mock.call({"file1": [(0, 2)]}, mock_parsing_func_dict[1], mock_server_connection),
                unittest.mock.call({"file2": [(6, 10)]}, mock_parsing_func_dict[1], mock_server_connection),
                unittest.mock.call({"file2": [(15, 18)]}, mock_parsing_func_dict[1], mock_server_connection),
            ],
            any_order=True,
        )
        mock_dataset_type_dict[2].read_ranges_from_files.assert_called_once_with(
            {"file1": [(7, 8)]}, mock_parsing_func_dict[2], mock_server_connection
        )

    def test_get_element_counts(self):
        # Mocking the Mixture class and its method mixture_in_rows
        mock_mixture = {"property1": 5, "property2": 3, "property3": 2}
        mock_dataset_type_dict = {1: MagicMock(), 2: MagicMock()}
        mock_parsing_func_dict = {1: MagicMock(return_value="parsed1"), 2: MagicMock(return_value="parsed2")}
        mock_file_path_dict = {1: "file1", 2: "file2"}

        # Creating an instance of the class that contains the method to be tested
        result_chunk = ResultChunk(
            MagicMock(),
            mock_dataset_type_dict,
            mock_file_path_dict,
            mock_parsing_func_dict,
            self.chunk_size,
            mock_mixture,
        )
        result_chunk._window_size = 10  # Assuming a window size of 10 for this test

        # Expected result calculation explanation:
        # property1: 0.5 * 10 = 5
        # property2: 0.3 * 10 = 3
        # property3: 0.2 * 10 = 2
        # Total = 10, so no need to adjust the first property
        expected_element_counts = [("property1", 5), ("property2", 3), ("property3", 2)]

        # Executing the method under test
        element_counts = result_chunk._get_element_counts()

        # Asserting that the method returns the expected result
        self.assertEqual(element_counts, expected_element_counts)


if __name__ == "__main__":
    unittest.main()
