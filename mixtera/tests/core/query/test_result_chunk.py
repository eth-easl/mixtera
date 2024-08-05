import multiprocessing as mp
import unittest
from queue import Empty
from unittest.mock import MagicMock, patch

from mixtera.core.client.server import ServerStub
from mixtera.core.query.result_chunk import END_OF_STREAM_OBJECT, ResultChunk


class TestResultChunk(unittest.TestCase):
    def setUp(self):
        self.chunker_index = MagicMock()
        self.dataset_type_dict = {1: MagicMock()}
        self.file_path_dict = {1: "path/to/file"}
        self.parsing_func_dict = {1: MagicMock()}
        self.mixture = MagicMock()
        self.chunk_size = 10

        self.result_streaming_args = MagicMock()
        self.mock_client = MagicMock(type=ServerStub)

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

        self.result_streaming_args.chunk_reading_degree_of_parallelism = 1
        self.result_streaming_args.chunk_reading_per_window_mixture = True
        self.result_streaming_args.chunk_reading_window_size = -1
        self.result_streaming_args.tunnel_via_server = False
        result_chunk.configure_result_streaming(self.mock_client, self.result_streaming_args)

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

        self.result_streaming_args.chunk_reading_degree_of_parallelism = 2
        self.result_streaming_args.chunk_reading_per_window_mixture = False
        self.result_streaming_args.chunk_reading_window_size = 128
        self.result_streaming_args.tunnel_via_server = False
        result_chunk.configure_result_streaming(self.mock_client, self.result_streaming_args)

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

    def test_iterate_samples_per_window_mixture_true(self):
        result_chunk = ResultChunk(
            self.chunker_index,
            self.dataset_type_dict,
            self.file_path_dict,
            self.parsing_func_dict,
            self.chunk_size,
            None,
        )

        result_chunk._per_window_mixture = True
        mock_iterators = [("property1", iter(["sample1", "sample2"]))]
        result_chunk._init_active_iterators = MagicMock()
        result_chunk._init_active_iterators.return_value = mock_iterators
        result_chunk._iterate_window_mixture = MagicMock()
        result_chunk._iterate_window_mixture.return_value = iter(["sample1", "sample2"])

        results = list(result_chunk._iterate_samples())

        result_chunk._iterate_window_mixture.assert_called_once_with(mock_iterators)
        self.assertEqual(results, ["sample1", "sample2"])

    def test_iterate_samples_per_window_mixture_false(self):
        result_chunk = ResultChunk(
            self.chunker_index,
            self.dataset_type_dict,
            self.file_path_dict,
            self.parsing_func_dict,
            self.chunk_size,
            None,
        )

        result_chunk._per_window_mixture = False
        mock_iterators = [("property2", iter(["sample3", "sample4"]))]
        result_chunk._init_active_iterators = MagicMock()
        result_chunk._init_active_iterators.return_value = mock_iterators
        result_chunk._iterate_overall_mixture = MagicMock()
        result_chunk._iterate_overall_mixture.return_value = iter(["sample3", "sample4"])

        results = list(result_chunk._iterate_samples())

        result_chunk._iterate_overall_mixture.assert_called_once_with(mock_iterators)
        self.assertEqual(results, ["sample3", "sample4"])

    def test_get_active_iterators_st(self):
        result_chunk = ResultChunk(
            self.chunker_index,
            self.dataset_type_dict,
            self.file_path_dict,
            self.parsing_func_dict,
            self.chunk_size,
            None,
        )
        result_chunk._degree_of_parallelism = 1
        mock_workloads = {"property1": "workload1", "property2": "workload2"}
        result_chunk._prepare_workloads = MagicMock(return_value=mock_workloads)

        iter1 = iter([])
        iter2 = iter([])
        expected_iterators = {"property1": iter1, "property2": iter2}
        result_chunk._get_iterator_for_workload_st = MagicMock()
        result_chunk._get_iterator_for_workload_st.side_effect = [iter1, iter2]

        active_iterators = result_chunk._init_active_iterators()

        self.assertEqual(active_iterators, expected_iterators)

    def test_get_active_iterators_mt(self):
        result_chunk = ResultChunk(
            self.chunker_index,
            self.dataset_type_dict,
            self.file_path_dict,
            self.parsing_func_dict,
            self.chunk_size,
            None,
        )
        result_chunk._degree_of_parallelism = 2  # Trigger the mt path
        mock_workloads = {"property1": "workload1", "property2": "workload2"}
        result_chunk._prepare_workloads = MagicMock(return_value=mock_workloads)

        # Simulate the processes for each workload
        mock_processes = {
            "property1": [(MagicMock(), MagicMock())],  # Each tuple represents a (Queue, Process)
            "property2": [(MagicMock(), MagicMock())],
        }
        result_chunk._spin_up_readers = MagicMock(return_value=mock_processes)

        result_chunk._mixture = {"property1": 0.5, "property2": 0.5}

        iter1 = iter([])
        iter2 = iter([])
        expected_iterators = {"property1": iter1, "property2": iter2}
        # Mock _get_iterator_for_workload_mt to return the iterators
        result_chunk._get_iterator_for_workload_mt = MagicMock()
        result_chunk._get_iterator_for_workload_mt.side_effect = [iter1, iter2]

        active_iterators = result_chunk._init_active_iterators()

        self.assertEqual(active_iterators, expected_iterators)

    def test_get_process_counts(self):
        result_chunk = ResultChunk(
            self.chunker_index,
            self.dataset_type_dict,
            self.file_path_dict,
            self.parsing_func_dict,
            self.chunk_size,
            None,
        )
        result_chunk._degree_of_parallelism = 4  # Trigger the mt path
        result_chunk._mixture = {"property1": 5, "property2": 5}

        expected_process_counts = {"property1": 2, "property2": 2}

        # Mocking mp.cpu_count() to return a fixed value
        mp.cpu_count = MagicMock(return_value=4)

        process_counts = result_chunk._get_process_counts()
        self.assertEqual(process_counts, expected_process_counts)

    def test_get_iterator_for_workload_st(self):
        result_chunk = ResultChunk(
            result_index=MagicMock(),
            dataset_type_dict={1: MagicMock(), 2: MagicMock()},
            file_path_dict={1: "path/to/file1", 2: "path/to/file2"},
            parsing_func_dict={1: MagicMock(), 2: MagicMock()},
            chunk_size=MagicMock(),
        )

        workload = [(1, 1, ["range1", "range2"]), (2, 2, ["range3"])]
        expected_data = ["data1", "data2", "data3"]

        # Setup mock return values
        result_chunk._dataset_type_dict[1].read_ranges_from_files.return_value = iter(["data1", "data2"])
        result_chunk._dataset_type_dict[2].read_ranges_from_files.return_value = iter(["data3"])

        # Collecting data from generator
        data_collected = list(result_chunk._get_iterator_for_workload_st(workload))

        self.assertEqual(data_collected, expected_data)

    def test_get_iterator_for_workload_mt(self):
        # Mocking the Queue and Process
        mock_queue1 = MagicMock()
        mock_queue2 = MagicMock()
        mock_proc1 = MagicMock()
        mock_proc2 = MagicMock()

        mock_proc1.is_alive.return_value = False
        mock_proc2.is_alive.return_value = False

        # Setting up the mock queues to return values then raise Empty
        mock_queue1.get.side_effect = ["data1", "data2", END_OF_STREAM_OBJECT]
        mock_queue1.empty.side_effect = [False, False, True]
        mock_queue2.get.side_effect = ["data3", END_OF_STREAM_OBJECT]
        mock_queue2.empty.side_effect = [False, True]

        processes = [(mock_queue1, mock_proc1), (mock_queue2, mock_proc2)]

        result_chunk = ResultChunk(
            result_index=MagicMock(),
            dataset_type_dict={1: MagicMock(), 2: MagicMock()},
            file_path_dict={1: "path/to/file1", 2: "path/to/file2"},
            parsing_func_dict={1: MagicMock(), 2: MagicMock()},
            chunk_size=MagicMock(),
        )

        # Collecting data from generator
        data_collected = list(result_chunk._get_iterator_for_workload_mt(processes))

        expected_data = ["data1", "data3", "data2"]
        self.assertEqual(data_collected, expected_data)

    def test_get_iterator_for_workload_mt_with_timeout_and_process_alive(self):
        # Mocking the Queue and Process
        mock_queue1 = MagicMock()
        mock_proc1 = MagicMock()

        mock_proc1.is_alive.return_value = True

        # Setting up the mock queue to timeout then return a value
        mock_queue1.get.side_effect = [Empty, "data1", END_OF_STREAM_OBJECT]
        mock_queue1.empty.return_value = False

        processes = [(mock_queue1, mock_proc1)]

        result_chunk = ResultChunk(
            result_index=MagicMock(),
            dataset_type_dict={1: MagicMock(), 2: MagicMock()},
            file_path_dict={1: "path/to/file1", 2: "path/to/file2"},
            parsing_func_dict={1: MagicMock(), 2: MagicMock()},
            chunk_size=MagicMock(),
        )

        with self.assertRaises(RuntimeError) as context:
            list(result_chunk._get_iterator_for_workload_mt(processes))

        self.assertTrue("Queue timeout reached but process is still alive." in str(context.exception))

    def test_get_iterator_for_workload_mt_process_dies_before_end_of_stream(self):
        # Mocking the Queue and Process
        mock_queue1 = MagicMock()
        mock_proc1 = MagicMock()

        mock_proc1.is_alive.side_effect = [True, True, False]  # Process dies after first check

        # Setting up the mock queue to return a value then simulate process death before END_OF_STREAM_OBJECT
        mock_queue1.get.side_effect = ["data1", Empty]
        mock_queue1.empty.return_value = False

        processes = [(mock_queue1, mock_proc1)]

        result_chunk = ResultChunk(
            result_index=MagicMock(),
            dataset_type_dict={1: MagicMock(), 2: MagicMock()},
            file_path_dict={1: "path/to/file1", 2: "path/to/file2"},
            parsing_func_dict={1: MagicMock(), 2: MagicMock()},
            chunk_size=MagicMock(),
        )

        # Collecting data from generator
        self.assertRaises(RuntimeError, list, result_chunk._get_iterator_for_workload_mt(processes))

    def test_iterate_window_mixture(self):
        result_chunk = ResultChunk(
            result_index=MagicMock(),
            dataset_type_dict={1: MagicMock(), 2: MagicMock()},
            file_path_dict={1: "path/to/file1", 2: "path/to/file2"},
            parsing_func_dict={1: MagicMock(), 2: MagicMock()},
            chunk_size=MagicMock(),
        )

        # Mocking the active_iterators
        active_iterators = {"prop1": iter(["data1", "data2"]), "prop2": iter(["data3"])}

        # Mocking _get_element_counts to return a specific distribution of elements
        result_chunk._get_element_counts = MagicMock(return_value=[("prop1", 2), ("prop2", 1)])

        # Expected data to be yielded from the iterator
        expected_data = ["data3", "data1", "data2"]

        # Collecting data from generator
        data_collected = list(result_chunk._iterate_window_mixture(active_iterators))

        self.assertEqual(data_collected, expected_data)

    def test_iterate_window_mixture_multiple_windows(self):
        result_chunk = ResultChunk(
            result_index=MagicMock(),
            dataset_type_dict={1: MagicMock(), 2: MagicMock(), 3: MagicMock()},
            file_path_dict={1: "path/to/file1", 2: "path/to/file2", 3: "path/to/file3"},
            parsing_func_dict={1: MagicMock(), 2: MagicMock(), 3: MagicMock()},
            chunk_size=MagicMock(),
        )

        # Mocking the active_iterators with more data to test multiple windows
        active_iterators = {
            "prop1": iter(["data1", "data2", "data3"]),
            "prop2": iter(["data4", "data5"]),
            "prop3": iter(["data6"]),  # Less data to ensure some properties finish before others
        }

        # Mocking _get_element_counts to return a specific distribution of elements
        result_chunk._get_element_counts = MagicMock(return_value=[("prop1", 2), ("prop2", 1), ("prop3", 1)])

        result_chunk._window_size = 4  # Assuming a window size of 2 for this test

        # Expected data to be yielded from the iterator, considering the window size and distribution
        expected_data = ["data1", "data6", "data4", "data2", "data3", "data5"]

        # Collecting data from generator
        data_collected = list(result_chunk._iterate_window_mixture(active_iterators))

        self.assertEqual(data_collected, expected_data)

    def test_iterate_overall_mixture(self):
        result_chunk = ResultChunk(
            result_index=MagicMock(),
            dataset_type_dict={1: MagicMock(), 2: MagicMock(), 3: MagicMock()},
            file_path_dict={1: "path/to/file1", 2: "path/to/file2", 3: "path/to/file3"},
            parsing_func_dict={1: MagicMock(), 2: MagicMock(), 3: MagicMock()},
            chunk_size=MagicMock(),
        )

        # Mocking the active_iterators with different lengths to simulate varied data sources
        active_iterators = {
            "prop1": iter(["data1", "data2", "data3"]),
            "prop2": iter(["data4"]),
            "prop3": iter(["data5", "data6"]),
        }

        # Since the order is shuffled and reproducibly random, we need to mock the randomness to predict the output
        seed_everything_mock = MagicMock()
        generate_hash_string_from_list_mock = MagicMock(return_value="some_hash_value")
        # Mock shuffle to sort instead, for predictability
        random_shuffle_mock = MagicMock(side_effect=lambda x: x.sort())

        with (
            patch("mixtera.utils.seed_everything", seed_everything_mock),
            patch("mixtera.utils.generate_hash_string_from_list", generate_hash_string_from_list_mock),
            patch("random.shuffle", random_shuffle_mock),
        ):

            # Expected data to be yielded from the iterator, considering the mocked shuffle (sort)
            expected_data = ["data1", "data4", "data5", "data2", "data6", "data3"]

            # Collecting data from generator
            data_collected = list(result_chunk._iterate_overall_mixture(active_iterators))

        self.assertEqual(data_collected, expected_data)

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
