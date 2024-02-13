import json
import tempfile
import unittest
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
from mixtera.core.processing.property_calculation import LocalPropertyCalculationExecutor


class TestLocalPropertyCalculationExecutor(unittest.TestCase):

    def setUp(self):
        self.setup_func = MagicMock()
        self.calc_func = MagicMock()
        self.dop = 1
        self.batch_size = 2
        self.executor = LocalPropertyCalculationExecutor(self.dop, self.batch_size, self.setup_func, self.calc_func)

    def test_initialization_calls_setup(self):
        self.setup_func.assert_called_once_with(self.executor)

    def test_load_data(self):
        test_files = [(0, "test_file_0.jsonl"), (1, "test_file_1.jsonl")]

        mocked_samples = [(0, 0, "line0"), (0, 1, "line1"), (1, 0, "line2"), (1, 1, "line3")]

        with patch.object(
            LocalPropertyCalculationExecutor, "_read_samples_from_file", return_value=iter(mocked_samples)
        ) as mock_read:
            self.executor.load_data(test_files, data_only_on_primary=True)
            mock_read.assert_called()
            self.assertEqual(len(self.executor._batches), 2)
            for idx, batch in enumerate(self.executor._batches):
                expected_data = np.array(["line0", "line1"]) if idx == 0 else np.array(["line2", "line3"])
                expected_file_id = np.array([idx, idx])
                expected_line_id = np.array([0, 1])

                np.testing.assert_array_equal(batch["data"], expected_data)
                np.testing.assert_array_equal(batch["file_id"], expected_file_id)
                np.testing.assert_array_equal(batch["line_id"], expected_line_id)

    @patch("builtins.open", new_callable=mock_open, read_data="sample1\nsample2\n")
    @patch("pathlib.Path.exists", return_value=True)
    def test_read_samples_from_valid_file(self, mock_file, mock_exists):  # pylint: disable=unused-argument
        file_id = 0
        file = "test_file.jsonl"

        # Call the method
        samples = list(self.executor._read_samples_from_file(file_id, file))

        # Check the returned values
        self.assertEqual(samples, [(0, 0, "sample1"), (0, 1, "sample2")])

    def test_read_samples_from_invalid_file_raises_runtime_error(self):
        with self.assertRaises(RuntimeError):
            list(LocalPropertyCalculationExecutor._read_samples_from_file(0, "nonexistent.jsonl"))

    def test_read_samples_from_non_jsonl_file_raises_not_implemented_error(self):
        with self.assertRaises(NotImplementedError):
            list(LocalPropertyCalculationExecutor._read_samples_from_file(0, "file.txt"))

    def test_run_aggregates_results(self):
        self.executor._batches = [
            {
                "data": np.array(["line0", "line1", "line0"]),
                "file_id": np.array([0, 0, 1]),
                "line_id": np.array([0, 1, 0]),
            }
        ]
        self.executor._calc_func = lambda executor, batch: [f"{sample}_calc" for sample in batch["data"]]
        result = self.executor.run()
        expected_result = {"line0_calc": [(0, 0, 1), (1, 0, 1)], "line1_calc": [(0, 1, 2)]}
        self.assertDictEqual(result, expected_result)

    def test_end_to_end(self):
        sample_data = ['{"name": "sample1"}', '{"name": "sample2"}']
        with (
            tempfile.NamedTemporaryFile(suffix=".jsonl", delete=True, mode="w") as temp_file1,
            tempfile.NamedTemporaryFile(suffix=".jsonl", delete=True, mode="w") as temp_file2,
        ):

            temp_file1.write("\n".join(sample_data))
            temp_file1.seek(0)  # Set file pointer to beginning

            temp_file2.write("\n".join(sample_data))
            temp_file2.seek(0)  # Set file pointer to beginning

            def setup_func(executor):
                executor.prefix = "pref_"

            def calc_func(executor, batch):
                return [executor.prefix + json.loads(sample)["name"] for sample in batch["data"]]

            executor = LocalPropertyCalculationExecutor(1, 2, setup_func, calc_func)

            executor.load_data([(0, temp_file1.name), (1, temp_file2.name)], data_only_on_primary=True)
            result = executor.run()

            expected_result = {"pref_sample1": [(0, 0, 1), (1, 0, 1)], "pref_sample2": [(0, 1, 2), (1, 1, 2)]}
            self.assertDictEqual(result, expected_result)
