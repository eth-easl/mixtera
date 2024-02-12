import json
import tempfile
import unittest
from unittest.mock import MagicMock, mock_open, patch

from mixtera.core.processing.property_calculation import LocalPropertyCalculationExecutor


class TestLocalPropertyCalculationExecutor(unittest.TestCase):

    def setUp(self):
        self.setup_func = MagicMock()
        self.calc_func = MagicMock()
        self.dop = 1
        self.batch_size = 2
        self.executor = LocalPropertyCalculationExecutor(
            "test_property", self.dop, self.batch_size, self.setup_func, self.calc_func
        )

    def test_initialization_calls_setup(self):
        self.setup_func.assert_called_once_with(self.executor)

    def test_load_data(self):
        with patch.object(
            LocalPropertyCalculationExecutor, "_read_samples_from_file", return_value=["sample1", "sample2"]
        ) as mock_read:
            self.executor.load_data(["file1.jsonl", "file2.jsonl"], data_only_on_primary=True)
            mock_read.assert_called()
            self.assertEqual(len(self.executor._sample_cache), 2)
            self.assertEqual(self.executor._sample_cache, [["sample1", "sample2"], ["sample1", "sample2"]])

    def test_run_aggregates_results(self):
        self.executor._sample_cache = [["sample1"], ["sample2"]]
        self.executor._calc_func = lambda executor, batch: {sample: {sample: [sample]} for sample in batch}
        result = self.executor.run()
        expected_result = {"sample1": {"sample1": ["sample1"]}, "sample2": {"sample2": ["sample2"]}}
        self.assertDictEqual(result, expected_result)

    @patch("builtins.open", new_callable=mock_open, read_data="sample1\nsample2\n")
    @patch("pathlib.Path.exists", return_value=True)
    def test_read_samples_from_valid_file(self, mock_file, mock_exists):  # pylint: disable=unused-argument
        samples = LocalPropertyCalculationExecutor._read_samples_from_file("file.jsonl")
        self.assertEqual(samples, ["sample1", "sample2"])

    def test_read_samples_from_invalid_file_raises_runtime_error(self):
        with self.assertRaises(RuntimeError):
            LocalPropertyCalculationExecutor._read_samples_from_file("nonexistent.jsonl")

    def test_read_samples_from_non_jsonl_file_raises_not_implemented_error(self):
        with self.assertRaises(NotImplementedError):
            LocalPropertyCalculationExecutor._read_samples_from_file("file.txt")

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

            def setup_func(executor):  # pylint: disable=unused-argument
                return None

            def calc_func(executor, batch):  # pylint: disable=unused-argument
                sample_names = [json.loads(sample)["name"] for sample in batch]
                return {sample_name: str(sample_name) for sample_name in sample_names}

            executor = LocalPropertyCalculationExecutor("test_property", 1, 2, setup_func, calc_func)

            executor.load_data([temp_file1.name, temp_file2.name], data_only_on_primary=True)
            result = executor.run()

            expected_result = {"sample1": {"bucket1": [1, 1]}, "sample2": {"bucket1": [1, 1]}}
            self.assertDictEqual(result, expected_result)
