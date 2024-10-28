import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from mixtera.hf import MixteraHFDataset
from mixtera.torch import MixteraTorchDataset
from mixtera.utils.checkpoint import _get_mixtera_hf_dataset_or_client_from_iterabledataset, handle_mixtera_checkpoint


class TestCheckpointHelpers(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        self.checkpoint_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_get_mixtera_hf_dataset_already_mixtera(self):
        """Test when the dataset is already a MixteraHFDataset."""
        mock_dataset_instance = MagicMock(spec=MixteraHFDataset)
        result = _get_mixtera_hf_dataset_or_client_from_iterabledataset(mock_dataset_instance)
        self.assertEqual(result, mock_dataset_instance)

    def test_get_mixtera_hf_dataset_nested(self):
        """Test when the MixteraHFDataset is nested inside iterables."""
        mock_mhfd = MagicMock(spec=MixteraHFDataset)

        class MockIterableDataset:
            def __init__(self, ex_iterable=None, _ex_iterable=None):
                self.ex_iterable = ex_iterable
                self._ex_iterable = _ex_iterable

        dataset = MockIterableDataset(_ex_iterable=MockIterableDataset(_ex_iterable=mock_mhfd))

        result = _get_mixtera_hf_dataset_or_client_from_iterabledataset(dataset)
        self.assertEqual(result, mock_mhfd)

    def test_get_mixtera_hf_dataset_not_found(self):
        """Test when no MixteraHFDataset is found."""

        class MockIterableDataset:
            def __init__(self, ex_iterable=None, _ex_iterable=None):
                self.ex_iterable = ex_iterable
                self._ex_iterable = _ex_iterable

        dataset = MockIterableDataset(_ex_iterable=MockIterableDataset())

        result = _get_mixtera_hf_dataset_or_client_from_iterabledataset(dataset)

        self.assertIsNone(result)

    @patch("mixtera.utils.checkpoint._recover_mixtera_dataset")
    def test_handle_mixtera_checkpoint_no_dataset(self, mock_recover):
        """Test handle_mixtera_checkpoint when no Mixtera dataset is recovered."""
        mock_recover.return_value = None

        handle_mixtera_checkpoint(
            dataloader_or_dataset=MagicMock(),
            checkpoint_path=self.checkpoint_path,
            dp_group_id=0,
            node_id=0,
            wait_for_disk=True,
        )
        # Since dataset is None, the function should return early and not throw any exceptions.
        mock_recover.assert_called_once()

    def test_handle_mixtera_checkpoint_success(self):
        """Test handle_mixtera_checkpoint with successful checkpointing."""
        mock_dataset = MagicMock(spec=MixteraTorchDataset)
        mock_client = MagicMock()
        mock_dataset.worker_status = [0, 1, 2]
        mock_query = MagicMock()
        mock_query.job_id = "test_job_id"
        mock_dataset._query = mock_query
        mock_dataset._client = mock_client

        mock_client.checkpoint.return_value = "test_checkpoint_id"
        mock_client.checkpoint_completed.side_effect = [False, True]

        with patch("mixtera.utils.checkpoint._recover_mixtera_dataset", return_value=mock_dataset):
            handle_mixtera_checkpoint(
                dataloader_or_dataset=mock_dataset,
                checkpoint_path=self.checkpoint_path,
                dp_group_id=0,
                node_id=0,
                wait_for_disk=True,
            )

            mock_client.checkpoint.assert_called_with("test_job_id", 0, 0, [0, 1, 2])
            self.assertEqual(mock_client.checkpoint_completed.call_count, 2)
            mock_client.checkpoint_completed.assert_called_with("test_job_id", "test_checkpoint_id", True)

            mixtera_id_path = self.checkpoint_path / "mixtera.id"
            self.assertTrue(mixtera_id_path.exists())
            with open(mixtera_id_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                self.assertEqual(content, "test_checkpoint_id")
