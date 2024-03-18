import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from mixtera.core.client import MixteraClient
from mixtera.core.client.local import LocalStub


class DummyLocalStub(LocalStub):
    def __init__(self, arg: Any) -> None:  # pylint: disable = super-init-not-called
        self.call_arg = arg


class TestMixteraClient(unittest.TestCase):
    @patch("mixtera.core.client.local.LocalStub")
    def test_from_directory_with_existing_dir(self, mock_stub):
        mock_stub.return_value = MagicMock()

        dir_path = Path(".")
        result = MixteraClient.from_directory(dir_path)
        mock_stub.assert_called_once_with(dir_path)
        self.assertIsInstance(result, MagicMock)

    @patch("mixtera.core.client.local.LocalStub", new=DummyLocalStub)
    def test_path_constructor_with_existing_dir(self):
        dir_path = Path(".")
        result = MixteraClient(dir_path)  # pylint: disable = abstract-class-instantiated
        self.assertIsInstance(result, DummyLocalStub)
        self.assertEqual(result.call_arg, dir_path)

    @patch("mixtera.core.client.local.LocalStub", new=DummyLocalStub)
    def test_strpath_constructor_with_existing_dir(
        self,
    ):
        dir_path = str(Path("."))
        result = MixteraClient(dir_path)  # pylint: disable = abstract-class-instantiated
        self.assertIsInstance(result, DummyLocalStub)
        self.assertEqual(result.call_arg, dir_path)

    def test_from_directory_with_non_existing_dir(self):
        dir_path = Path("/non/existing/directory")
        with self.assertRaises(RuntimeError):
            MixteraClient.from_directory(dir_path)

    # TODO(MaxiBoether): write test for server instantiation
