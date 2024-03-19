import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from mixtera.core.client import MixteraClient
from mixtera.core.client.local import LocalStub
from mixtera.core.client.server import ServerStub


class DummyLocalStub(LocalStub):
    def __init__(self, arg: Any) -> None:  # pylint: disable = super-init-not-called
        self.call_arg = arg


class DummyServerStub(ServerStub):
    def __init__(self, host: Any, port: Any) -> None:  # pylint: disable=super-init-not-called
        self.host = host
        self.port = port


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

    @patch("mixtera.core.client.server.ServerStub")
    def test_from_remote(self, mock_stub):
        mock_stub.return_value = MagicMock()

        host = "localhost"
        port = 8080
        result = MixteraClient.from_remote(host, port)
        mock_stub.assert_called_once_with(host, port)
        self.assertIsInstance(result, MagicMock)

    def test_tuple_constructor_with_host_and_port(self):
        # We cannot use our mock class here due to the args/kwargs shenanagans
        host = "localhost"
        port = 8080
        result = MixteraClient((host, port))  # pylint: disable=abstract-class-instantiated
        self.assertIsInstance(result, ServerStub)
        self.assertEqual(result._host, host)
        self.assertEqual(result._port, port)

    @patch("mixtera.core.client.server.ServerStub", new=DummyServerStub)
    def test_two_args_constructor_with_host_and_port(self):
        host = "localhost"
        port = 8080
        result = MixteraClient(host, port)  # pylint: disable=abstract-class-instantiated
        self.assertIsInstance(result, DummyServerStub)
        self.assertEqual(result.host, host)
        self.assertEqual(result.port, port)

    def test_invalid_params(self):
        with self.assertRaises(ValueError):
            MixteraClient(123, "invalid", "invalid")  # pylint: disable=abstract-class-instantiated

        with self.assertRaises(ValueError):
            MixteraClient()  # pylint: disable=abstract-class-instantiated
