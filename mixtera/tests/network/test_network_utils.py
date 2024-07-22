# pylint: disable=attribute-defined-outside-init

import asyncio
import struct
import unittest
from unittest.mock import AsyncMock, MagicMock, call

import dill
from mixtera.network.network_utils import (
    read_bytes,
    read_float,
    read_int,
    read_pickeled_object,
    read_utf8_string,
    write_float,
    write_int,
    write_pickeled_object,
    write_utf8_string,
)


def create_mock_reader(*args):
    mock_reader = MagicMock(asyncio.StreamReader)
    mock_reader.read = AsyncMock(side_effect=list(args))
    return mock_reader


def create_mock_writer():
    mock_writer = MagicMock(asyncio.StreamWriter)
    mock_writer.drain = AsyncMock()
    mock_writer.write = MagicMock()
    return mock_writer


class TestNetworkUtils(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.reader = create_mock_reader()
        self.writer = create_mock_writer()
        self.timeout = 10.0

    async def test_read_bytes_success(self):
        data = b"1234567890"
        self.reader.read = AsyncMock(return_value=data)
        result = await read_bytes(len(data), self.reader, self.timeout)
        self.assertEqual(result, bytearray(data))

    async def test_read_bytes_multiple(self):
        data = b"1234567890"
        chunked_data = [b"12", b"34", b"5678", b"90"]
        self.reader.read = AsyncMock(side_effect=chunked_data)
        result = await read_bytes(len(data), self.reader, self.timeout)
        self.assertEqual(result, bytearray(data))

    async def test_read_bytes_timeout(self):
        self.reader.read = AsyncMock(side_effect=asyncio.TimeoutError)
        with self.assertRaises(asyncio.TimeoutError):
            await read_bytes(10, self.reader, self.timeout)

    async def test_read_bytes_connection_closed(self):
        self.reader.read = AsyncMock(side_effect=[b"12", b""])
        with self.assertRaises(ConnectionError):
            await read_bytes(10, self.reader, self.timeout)

    async def test_read_int_success(self):
        data = (123).to_bytes(2, byteorder="big", signed=True)
        self.reader.read = AsyncMock(return_value=data)
        result = await read_int(len(data), self.reader, self.timeout)
        self.assertEqual(result, 123)

    async def test_read_int_timeout(self):
        self.reader.read = AsyncMock(side_effect=asyncio.TimeoutError)
        with self.assertRaises(asyncio.TimeoutError):
            await read_int(2, self.reader, self.timeout)

    async def test_write_int_success(self):
        data = 123
        num_bytes = 2
        await write_int(data, num_bytes, self.writer)
        self.writer.write.assert_called_with(data.to_bytes(num_bytes, byteorder="big", signed=True))
        self.writer.drain.assert_awaited_once()

    async def test_read_utf8_string_success(self):
        string = "Test"
        string_data = len(string).to_bytes(4, byteorder="big") + string.encode("utf-8")
        self.reader.read = AsyncMock(side_effect=[string_data[:4], string_data[4:]])
        result = await read_utf8_string(4, self.reader)
        self.assertEqual(result, string)

    async def test_read_utf8_string_failure(self):
        self.reader.read = AsyncMock(side_effect=[b"\x00\x00\x00\x04", asyncio.TimeoutError()])
        with self.assertRaises(asyncio.TimeoutError):
            await read_utf8_string(4, self.reader)

    async def test_write_utf8_string_success(self):
        string = "Test"
        size_bytes = 4
        await write_utf8_string(string, size_bytes, self.writer)
        encoded_string = string.encode("utf-8")
        expected_calls = [call(len(encoded_string).to_bytes(size_bytes, byteorder="big")), call(encoded_string)]
        self.writer.write.assert_has_calls(expected_calls)
        self.writer.drain.assert_awaited_once()

    async def test_write_pickle_object(self):
        obj = {"test": 123}
        size_bytes = 4
        await write_pickeled_object(obj, size_bytes, self.writer)
        pickled_data = dill.dumps(obj)
        expected_calls = [call(len(pickled_data).to_bytes(size_bytes, byteorder="big")), call(pickled_data)]
        self.writer.write.assert_has_calls(expected_calls)
        self.writer.drain.assert_awaited_once()

    async def test_read_pickle_object_success(self):
        obj = {"test": 123}
        pickled_data = dill.dumps(obj)
        size_data = len(pickled_data).to_bytes(4, byteorder="big")
        self.reader.read = AsyncMock(side_effect=[size_data, pickled_data])
        result = await read_pickeled_object(4, self.reader)
        self.assertEqual(result, obj)

    async def test_read_pickle_object_failure(self):
        self.reader.read = AsyncMock(side_effect=[b"\x00\x00\x00\x04", asyncio.TimeoutError()])
        with self.assertRaises(asyncio.TimeoutError):
            await read_pickeled_object(4, self.reader)

    async def test_write_float_success(self):
        data = 123.456
        await write_float(data, self.writer)
        self.writer.write.assert_called_with(struct.pack(">d", data))
        self.writer.drain.assert_awaited_once()

    async def test_read_float_success(self):
        data = 123.456
        data_bytes = struct.pack(">d", data)
        self.reader.read = AsyncMock(return_value=data_bytes)
        result = await read_float(self.reader, self.timeout)
        self.assertEqual(result, data)

    async def test_read_float_timeout(self):
        self.reader.read = AsyncMock(side_effect=asyncio.TimeoutError)
        with self.assertRaises(asyncio.TimeoutError):
            await read_float(self.reader, self.timeout)

    async def test_read_float_connection_closed(self):
        self.reader.read = AsyncMock(side_effect=[b"\x00\x00\x00\x04", b""])
        with self.assertRaises(ConnectionError):
            await read_float(self.reader, self.timeout)
