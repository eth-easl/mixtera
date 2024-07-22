import asyncio
import struct
from typing import Any, Optional

import dill


async def read_bytes(num_bytes: int, reader: asyncio.StreamReader, timeout: float = 10.0) -> Optional[bytearray]:
    """
    Asynchronously read exactly `num_bytes` from `asyncio.StreamReader`, with a timeout.
    The difference to reader.readexactly() is that we do not assume all data is available in the stream yet,
    to avoid race conditions.
    Args:
        reader (asyncio.StreamReader): The stream reader from which to read.
        num_bytes (int): The exact number of bytes to read.
        timeout (float): The number of seconds to wait before timing out.
    Returns:
        bytearray: The read bytes. None, if connection is closed.
    Raises:
        asyncio.TimeoutError: If the timeout is exceeded before `num_bytes` could be read.
    """
    buffer = bytearray()
    end_time = asyncio.get_event_loop().time() + timeout

    while len(buffer) < num_bytes:
        # Calculate remaining time before the timeout occurs.
        remaining_time = end_time - asyncio.get_event_loop().time()
        if remaining_time <= 0:
            raise asyncio.TimeoutError("Reading bytes timed out")

        # Read up to the remaining number of bytes or whatever is available.
        chunk = await asyncio.wait_for(reader.read(num_bytes - len(buffer)), timeout=remaining_time)
        buffer.extend(chunk)

        if not chunk:
            # If an empty chunk is returned, it means the stream has closed.
            # This should only happen if no data has been read yet (invariant)
            if len(buffer) == 0:
                # all good, inform caller
                return None

            raise ConnectionError(
                "Connection closed while we still have " + f"{num_bytes - len(buffer)} bytes to read."
            )

    return buffer


async def write_bytes_obj(
    data: bytes | None, size_bytes: int, writer: asyncio.StreamWriter, drain: bool = True
) -> None:
    """
    Asynchronously writes a bytes object or None to the stream writer.

    This function first writes the size of the data as a header, then writes the data itself.
    If the data is None, it writes a zero-length header.

    Args:
        data (bytes | None): The bytes object to write, or None.
        size_bytes (int): The number of bytes to use for the size header.
        writer (asyncio.StreamWriter): The stream writer to write the data to.
        drain (bool): Whether to call writer.drain() after writing. Defaults to True.

    Returns:
        None

    Note:
        If data is None, only a zero-length header is written.
        The size header is always written in big-endian format.
    """
    if data is None:
        zero_len = 0
        writer.write(zero_len.to_bytes(size_bytes, "big"))
    else:
        writer.write(len(data).to_bytes(size_bytes, "big"))
        writer.write(data)

    if drain:
        await writer.drain()


async def read_bytes_obj(size_bytes: int, reader: asyncio.StreamReader) -> bytes | None:
    """
    Asynchronously reads a bytes object or None from the stream reader.

    This function first reads the size header, then reads the actual data based on the size.
    If the size header indicates zero length, it returns None.

    Args:
        size_bytes (int): The number of bytes used for the size header.
        reader (asyncio.StreamReader): The stream reader to read the data from.

    Returns:
        bytes | None: The read bytes object, or None if the size header indicated zero length
                      or if an error occurred during reading.

    Raises:
        asyncio.TimeoutError: If a timeout occurs while reading (inherited from read_int and read_bytes).

    Note:
        The size header is expected to be in big-endian format.
    """

    if (obj_size := await read_int(size_bytes, reader)) is not None:
        if obj_size == 0:
            return None
        return await read_bytes(obj_size, reader)
    return None


async def read_int(num_bytes: int, reader: asyncio.StreamReader, timeout: float = 10.0) -> Optional[int]:
    """
    Asynchronously read exactly `num_bytes` from `asyncio.StreamReader`, with a timeout, and parses this to an int.
    Args:
        num_bytes (int): The exact number of bytes to read.
        reader (asyncio.StreamReader): The stream reader from which to read.
        timeout (float): The number of seconds to wait before timing out.
    Returns:
        Optional[int]: The read integer. None, if error occurs.
    Raises:
        asyncio.TimeoutError: If the timeout is exceeded before `num_bytes` could be read.
    """

    return (
        int.from_bytes(bytes_data, byteorder="big", signed=True)
        if (bytes_data := await read_bytes(num_bytes, reader, timeout=timeout)) is not None
        else None
    )


async def write_int(data: int, num_bytes: int, writer: asyncio.StreamWriter, drain: bool = True) -> None:
    """
    Asynchronously writes an integer with exactly `num_bytes` using a asyncio.StreamWriter.
    Args:
        data (int): The integer to write.
        num_bytes (int): How many bytes to serialize the int to.
        writer (asyncio.StreamWriter): The stream writer which should write the data.
        drain (bool): Whether to call writer.drain() afterwards. Defaults to True.
    """
    writer.write(data.to_bytes(num_bytes, "big", signed=True))
    if drain:
        await writer.drain()


async def read_utf8_string(size_bytes: int, reader: asyncio.StreamReader) -> Optional[str]:
    """
    Asynchronously read an utf8 string from `asyncio.StreamReader`.
    Args:
        size_bytes (int): The size of the header in bytes.
        reader (asyncio.StreamReader): The stream reader from which to read.
    Returns:
        Optional[str]: The read string. None, if error occurs.
    Raises:
        asyncio.TimeoutError: If the timeout is exceeded before `num_bytes` could be read.
    """
    if (string_size := await read_int(size_bytes, reader)) is not None:
        if (string_data := await read_bytes(string_size, reader)) is not None:
            return string_data.decode("utf-8")
    return None


async def write_utf8_string(string: str, size_bytes: int, writer: asyncio.StreamWriter, drain: bool = True) -> None:
    """
    Asynchronously writes an utf8 string using a asyncio.StreamWriter.
    Args:
        string (str): The string to write.
        size_bytes (int): How many bytes the header should be.
        writer (asyncio.StreamWriter): The stream writer which should write the data.
        drain (bool): Whether to call writer.drain() afterwards. Defaults to True.
    """

    training_id_bytes = string.encode(encoding="utf-8")
    writer.write(len(training_id_bytes).to_bytes(size_bytes, "big"))
    writer.write(training_id_bytes)

    if drain:
        await writer.drain()


async def write_pickeled_object(obj: Any, size_bytes: int, writer: asyncio.StreamWriter, drain: bool = True) -> None:
    """
    Asynchronously writes an arbitrary Python object (pickled using dill) using a asyncio.StreamWriter.
    Args:
        obj (Any): The object to write.
        size_bytes (int): How many bytes the header should be.
        writer (asyncio.StreamWriter): The stream writer which should write the data.
        drain (bool): Whether to call writer.drain() afterwards. Defaults to True.
    """
    obj_bytes = dill.dumps(obj)
    writer.write(len(obj_bytes).to_bytes(size_bytes, "big"))
    writer.write(obj_bytes)

    if drain:
        await writer.drain()


async def read_pickeled_object(size_bytes: int, reader: asyncio.StreamReader) -> Any:
    """
    Asynchronously read an arbitrary pickeld Python object from `asyncio.StreamReader`.
    Args:
        size_bytes (int): The size of the header in bytes.
        reader (asyncio.StreamReader): The stream reader from which to read.
    Returns:
        Optional[str]: The read string. None, if error occurs.
    Raises:
        asyncio.TimeoutError: If the timeout is exceeded before `num_bytes` could be read.
    """
    if (obj_size := await read_int(size_bytes, reader)) is not None:
        if (obj_data := await read_bytes(obj_size, reader)) is not None:
            return dill.loads(obj_data)
    return None


async def write_float(data: float, writer: asyncio.StreamWriter, drain: bool = True) -> None:
    """
    Asynchronously writes a float using a asyncio.StreamWriter.

    Does not require a size header, as double precision floats are always 8 bytes long.

    Args:
        data (float): The float to write.
        writer (asyncio.StreamWriter): The stream writer which should write the data.
        drain (bool): Whether to call writer.drain() afterwards. Defaults to True.
    """
    #  Pack the float into 8 bytes using big-endian format
    bytes_data = struct.pack(">d", data)
    writer.write(bytes_data)
    if drain:
        await writer.drain()


async def read_float(reader: asyncio.StreamReader, timeout: float = 10.0) -> Optional[float]:
    """
    Asynchronously read a float from `asyncio.StreamReader`.

    Does not require a size header, as double precision floats are always 8 bytes long.

    Args:
        reader (asyncio.StreamReader): The stream reader from which to read.
    Returns:
        Optional[float]: The read float. None, if error occurs.
    Raises:
        asyncio.TimeoutError: If the timeout is exceeded before `num_bytes` could be read.
    """
    # Read 8 bytes from the reader
    bytes_data = await read_bytes(8, reader, timeout=timeout)
    if bytes_data is not None and len(bytes_data) == 8:
        # Unpack the bytes into a float using big-endian format
        return struct.unpack(">d", bytes_data)[0]
    return None
