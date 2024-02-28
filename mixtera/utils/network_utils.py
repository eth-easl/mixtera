import asyncio
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


async def read_int(num_bytes: int, reader: asyncio.StreamReader, timeout: float = 10.0) -> Optional[int]:
    return (
        int.from_bytes(bytes_data, byteorder="big", signed=True)
        if (bytes_data := await read_bytes(num_bytes, reader, timeout=timeout)) is not None
        else None
    )


async def write_int(data: int, num_bytes: int, writer: asyncio.StreamWriter, drain: bool = True) -> None:
    writer.write(data.to_bytes(num_bytes, "big", signed=True))
    if drain:
        await writer.drain()


async def read_utf8_string(size_bytes: int, reader: asyncio.StreamReader) -> Optional[str]:
    if (string_size := await read_int(size_bytes, reader)) is not None:
        if (string_data := await read_bytes(string_size, reader)) is not None:
            return string_data.decode("utf-8")
    return None


async def write_utf8_string(string: str, size_bytes: int, writer: asyncio.StreamWriter, drain: bool = True) -> None:
    training_id_bytes = string.encode(encoding="utf-8")
    writer.write(len(training_id_bytes).to_bytes(size_bytes, "big"))
    writer.write(training_id_bytes)

    if drain:
        await writer.drain()


async def write_pickeled_object(obj: Any, size_bytes: int, writer: asyncio.StreamWriter, drain: bool = True) -> None:
    obj_bytes = dill.dumps(obj)
    writer.write(len(obj_bytes).to_bytes(size_bytes, "big"))
    writer.write(obj_bytes)

    if drain:
        await writer.drain()


async def read_pickeled_object(size_bytes: int, reader: asyncio.StreamReader) -> Any:
    if (obj_size := await read_int(size_bytes, reader)) is not None:
        if (obj_data := await read_bytes(obj_size, reader)) is not None:
            return dill.loads(obj_data)
    return None
