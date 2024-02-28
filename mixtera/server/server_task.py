from enum import IntEnum


class ServerTask(IntEnum):
    REGISTER_QUERY = 0
    STREAM_DATA = 1
    GET_QUERY_ID = 2
