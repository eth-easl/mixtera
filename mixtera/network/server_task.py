from enum import IntEnum


class ServerTask(IntEnum):
    REGISTER_QUERY = 0
    READ_FILE = 1
    GET_QUERY_ID = 2
    GET_META_RESULT = 3
    GET_NEXT_RESULT_CHUNK = 4