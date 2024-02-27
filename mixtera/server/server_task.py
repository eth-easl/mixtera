from enum import IntEnum


class ServerTask(IntEnum):
    RegisterQuery = 0
    StreamData = 1
    GetQueryId = 2
