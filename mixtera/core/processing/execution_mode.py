from enum import Enum, auto


class ExecutionMode(Enum):
    LOCAL = 1
    RAY = auto()
