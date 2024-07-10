from enum import IntEnum, auto


class DatasetType(IntEnum):
    GENERIC_DATASET = 0
    JSONL_DATASET = auto()
    CROISSANT_DATASET = auto()
