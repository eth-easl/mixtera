from enum import IntEnum, auto


class DatasetType(IntEnum):
    GENERIC_DATASET = 0
    JSONL_DATASET = auto()
    CROISSANT_DATASET = auto()
    WEB_DATASET = auto()
    PARQUET_DATASET = auto()
    CC12M_DATASET = auto()
    MSCOCO_DATASET = auto()
    LAION400M_DATASET = auto()
    COYO700M_DATASET = auto()
    DOMAINNET_DATASET = auto()
