from enum import IntEnum, auto


class ServerTask(IntEnum):
    REGISTER_QUERY = 0
    QUERY_EXEC_STATUS = auto()
    READ_FILE = auto()
    GET_QUERY_ID = auto()
    GET_META_RESULT = auto()
    GET_NEXT_RESULT_CHUNK = auto()
    REGISTER_DATASET = auto()
    DATASET_REGISTRATION_STATUS = auto()
    REGISTER_METADATA_PARSER = auto()
    CHECK_DATASET_EXISTS = auto()
    LIST_DATASETS = auto()
    REMOVE_DATASET = auto()
    ADD_PROPERTY = auto()
    CHECKPOINT = auto()
    CHECKPOINT_COMPLETED = auto()
    RESTORE_CHECKPOINT = auto()
    RECEIVE_FEEDBACK = auto()
