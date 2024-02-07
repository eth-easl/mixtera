from mixtera.datacollection import DatasetTypes


def test_enum_values():
    assert DatasetTypes.JSONL_COLLECTION.value == 1
    assert DatasetTypes.JSONL_SINGLEFILE.value == 2


def test_enum_names():
    assert DatasetTypes.JSONL_COLLECTION.name == "JSONL_COLLECTION"
    assert DatasetTypes.JSONL_SINGLEFILE.name == "JSONL_SINGLEFILE"
