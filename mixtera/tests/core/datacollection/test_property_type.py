from mixtera.core.datacollection import PropertyType


def test_enum_values():
    assert PropertyType.CATEGORICAL.value == 1
    assert PropertyType.NUMERICAL.value == 2


def test_enum_names():
    assert PropertyType.CATEGORICAL.name == "CATEGORICAL"
    assert PropertyType.NUMERICAL.name == "NUMERICAL"
