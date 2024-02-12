from mixtera.core.processing import ExecutionMode


def test_enum_values():
    assert ExecutionMode.LOCAL.value == 1
    assert ExecutionMode.RAY.value == 2


def test_enum_names():
    assert ExecutionMode.LOCAL.name == "LOCAL"
    assert ExecutionMode.RAY.name == "RAY"
