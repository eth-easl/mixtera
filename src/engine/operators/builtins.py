from src.engine.datasets import MixteraDataset
from ._base import Operator, childOp
from .query import register


@childOp
class Select(Operator):
    def __init__(self, condition: str) -> None:
        super().__init__()
        self.condition = condition

    def apply(self):
        return self.ds.find_by_key(self.condition)

    def __repr__(self):
        return f"select<{self.ds}>({self.condition})"

    def insert(self, parent):
        """
        .select(condition) -> select<condition> -> remaining query plan
        """
        self.children.append(parent)
        return self


class Union(Operator):
    def __init__(self, operator_a) -> None:
        super().__init__()

        self.operator_a = operator_a
        self.operator_a.root.display(0)

    def apply(self, operator_a, operator_b):
        # check if the apply method of operator_a and operator_b returned two lists
        res_a, res_b = operator_a.apply(), operator_b.apply()
        assert isinstance(res_a, list) and isinstance(res_b, list)
        # deduplicate
        return list(set(res_a + res_b))

    def __repr__(self):
        return f"union<>()"

    def insert(self, parent):
        # op_a.union(op_b) -> union as new root with two children
        parent.display(0)
        self.operator_a.root.display(0)
        self.children = [self.operator_a.root, parent]
        return self


register(Select)
register(Union)