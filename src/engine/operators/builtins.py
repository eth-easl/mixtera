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

    def process(self, condition):
        print(f"selecting by {condition}")

    def insert(self, parent):
        """
        .select(condition) -> select<condition> -> remaining query plan
        """
        self.children.append(parent)
        return self


class UnionOperator(Operator):
    def __init__(self, ds: MixteraDataset) -> None:
        super().__init__(ds)

    def apply(self, operator_a, operator_b):
        # check if the apply method of operator_a and operator_b returned two lists
        res_a, res_b = operator_a.apply(), operator_b.apply()
        assert isinstance(res_a, list) and isinstance(res_b, list)
        # deduplicate
        return list(set(res_a + res_b))

    def insert(self, left):
        # for union operator, insertion means adding the left operator as a child to the current union operator
        self.children.append(left)


register(Select)
