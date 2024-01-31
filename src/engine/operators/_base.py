from src.engine.datasets import MixteraDataset


class Operator:
    def __init__(self, ds: MixteraDataset) -> None:
        self.ds = ds
        self.children = []

    def __repr__(self):
        return f"{self.__class__.__name__}"

class SelectOperator(Operator):
    def __init__(self, ds: MixteraDataset, condition) -> None:
        super().__init__(ds)
        self.condition = condition

    def apply(self):
        return self.ds.find_by_key(self.condition)

    def insert(self, left):
        # for select operator, insertion means adding a child to the left operator
        left.insert(self)
        return left
    
    def __repr__(self):
        return f"select<{self.ds}>({self.condition})"



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