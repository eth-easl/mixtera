from ._base import Operator, parent_op
from loguru import logger

@parent_op
class Select(Operator):
    def __init__(self, condition: str) -> None:
        super().__init__()
        self.condition = condition

    def apply(self):
        # get child results
        assert len(self.children) <= 1, f"Select operator should have at most child, Found {len(self.children)} children"
        if len(self.children) == 0:
            self.results = self.ds.find_by_key(self.condition)
            return self.results
        child_results = self.children[0].results
        # filter the results from child_results
        self.results = self.ds.find_by_key(self.condition)
        self.results = [r for r in self.results if r in child_results]
        return self.results
    
    def __repr__(self):
        return f"select<{self.ds}>({self.condition})"

@parent_op
class Union(Operator):
    def __init__(self, operator_a) -> None:
        super().__init__()
        self.operator_a = operator_a
        self.operator_a.root.display(0)
        self.children = [self.operator_a.root]
    
    def apply(self):
        final_results = []
        self.results = [x.results for x in self.children]
        for result in self.results:
            final_results.extend(result)
        self.results = list(set(final_results))
        return self.results

    def __repr__(self):
        return f"union<>()"

@parent_op
class Materialize(Operator):
    def __init__(self, streaming=False) -> None:
        super().__init__()
        self.streaming = streaming

    def apply(self):
        assert len(self.children) == 1, f"Materialize operator should have exactly one child, Found {len(self.children)} children"
        self.results = self.children[0].results
        logger.info(f"Going to materialize {len(self.results)} results...")
        if self.streaming:
            self.results = yield from self.ds.stream_values(self.results)
        else:
            self.results = self.ds.read_values(self.results)
        return True
    
    def __repr__(self):
        return f"materialize<>()"
