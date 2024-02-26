from .operator import Operator
from itertools import chain
class Materialize(Operator):
    def __init__(self, streaming:bool=False) -> None:
        super().__init__()
        self.streaming = streaming
        
    def apply(self) -> None:
        assert len(self.children) == 1, f"Materialize operator must have 1 child, got {len(self.children)}"
        self.results = self.children[0].results
        self.results = [self.mdc.get_samples_from_ranges(res) for res in self.results]
        self.results = chain(*self.results)
        if not self.streaming:
            self.results = list(self.results)
            
    def __repr__(self) -> str:
        return f"materialize<{self.mdc}>"