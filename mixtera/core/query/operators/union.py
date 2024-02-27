from mixtera.core.query.query import Query

from ._base import Operator


class Union(Operator):
    """Union operator is used to combine the results of two queries.
    Union operator has bag semantics, meaning that it will not remove duplicates.

    Args:
        Operator (_type_): a query to combine with the current query.
    """

    def __init__(self, query_a: Query) -> None:
        super().__init__()
        self.children.append(query_a.root)

    def apply(self) -> None:
        assert len(self.children) == 2, f"Union operator must have 2 children, got {len(self.children)}"
        final_results = []
        self.results = [x.results for x in self.children]
        for result in self.results:
            final_results.extend(result)
        self.results = final_results
        print(self.results)
        
    def __repr__(self) -> str:
        return "union<>()"
