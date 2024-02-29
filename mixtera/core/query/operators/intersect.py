from mixtera.core.query.query import QueryPlan

from ._base import Operator


class Intersection(Operator):
    """Intersection operator is used to return the intersection of the results of two queries.

    Args:
        Operator (Query): a query to intersect with the current query.
    """

    def __init__(self, query_a: QueryPlan) -> None:
        super().__init__()
        self.children.append(query_a.root)

    def apply(self) -> None:
        assert len(self.children) == 2, f"Intersection operator must have 2 children, got {len(self.children)}"
        self.results = list(set(self.children[0].results) & set(self.children[1].results))

    def __repr__(self) -> str:
        return "intersection<>()"
