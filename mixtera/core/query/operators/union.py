from typing import TYPE_CHECKING

from mixtera.utils import flatten

from ._base import Operator

if TYPE_CHECKING:
    from mixtera.core.datacollection.local import LocalDataCollection
    from mixtera.core.query.query import Query


class Union(Operator):
    """Union operator is used to combine the results of two queries.
    Union operator has bag semantics, meaning that it will not remove duplicates.

    Args:
        Operator (Query): a query to combine with the current query.
    """

    def __init__(self, query_a: "Query") -> None:
        super().__init__()
        self.children.append(query_a.root)

    def execute(self, ldc: "LocalDataCollection") -> None:
        del ldc
        assert len(self.children) == 2, f"Union operator must have 2 children, got {len(self.children)}"
        self.results = flatten([x.results for x in self.children])

    def __repr__(self) -> str:
        return "union<>()"
