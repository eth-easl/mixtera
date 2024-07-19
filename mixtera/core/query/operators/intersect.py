from typing import TYPE_CHECKING, cast

from mixtera.core.query.query_plan import QueryPlan
from mixtera.utils import intervals_to_ranges, ranges_to_intervals

from ._base import Operator

if TYPE_CHECKING:
    from mixtera.core.client.local import MixteraDataCollection


class Intersection(Operator):
    """Intersection operator is used to return the intersection of the results of two queries.

    Args:
        Operator (Query): a query to intersect with the current query.
    """

    def __init__(self, query_a: QueryPlan) -> None:
        super().__init__()
        self.children.append(query_a.root)

    def execute(self, mdc: "MixteraDataCollection") -> None:
        del mdc
        assert len(self.children) == 2, f"Intersection operator must have 2 children, got {len(self.children)}"
        if self.children[0].results and self.children[1].results:
            self.children[0].results.intersect(self.children[1].results)
            self.results = self.children[0].results

    def __str__(self) -> str:
        return "intersection<>()"
