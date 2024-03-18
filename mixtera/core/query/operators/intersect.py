from typing import TYPE_CHECKING

from loguru import logger
from mixtera.core.query.query_plan import QueryPlan

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

    def execute(self, ldc: "MixteraDataCollection") -> None:
        del ldc
        assert len(self.children) == 2, f"Intersection operator must have 2 children, got {len(self.children)}"
        # TODO(#39): This is a dummy implementation, we need to implement the real intersection logic.
        # Will do it in a following PR.
        logger.warning(
            "Intersection operator is not implemented properly yet, "
            "returning the intersection of the results of the two queries."
        )
        if self.children[0].results and self.children[1].results:
            self.results = [x for x in self.children[0].results if x in self.children[1].results]

    def __str__(self) -> str:
        return "intersection<>()"
