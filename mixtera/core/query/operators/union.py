from typing import TYPE_CHECKING

from loguru import logger

from ._base import Operator

if TYPE_CHECKING:
    from mixtera.core.client.local import MixteraDataCollection
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

    def execute(self, ldc: "MixteraDataCollection") -> None:
        del ldc
        assert len(self.children) == 2, f"Union operator must have 2 children, got {len(self.children)}"
        logger.warning(
            "Union operator only supports bag semantics for now, meaning that it will not remove duplicates."
        )
        if self.children[0].results and self.children[1].results:
            self.children[0].results.merge(self.children[1].results)
            self.results = self.children[0].results

    def __str__(self) -> str:
        return "union<>()"
