from loguru import logger
from mixtera.core.query.query import Query

from ._base import Operator


class Union(Operator):
    """Union operator is used to combine the results of two queries.
    Union operator has bag semantics, meaning that it will not remove duplicates.

    Args:
        Operator (Query): a query to combine with the current query.
    """

    def __init__(self, query_a: Query) -> None:
        super().__init__()
        self.children.append(query_a.root)

    def execute(self) -> None:
        assert len(self.children) == 2, f"Union operator must have 2 children, got {len(self.children)}"
        logger.warning(
            "Union operator only supports bag semantics for now, meaning that it will not remove duplicates."
        )
        if self.children[0].results and self.children[1].results:
            self.children[0].results.merge(self.children[1].results)
            self.results = self.children[0].results

    def __str__(self) -> str:
        return "union<>()"
