from typing import TYPE_CHECKING

from ._base import Operator

if TYPE_CHECKING:
    from mixtera.core.datacollection.local import LocalDataCollection


class Materialize(Operator):
    """Materialize operator is used to materialize the results of a query.
    This operator should not be used for now. We keep it here as it might
    be useful in the future for queries that need materialization in the
    middle (e.g., deduplication, filter by text length).
    Args:
        streaming (bool): If True, the results will be streamed. Defaults to False.
    """

    def __init__(self, streaming: bool = False) -> None:
        super().__init__()
        self.streaming = streaming

    def execute(self, ldc: "LocalDataCollection") -> None:
        assert len(self.children) == 1, f"Materialize operator must have 1 child, got {len(self.children)}"
        self.results = self.children[0].results
        # (todo: xiaozhe): It is still unsure if/when we need to have materialize in the query plan.
        # Leave also the streaming logic for future.
        self.results = list(ldc.get_samples_from_ranges(res) for res in self.results)

    def __repr__(self) -> str:
        return "materialize<>"
