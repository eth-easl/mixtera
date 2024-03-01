from ._base import Operator


class Materialize(Operator):
    """Materialize operator is used to materialize the results of a query.

    Args:
        streaming (bool): If True, the results will be streamed. Defaults to False.
    """

    def __init__(self, streaming: bool = False) -> None:
        super().__init__()
        self.streaming = streaming

    def apply(self) -> None:
        assert len(self.children) == 1, f"Materialize operator must have 1 child, got {len(self.children)}"
        assert self.mdc is not None, "Materialize operator must have a MixteraDataCollection"
        self.results = self.children[0].results
        # (todo: xiaozhe): It is still unsure if/when we need to have materialize in the query plan.
        # Leave also the streaming logic for future.
        self.results = list(self.mdc.get_samples_from_ranges(res) for res in self.results)

    def __repr__(self) -> str:
        return f"materialize<{self.mdc}>"
