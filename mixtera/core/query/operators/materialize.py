from ._base import Operator


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

    def execute(self) -> None:
        assert len(self.children) == 1, f"Materialize operator must have 1 child, got {len(self.children)}"
        assert self.mdc is not None, "Materialize operator must have a MixteraDataCollection"
        self.results = self.children[0].results
        # TODO(#44): It is still unsure if/when we need to have materialize in the query plan.
        # Leave also the streaming logic for future.
        self.results = list(self.mdc.get_samples_from_ranges(res) for res in self.results)

    def __str__(self) -> str:
        return f"materialize<{self.mdc}>"
