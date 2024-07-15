from typing import TYPE_CHECKING

from loguru import logger
import portion as P
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

    def execute(self, mdc: "MixteraDataCollection") -> None:
        del mdc
        assert len(self.children) == 2, f"Intersection operator must have 2 children, got {len(self.children)}"

        if self.children[0].results and self.children[1].results:
            self.results = {}
            for feature, datasets in self.children[0].results.items():
                if feature in self.children[1].results:
                    self.results[feature] = {}
                    for dataset_id, file_entries in datasets.items():
                        if dataset_id in self.children[1].results[feature]:
                            self.results[feature][dataset_id] = {}
                            for file_id, data in file_entries.items():
                                if file_id in self.children[1].results[feature][dataset_id]:
                                    data1 = data
                                    data2 = self.children[1].results[feature][dataset_id][file_id]

                                    self.results[feature][dataset_id][file_id] = Intersection._intersect_index_data(data1, data2)
    
    @staticmethod
    def _intersect_index_data(data1: list[tuple[int, int] | list[int]], data2: list[tuple[int, int] | list[int]]) -> list[tuple[int, int] | list[int]]:
        """Intersect two sets of index data, handling both ranges and row identifiers."""
        if len(data1) == 0 or len(data2) == 0:
            return []
        if isinstance(data1[0], tuple):
            return Intersection._intersect_ranges(data1, data2)
        if isinstance(data1[0], list):
            return [x for x in data1 if x in data2]
        else:
            raise ValueError("Invalid data format")
        
    def _ranges_to_intervals(ranges: list[tuple[int, int]]) -> P.Interval:
        """Convert a list of range tuples to portion intervals."""
        return P.Interval(*[P.closed(start, end) for start, end in ranges])

    def _intervals_to_ranges(intervals: P.Interval) -> list[tuple[int, int]]:
        """Convert portion intervals back to a list of range tuples."""
        return [(int(interval.lower), int(interval.upper)) for interval in intervals]

    def _intersect_ranges(range1: list[tuple[int, int]], range2: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Intersect two sets of ranges using the portion library."""
        interval1 = Intersection._ranges_to_intervals(range1)
        interval2 = Intersection._ranges_to_intervals(range2)
        intersection = interval1 & interval2
        return Intersection._intervals_to_ranges(intersection)

    def __str__(self) -> str:
        return "intersection<>()"
