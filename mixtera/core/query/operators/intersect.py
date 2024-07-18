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
            self.results = {
                property_key: {
                    feature: {
                        dataset_id: {
                            file_id: self._intersect_index_data(
                                data, self.children[1].results[property_key][feature][dataset_id][file_id]
                            )
                            for file_id, data in file_entries.items()
                            if file_id in self.children[1].results[property_key][feature][dataset_id]
                        }
                        for dataset_id, file_entries in datasets.items()
                        if dataset_id in self.children[1].results[property_key][feature]
                    }
                    for feature, datasets in features.items()
                    if feature in self.children[1].results[property_key]
                }
                for property_key, features in self.children[0].results.items()
                if property_key in self.children[1].results
            }

    def _intersect_index_data(
        self, data1: list[tuple[int, int]] | list[int], data2: list[tuple[int, int]] | list[int]
    ) -> list[tuple[int, int]] | list[int]:
        """Intersect two sets of index data, handling both ranges and row identifiers."""
        if len(data1) == 0 or len(data2) == 0:
            return []
        if all(isinstance(data_point, tuple) for data_point in data1) and all(
            isinstance(data_point, tuple) for data_point in data2
        ):
            data1_casted = cast(list[tuple[int, int]], data1)
            data2_casted = cast(list[tuple[int, int]], data2)
            return self._intersect_ranges(data1_casted, data2_casted)
        if all(isinstance(data_point, int) for data_point in data1) and all(
            isinstance(data_point, int) for data_point in data2
        ):
            intersection = set(data1) & set(data2)
            return cast(list[int], list(intersection))
        raise ValueError("Invalid data format")

    def _intersect_ranges(self, range1: list[tuple[int, int]], range2: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Intersect two sets of ranges using the portion library."""
        interval1 = ranges_to_intervals(range1)
        interval2 = ranges_to_intervals(range2)
        intersection = interval1 & interval2
        return intervals_to_ranges(intersection)

    def __str__(self) -> str:
        return "intersection<>()"
