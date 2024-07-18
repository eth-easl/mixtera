from typing import TYPE_CHECKING, cast

from loguru import logger
from mixtera.utils import intervals_to_ranges, ranges_to_intervals

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

    def execute(self, mdc: "MixteraDataCollection") -> None:
        del mdc
        assert len(self.children) == 2, f"Union operator must have 2 children, got {len(self.children)}"
        logger.warning(
            "Union operator only supports bag semantics for now, meaning that it will not remove duplicates."
        )
        if self.children[0].results and self.children[1].results:
            print(type(self.children[0].results))
            self.results = {
                property_key: {
                    feature: {
                        dataset_id: {
                            file_id: self._union_index_data(
                                data, self.children[1].results[property_key][feature][dataset_id].get(file_id, [])
                            )
                            for file_id, data in file_entries.items()
                        }
                        for dataset_id, file_entries in datasets.items()
                    }
                    for feature, datasets in features.items()
                }
                for property_key, features in self.children[0].results.items()
            }

    def _union_index_data(self, data1, data2):
        if len(data1) == 0:
            return data2
        if len(data2) == 0:
            return data1
        if all(isinstance(data_point, tuple) for data_point in data1) and all(
            isinstance(data_point, tuple) for data_point in data2
        ):
            data1_casted = cast(list[tuple[int, int]], data1)
            data2_casted = cast(list[tuple[int, int]], data2)
            return self._union_ranges(data1_casted, data2_casted)
        if all(isinstance(data_point, int) for data_point in data1) and all(
            isinstance(data_point, int) for data_point in data2
        ):
            union = set(data1) | set(data2)
            return cast(list[int], list(union))
        raise ValueError("Invalid data format")

    def _union_ranges(self, range1, range2):
        interval1 = ranges_to_intervals(range1)
        interval2 = ranges_to_intervals(range2)
        union = interval1 | interval2
        return intervals_to_ranges(union)

    def __str__(self) -> str:
        return "union<>()"
