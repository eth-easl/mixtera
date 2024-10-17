from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from mixtera.core.query.mixture import MixtureKey

IndexRowRangeType = list[tuple[int, int]]

# Chunker index: mixture_key -> dataset_id -> file_id -> list of ranges
ChunkerIndexDatasetEntries = dict[int, dict[int | str, IndexRowRangeType]]

#  We need to use the typing.Dict type here to avoid circular imports
ChunkerIndex = Dict["MixtureKey", ChunkerIndexDatasetEntries]
