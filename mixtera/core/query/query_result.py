import multiprocessing as mp
from typing import Callable, Optional, Type

import dill
from loguru import logger
from mixtera.core.datacollection import MixteraDataCollection
from mixtera.core.datacollection.datasets import Dataset
from mixtera.core.datacollection.index import IndexType, InvertedIndex
from mixtera.core.datacollection.index.index_collection import IndexFactory, \
    IndexTypes, raw_index_dict_instantiator, create_inverted_index
from mixtera.utils import defaultdict_to_dict

import portion


class QueryResult:
    """QueryResult is a class that represents the results of a query.
    The QueryResult object is iterable and yields the results in chunks of size `chunk_size`.

    The QueryResult object also has three meta properties: `dataset_type`,
    `file_path` and `parsing_func`, each of which is a dictionary that maps
    dataset/file ids to their respective types, paths and parsing functions.
    """

    def __init__(self, mdc: MixteraDataCollection, results: IndexType, chunk_size: int = 1) -> None:
        """
        Args:
            mdc (LocalDataCollection): The LocalDataCollection object.
            results (IndexType): The results of the query.
            chunk_size (int): The chunk size of the results.
        """
        self.chunk_size = chunk_size
        self.results = results
        self._meta = self._parse_meta(mdc)
        self._chunks: list[IndexType] = []
        self.chunking()
        # A process holding a QueryResult might fork (e.g., for dataloaders).
        # Hence, we need to store the locks etc in shared memory.
        self._manager = mp.Manager()
        self._lock = self._manager.Lock()
        self._index = self._manager.Value("i", 0)
        logger.debug(f"Instantiated QueryResult with {len(self._chunks)} chunks.")

    def _parse_meta(self, mdc: MixteraDataCollection) -> dict:
        dataset_ids = set()
        file_ids = set()

        total_length = 0
        for prop_name in self.results.get_all_features():
            for prop_val in self.results.get_dict_index_by_feature(prop_name).values():
                dataset_ids.update(prop_val.keys())
                for files in prop_val.values():
                    file_ids.update(files)
                    for file_ranges in files.values():
                        total_length += sum(end - start for start, end in file_ranges)

        return {
            "dataset_type": {did: mdc._get_dataset_type_by_id(did) for did in dataset_ids},
            "parsing_func": {did: mdc._get_dataset_func_by_id(did) for did in dataset_ids},
            "file_path": {fid: mdc._get_file_path_by_id(fid) for fid in file_ids},
            "total_length": total_length,
        }

    def _invert_result(self) -> InvertedIndex:
        """
        Returns a dictionary that points from files to lists of ranges annotated
        with properties.

        Returns:
            A dictionary of the form:
                {
                    "dataset_id": {
                        "file_id": {[
                            (lo_bound, hi_bound, {feature_1_name: feature_1_value, ...}),
                        ...
                        ]},
                        ...
                    },
                    ...
                }

        """

        # Invert index
        inverted_dictionary: InvertedIndex = create_inverted_index()
        for property_name, property_values in self.results.items():
            for property_value, datasets in property_values.items():
                for dataset_id, files in datasets.items():
                    for file_id, ranges in files.items():
                        for range in ranges:
                            inverted_dictionary[dataset_id][file_id].append(
                                (portion.closedopen(range[0], range[1]), {property_name: property_value}))



        return





    def chunking(self) -> None:
        # structure of self.chunks: [index_type, index_type, ...]
        # each index_type is an index, but contains exactly
        # chunk_size items, except the last one.
        chunks: list[dict] = []
        # here we chunk the self.results from a large index_type
        # into smaller index_types.
        current_chunk: IndexType = raw_index_dict_instantiator()
        current_chunk_length = 0
        # this is likely not a good impl, optimize it later.
        # now just ensure correct chunk_size and each chunk is an index.
        # pylint: disable-next=too-many-nested-blocks
        for property_name in self.results.get_all_features():
            for property_value, property_dict in self.results.get_dict_index_by_feature(property_name).items():
                for did, files in property_dict.items():
                    for fid, file_ranges in files.items():
                        for start, end in file_ranges:
                            # TODO(#35): we may want to optimize this later together with mixture.
                            # for now this is a simple implementaion (for testing)
                            # case 1: the entire range fits into the current chunk, add it to the current chunk
                            if end - start < self.chunk_size - current_chunk_length:
                                current_chunk[property_name][property_value][did][fid].append((start, end))
                                current_chunk_length += end - start
                                # chunk is not full yet
                            # case 2: the entire range does not fit into the current chunk
                            else:
                                # # step 1: add the first part of the range to the current chunk, to fill it up
                                current_chunk[property_name][property_value][did][fid].append(
                                    (start, start + self.chunk_size - current_chunk_length)
                                )
                                # this time the chunk is full
                                chunks.append(current_chunk)
                                # step 2: calculate how many chunks are needed for the rest of the range
                                remaining_length = end - (start + self.chunk_size - current_chunk_length)
                                remaining_chunks = remaining_length // self.chunk_size
                                if remaining_chunks > 0:
                                    # step 3: add the remaining chunks
                                    for i in range(remaining_chunks):
                                        new_chunk = raw_index_dict_instantiator()
                                        new_chunk[property_name] = {
                                            property_value: {
                                                did: {
                                                    fid: [
                                                        (
                                                            start
                                                            + self.chunk_size
                                                            - current_chunk_length
                                                            + i * self.chunk_size,
                                                            start
                                                            + self.chunk_size
                                                            - current_chunk_length
                                                            + (i + 1) * self.chunk_size,
                                                        )
                                                    ]
                                                }
                                            }
                                        }
                                        chunks.append(new_chunk)
                                remaining_length = remaining_length % self.chunk_size
                                current_chunk = raw_index_dict_instantiator()
                                current_chunk_length = 0
                                start_new = start + remaining_chunks * self.chunk_size
                                if remaining_length > 0:
                                    current_chunk[property_name] = {
                                        property_value: {did: {fid: [(start_new, start_new + remaining_length)]}}
                                    }
                                    current_chunk_length += remaining_length
                        # reset current chunk after this file - we anyway need a new index for the next file
        if current_chunk_length > 0:
            chunks.append(current_chunk)
        for chunk in chunks:
            chunk_index = IndexFactory.create_index(IndexTypes.IN_MEMORY_DICT_RANGE)
            chunk_index._index = defaultdict_to_dict(chunk)
            self._chunks.append(chunk_index)

    @property
    def dataset_type(self) -> dict[int, Type[Dataset]]:
        return self._meta["dataset_type"]

    @property
    def file_path(self) -> dict[int, str]:
        return self._meta["file_path"]

    @property
    def parsing_func(self) -> dict[int, Callable[[str], str]]:
        return self._meta["parsing_func"]

    def __iter__(self) -> "QueryResult":
        return self

    def __next__(self) -> IndexType:
        """Iterate over the results of the query with a chunk size thread-safe.
        This method is very dummy right now without ensuring the correct mixture.
        """
        local_index: Optional[int] = None
        with self._lock:
            if self._index.value < len(self._chunks):
                local_index = self._index.value
                self._index.value += 1
        # We exit the scope of the lock as early as possible.
        # For now the actual slicing does not need to be locked.

        if local_index is not None:
            return self._chunks[local_index]

        raise StopIteration

    def __getstate__(self) -> dict:
        # _meta is not pickable using the default pickler (used by torch),
        # so we have to rely on dill here
        state = self.__dict__.copy()
        meta_pickled = dill.dumps(state["_meta"])
        del state["_meta"]
        # Also, we cannot pickle the manager, but also don't need it in the subclasses.
        if "_manager" in state:
            del state["_manager"]

        # Return a dictionary with the pickled attribute and other picklable attributes
        return {"other": state, "meta_pickled": meta_pickled}

    def __setstate__(self, state: dict) -> None:
        self.__dict__ = state["other"]
        self._meta = dill.loads(state["meta_pickled"])
