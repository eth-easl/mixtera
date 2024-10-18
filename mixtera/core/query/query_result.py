import multiprocessing as mp
import random
from collections import defaultdict
from typing import Any, Callable, Generator, Type

import dill
import polars as pl
from loguru import logger
from mixtera.core.datacollection import MixteraDataCollection
from mixtera.core.datacollection.datasets import Dataset
from mixtera.core.datacollection.index import ChunkerIndex, ChunkerIndexDatasetEntries
from mixtera.core.datacollection.index.index_collection import create_chunker_index
from mixtera.core.query.mixture import Mixture, MixtureKey
from mixtera.core.query.result_chunk import ResultChunk
from mixtera.utils.utils import defaultdict_to_dict, merge_sorted_lists, seed_everything_from_list
from tqdm import tqdm


class QueryResult:
    """QueryResult is a class that represents the results of a query.
    The QueryResult object is iterable and yields the results in chunks of size `chunk_size`.

    The QueryResult object also has three meta properties: `dataset_type`,
    `file_path` and `parsing_func`, each of which is a dictionary that maps
    dataset/file ids to their respective types, paths and parsing functions.
    """

    def __init__(self, mdc: MixteraDataCollection, results: pl.DataFrame, mixture: Mixture) -> None:
        """
        Args:
            mdc (MixteraDataCollection): The MixteraDataCollection object.
            results (pl.DataFrame): The results of the query.
            mixture: A mixture object defining the mixture to be reflected in the chunks.
        """
        # Prepare chunker index for iterable chunking
        self._mixture = mixture
        logger.debug("Instantiating QueryResult..")
        logger.debug("Creating chunker index.")
        self._chunker_index: ChunkerIndex = QueryResult._create_chunker_index(results)
        logger.debug("Chunker index created, informing mixture and parsing metadata.")
        self._mixture.inform(self._chunker_index)

        # Set up the auxiliary data structures
        self._meta = self._parse_meta(mdc, results)

        # A process holding a QueryResult might fork (e.g., for dataloaders).
        # Hence, we need to store the locks etc in shared memory.

        # Cross-process iterator state
        self._lock = mp.Lock()
        self._index = mp.Value("i", 0)

        #  The generator will be created lazily when calling __next__
        self._generator: Generator[ResultChunk, tuple[Mixture, int], None] | None = None
        self._num_returns_gen = 0
        logger.debug("QueryResult instantiated.")

    def _parse_meta(self, mdc: MixteraDataCollection, results: pl.DataFrame) -> dict:
        dataset_ids = set(results["dataset_id"].unique())
        file_ids = set(results["file_id"].unique())

        total_length = len(results)

        return {
            "dataset_type": {did: mdc._get_dataset_type_by_id(did) for did in dataset_ids},
            "parsing_func": {did: mdc._get_dataset_func_by_id(did) for did in dataset_ids},
            "file_path": {fid: mdc._get_file_path_by_id(fid) for fid in file_ids},
            "total_length": total_length,
        }

    @staticmethod
    def _create_chunker_index(df: pl.DataFrame) -> ChunkerIndex:
        logger.info("Converting to chunker index structure...")
        chunker_index = create_chunker_index()

        # Build column index mapping to access columns by indices
        col_idx = {name: idx for idx, name in enumerate(df.columns)}

        # Identify the property columns
        exclude_keys = ["dataset_id", "file_id", "group_id", "interval_start", "interval_end"]
        property_columns = [col for col in df.columns if col not in exclude_keys]

        # Initialize variables for caching
        prev_properties = None
        current_mixture_key = None

        for row in tqdm(df.iter_rows(named=False), desc="Building chunker index.", total=len(df)):
            # Access values by index
            dataset_id = row[col_idx["dataset_id"]]
            file_id = row[col_idx["file_id"]]
            interval_start = row[col_idx["interval_start"]]
            interval_end = row[col_idx["interval_end"]]
            interval = [interval_start, interval_end]

            # Build properties dictionary per row as before
            properties = {
                k: row[col_idx[k]] if isinstance(row[col_idx[k]], list) else [row[col_idx[k]]]
                for k in property_columns
                if row[col_idx[k]] is not None and not (isinstance(row[col_idx[k]], list) and len(row[col_idx[k]]) == 0)
            }

            # Check if properties have changed
            if properties != prev_properties:
                current_mixture_key = MixtureKey(properties)
                prev_properties = properties

            # Append interval to the chunker index
            chunker_index[current_mixture_key][dataset_id][file_id].append(interval)

        logger.info("Chunker index creation completed")
        return chunker_index

    @staticmethod
    def _create_chunker_index_old(df: pl.DataFrame) -> ChunkerIndex:
        logger.info("Converting to chunker index structure...")
        chunker_index = create_chunker_index()

        for row in tqdm(df.iter_rows(named=True), desc="Building chunker index.", total=len(df)):
            dataset_id = row["dataset_id"]
            file_id = row["file_id"]
            properties = {
                k: v if isinstance(v, list) else [v]
                for k, v in row.items()
                if k not in ["dataset_id", "file_id", "group_id", "interval_start", "interval_end"]
                and v is not None  # Exclude properties from mixture key that do not have an assigned value
                and not (isinstance(v, list) and len(v) == 0)  # Sanity check to avoid empty lists as well
            }
            mixture_key = MixtureKey(properties)
            interval = [row["interval_start"], row["interval_end"]]
            chunker_index[mixture_key][dataset_id][file_id].append(interval)

        logger.info("Chunker index creation completed")

        return chunker_index

    @staticmethod
    def _generate_per_mixture_component_chunks(
        chunker_index: ChunkerIndex, component_key: MixtureKey
    ) -> Generator[ChunkerIndexDatasetEntries, int, None]:
        """
        This method computes the partial chunks for each component of a mixture. A component here is considered one
        of the chunk's fundamental property value combinations (e.g. 25% of a chunk is Medicine in English). The method
        identifies the target intervals using the passed component_key parameter (for the aforementioned example,
        this would be language:english;topic:medicine). The cardinality of a partial chunk is given by the
        cardinality of the chunk multiplied with the fraction of this property combination.

        This method is a coroutine that accepts an integer indicating the size of this component in a chunk as input.

        Args:
            chunker_index: The chunking index
            component_key: chunking index key

        Returns:
            Yields component chunks. The list has the following format:
            [
                {
                    dataset_0_id: {
                        file_0_id: [
                            [low_bound_0, high_bound_0],
                            ...
                        },
                        ...
                    },
                    ...
                },
                ...
            ]

            Each chunk has the same property combination. In the given example, all dictionaries in the list contain
            ranges that identify component_cardinality rows with the property combination specified by component_key.
        """
        target_ranges = chunker_index[component_key]

        component_cardinality = yield
        current_cardinality = 0

        # dataset_id -> file_id -> list[intervals]
        current_partition: dict[Any, dict[Any, list[tuple[int, int]]]] = defaultdict(lambda: defaultdict(list))

        for dataset_id, document_entries in target_ranges.items():
            for file_id, ranges in document_entries.items():
                for base_range in ranges:
                    current_range = (base_range[0], base_range[1])
                    continue_processing = current_range[1] > current_range[0]
                    while continue_processing:
                        range_cardinality = current_range[1] - current_range[0]
                        if current_cardinality + range_cardinality < component_cardinality:
                            # This is the case when the remaining part of the range is smaller than the
                            # current_partition. We have now completed processing the original range, and can move on
                            # to the next which is given by the innermost for loop (i.e. the one looping over 'ranges').
                            current_partition[dataset_id][file_id].append(current_range)
                            current_cardinality += range_cardinality
                            continue_processing = False
                        else:
                            # This is the case where the current range is greater than the size of a chunk. We take as
                            # much as needed from the current range to add to create the chunk (which is now fully
                            # occupied by this range), create a new chunk, and split the current range such that we
                            # do not consider the range added to the previous chunk.
                            diff = current_cardinality + range_cardinality - component_cardinality
                            current_partition[dataset_id][file_id].append((current_range[0], current_range[1] - diff))
                            component_cardinality = yield defaultdict_to_dict(current_partition)

                            # Prepare the rest of the range and new component
                            current_range = (current_range[1] - diff, current_range[1])
                            current_partition = defaultdict(lambda: defaultdict(list))
                            current_cardinality = 0

                            # Stop if range has been exhausted perfectly
                            continue_processing = current_range[1] > current_range[0]

        if current_cardinality > 0:
            # Normally we would want to record the component cardinality here as well, but since this is the last
            # generated chunk, it does not make sense to capture it as there is no other data left
            yield defaultdict_to_dict(current_partition)

    def update_mixture(self, mixture: Mixture) -> None:
        """
        Updates the mixture to be used.
        There are two use cases:
         1) Update mixture for future chunks, i.e., dynamic mixing
         2) Be able to re-use QueryResult objects that have been cached
            for different mixtures

        Args:
            mixture: the new mixture object
        """
        with self._lock:
            self._mixture = mixture
            self._mixture.inform(self._chunker_index)

    def _chunk_generator(self) -> Generator[ResultChunk, tuple[Mixture, int], None]:
        """
        Implements the chunking logic. This method yields chunks relative to  a mixture object.

        This method is a coroutine that accepts a mixture object that dictates the size of each chunk and potentially
        the mixture itself. The coroutine also accepts a target index specifying which chunk should be yielded next.
        This latter parameter is useful when chunking in a multiprocessed environment and at most once visitation
        guarantees are required.
        """
        current_chunk_index = 0
        chunker_index_keys = list(self._chunker_index.keys())
        chunker_index_keys_idx = 0
        empty_key_idx: set[int] = set()
        seed_everything_from_list(chunker_index_keys)
        random.shuffle(chunker_index_keys)

        # Initialize component iterators
        component_iterators = {
            key: self._generate_per_mixture_component_chunks(self._chunker_index, key) for key in chunker_index_keys
        }
        for iterator in component_iterators.values():
            try:
                next(iterator)
            except StopIteration:
                return

        previous_mixture = None
        base_mixture, target_chunk_index = yield
        while True:
            mixture = base_mixture.mixture_in_rows()
            if mixture:
                if previous_mixture != mixture:
                    logger.debug(f"Obtained new mixture: {mixture}")
                    previous_mixture = mixture

                chunk: ChunkerIndex = create_chunker_index()
                remaining_sizes: dict[MixtureKey, int] = {  # pylint: disable=unnecessary-comprehension
                    key: size for key, size in mixture.items()
                }

                for mixture_key in remaining_sizes.keys():
                    logger.debug(f"Handling key {mixture_key}, remaining sizes: {remaining_sizes}")

                    while remaining_sizes[mixture_key] > 0:
                        progress_made = False
                        for component_key, iterator in sorted(component_iterators.items(), key=lambda x: x[0]):
                            if mixture_key == component_key:
                                try:
                                    component_chunk: ChunkerIndexDatasetEntries = iterator.send(
                                        remaining_sizes[mixture_key]
                                    )

                                    # Update remaining size
                                    chunk_size = sum(
                                        sum(end - start for start, end in ranges)
                                        for files in component_chunk.values()
                                        for ranges in files.values()
                                    )

                                    assert (
                                        chunk_size <= remaining_sizes[mixture_key]
                                    ), f"We took too much data ({chunk_size}) for {mixture_key}: {remaining_sizes}"
                                    remaining_sizes[mixture_key] = remaining_sizes[mixture_key] - chunk_size

                                    logger.debug(f"Received chunk size: {chunk_size} for {mixture_key}")

                                    # Merge the component chunk into the main chunk
                                    for dataset_id, files in component_chunk.items():
                                        for file_id, ranges in files.items():
                                            chunk[mixture_key][dataset_id][file_id] = (
                                                ranges
                                                if file_id not in chunk[mixture_key][dataset_id]
                                                else merge_sorted_lists(chunk[mixture_key][dataset_id][file_id], ranges)
                                            )
                                            # If we extended the ranges of that file, we need to sort them since, e.g.,
                                            # the JSONL file wrapper expects them in sorted order
                                            # Since we now ranges are sorted and the existing ranges
                                            # are sorted as well, we use a merge operation.

                                    progress_made = True

                                    if remaining_sizes[mixture_key] == 0:
                                        logger.debug(f"Finished data for {mixture_key}: {remaining_sizes}")
                                        break  # Do not consider another iterator if we're done

                                except StopIteration:
                                    logger.debug("Continuing on StopIteration")
                                    continue

                        # No matching components found or all are exhausted, unable to complete the chunk
                        if not progress_made:
                            logger.debug("Did not make progress, unable to complete chunk.")
                            return

                # Check if we have enough data for all mixture keys
                # TODO(#111): Make it possible to support best effort here.
                # Right now, if we cannot fulfill the mixture for that chunk, we stop.
                if all(size == 0 for size in remaining_sizes.values()):
                    if current_chunk_index == target_chunk_index:
                        logger.debug("Yielding a chunk.")
                        base_mixture, target_chunk_index = yield ResultChunk(
                            defaultdict_to_dict(chunk),
                            self.dataset_type,
                            self.file_path,
                            self.parsing_func,
                            base_mixture.chunk_size,
                            mixture,
                        )
                    else:
                        logger.debug(
                            f"current_chunk_index = {current_chunk_index} != target_chunk_index = {target_chunk_index}"
                        )
                # Not enough data to complete the chunk, end generation
                else:
                    logger.debug("Not enough data, ending chunk generation")
                    return
            else:
                chunk = None
                while len(empty_key_idx) < len(chunker_index_keys) and chunk is None:
                    chunker_index_keys_idx = (chunker_index_keys_idx + 1) % len(chunker_index_keys)
                    if chunker_index_keys_idx in empty_key_idx:
                        # Note that this can be removed but needs some adjustments in the tests (only impacts ordering)
                        chunker_index_keys_idx = (chunker_index_keys_idx + 1) % len(chunker_index_keys)
                        continue

                    key = chunker_index_keys[chunker_index_keys_idx]
                    try:
                        chunk = component_iterators[key].send(base_mixture.chunk_size)
                    except StopIteration:
                        # The current key is exhausted; will need to produce chunks from the next available key
                        empty_key_idx.add(chunker_index_keys_idx)

                if chunk is None:
                    # There were no more available chunks; we mark the end of this query result with StopIteration
                    return

                # Chunk has been successfully generated
                if current_chunk_index == target_chunk_index:
                    chunk = {chunker_index_keys[chunker_index_keys_idx]: chunk}
                    base_mixture, target_chunk_index = yield ResultChunk(
                        chunk, self.dataset_type, self.file_path, self.parsing_func, base_mixture.chunk_size, None
                    )
            current_chunk_index += 1

    @property
    def chunk_size(self) -> int:
        return self._mixture.chunk_size

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

    def __next__(self) -> ResultChunk:
        """Iterate over the results of the query."""
        with self._index.get_lock():
            chunk_target_index = self._index.get_obj().value
            self._index.get_obj().value += 1

        with self._lock:
            #  The generator is created lazily since the QueryResult object might be pickled
            # (and the generator was deleted from the state)
            if self._generator is None:
                self._generator = self._chunk_generator()
                next(self._generator)

                assert (
                    self._num_returns_gen == 0
                ), f"Generator was not reset properly. Got {self._num_returns_gen} returns."

            self._num_returns_gen += 1
            return self._generator.send((self._mixture, chunk_target_index))

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()

        # Remove the generator since it is not pickable (and will be recreated on __next__)
        del state["_generator"]

        #  The following attributes are pickled using dill since they are not pickable by
        # the default pickler (used by torch)
        dill_pickled_attributes = {}
        for attrib in ["_meta", "_chunker_index"]:
            attrib_pickled = dill.dumps(state[attrib])
            del state[attrib]
            dill_pickled_attributes[attrib] = attrib_pickled

        # Return a dictionary with the pickled attribute and other picklable attributes
        return {"other": state, "dilled": dill_pickled_attributes}

    def __setstate__(self, state: dict) -> None:
        self.__dict__ = state["other"]
        self._generator = None
        for attrib, attrib_pickled in state["dilled"].items():
            setattr(self, attrib, dill.loads(attrib_pickled))
