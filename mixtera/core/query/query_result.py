import json
import multiprocessing as mp
import os
import pickle
import random
from collections import defaultdict
from copy import deepcopy
from itertools import cycle
from pathlib import Path
from typing import Any, Callable, Generator, Type

import dill
import pyarrow as pa
from loguru import logger
from mixtera.core.datacollection import MixteraDataCollection
from mixtera.core.datacollection.datasets import Dataset
from mixtera.core.datacollection.index import ChunkerIndex, ChunkerIndexDatasetEntries
from mixtera.core.datacollection.index.index_collection import create_chunker_index
from mixtera.core.query.chunker import create_chunker_index as cpp_create
from mixtera.core.query.mixture import Mixture, MixtureKey
from mixtera.core.query.mixture.dynamic_mixture import DynamicMixture
from mixtera.core.query.result_chunk import ResultChunk
from mixtera.utils.utils import (
    defaultdict_to_dict,
    deserialize_chunker_index,
    distribute_by_ratio,
    merge_sorted_lists,
    seed_everything_from_list,
    serialize_chunker_index,
)
from pyarrow import compute as pc


class QueryResult:
    """QueryResult is a class that represents the results of a query.
    The QueryResult object is iterable and yields the results in chunks of size `chunk_size`.

    The QueryResult object also has three meta properties: `dataset_type`,
    `file_path` and `parsing_func`, each of which is a dictionary that maps
    dataset/file ids to their respective types, paths and parsing functions.
    """

    def __init__(
        self,
        mdc: MixteraDataCollection,
        results: pa.Table,
        mixture: Mixture,
        query_log_dir: Path | None = None,
        stop_on_none: bool = True,
        _infinite_cycle: bool = False,
    ) -> None:
        """
        Args:
            mdc (MixteraDataCollection): The MixteraDataCollection object.
            results (pl.DataFrame): The results of the query.
            mixture: A mixture object defining the mixture to be reflected in the chunks.
            stop_on_none: Typically, the QueryResult is consumed by the ChunkDistributor,
              in which case we allow multiple tries to generate a chunk (e.g., with a new mixture).
              However some consumers (e.g., tests) need to iterate once over all data and be done with it.
        """
        # Prepare chunker index for iterable chunking
        self._mixture = mixture
        self.stop_on_none = stop_on_none
        logger.debug("Instantiating QueryResult..")
        logger.debug("Creating chunker index.")
        self._chunker_index: ChunkerIndex = QueryResult._create_chunker_index(results)
        logger.debug("Chunker index created, informing mixture and parsing metadata.")
        self._mixture.process_index(self._chunker_index)

        # Set up the auxiliary data structures
        self._meta = self._parse_meta(mdc, results)

        # Setup global key => ID map for all IDs in the chunker index
        self._key_id_map: dict[MixtureKey, int] = {}
        self._update_key_id_map()

        # A process holding a QueryResult might fork (e.g., for dataloaders).
        # Hence, we need to store the locks etc in shared memory.

        # Cross-process iterator state
        self._lock = mp.Lock()
        self._index = mp.Value("i", 0)

        # initialize caching for infinite cycling.
        # This will store every successfully generated chunk.
        self._cached_chunks: list[ResultChunk] = []
        # Once the normal generator is exhausted, if infinite cycling is enabled,
        # we create a cycle generator over the cached chunks.
        self._cycle_generator: cycle | None = None
        self._infinite_cycle = _infinite_cycle

        # The generator will be created lazily when calling __next__
        self._generator: Generator[ResultChunk, tuple[Mixture, int], None] | None = None
        self._num_returns_gen = 0
        logger.debug("QueryResult instantiated.")

        self._mixture_log: list[tuple[int, Mixture]] = []
        self._query_log_dir = query_log_dir
        if query_log_dir is not None:
            query_log_dir.mkdir(exist_ok=True)

    def _update_key_id_map(self) -> None:
        updated = False
        initial_setup = len(self._key_id_map.keys()) == 0
        current_id = 0 if initial_setup else max(self._key_id_map.values()) + 1

        keys = set(self._mixture.mixture_in_rows().keys())

        if initial_setup:
            # This allows us to have a ID available whenever we switch to a None mixture.
            keys.update(set(self._chunker_index.keys()))
            updated = True

        for key in sorted(keys):
            if key not in self._key_id_map:
                self._key_id_map[key] = current_id
                current_id += 1
                updated = True

        if updated:
            self._mixture.process_id_map(self._key_id_map)
            logger.debug(f"Updated key-id-map:\n{self._key_id_map}\n")

    def _persist_mixture_log(self) -> None:
        if self._query_log_dir is None:
            return

        with self._index.get_lock():
            curr_chunk_idx = self._index.get_obj().value

        logger.debug("Persisting mixture log to disk...")

        with open(self._query_log_dir / "mixture.log", "w", encoding="utf-8") as fp:
            json.dump(
                {
                    "log:": {chk_id: mixture.stringified_mixture() for chk_id, mixture in self._mixture_log},
                    "curr_chunk_idx": curr_chunk_idx,
                },
                fp,
                indent=4,
                sort_keys=True,
            )

        logger.debug("Mixture log persisted.")

    def _persist_chunk_idx(self, current_chunk_index: int) -> None:
        if self._query_log_dir is None:
            return

        with open(self._query_log_dir / "chunk.idx", "w", encoding="utf-8") as fp:
            fp.write(str(current_chunk_index))

    def _parse_meta(self, mdc: MixteraDataCollection, results: pa.Table) -> dict:
        dataset_ids = set(pc.unique(results["dataset_id"]).to_pylist())
        file_ids = set(pc.unique(results["file_id"]).to_pylist())

        total_length = len(results)

        return {
            "dataset_type": {did: mdc._get_dataset_type_by_id(did) for did in dataset_ids},
            "parsing_func": {did: mdc._get_dataset_func_by_id(did) for did in dataset_ids},
            "file_path": {fid: mdc._get_file_path_by_id(fid) for fid in file_ids},
            "total_length": total_length,
        }

    @staticmethod
    def _create_chunker_index(table: pa.Table) -> ChunkerIndex:
        """
        Converts a PyArrow Table containing query results into a ChunkerIndex data structure.

        The ChunkerIndex is a nested dictionary structure that organizes data intervals based on their properties,
        enabling efficient chunking and data retrieval according to specified mixture criteria.

        This method processes the input table in parallel by splitting it into batches.
        Each batch is processed to build a partial ChunkerIndex,
        and these partial indices are then merged into a single ChunkerIndex.

        Args:
            table (pa.Table): A PyArrow Table resulting from the query, containing intervals and associated properties.

        Returns:
            ChunkerIndex: A nested dictionary mapping mixture keys to dataset IDs, file IDs, and intervals.
        """
        logger.info("Converting to chunker index structure...")
        num_cores = os.cpu_count() or 1
        num_workers = max(num_cores - 4, 1)  # TODO(#124): Make this configurable.
        in_test = os.environ.get("PYTEST_CURRENT_TEST")
        return cpp_create(table, num_workers if not in_test else 1)

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

        for dataset_id, document_entries in sorted(target_ranges.items(), key=lambda x: x[0]):
            for file_id, ranges in sorted(document_entries.items(), key=lambda x: x[0]):
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
            self._mixture.process_index(self._chunker_index)
            self._update_key_id_map()

    def _chunk_generator(self) -> Generator[ResultChunk, tuple[Mixture, int], None]:
        """
        Implements the chunking logic. This method yields chunks relative to  a mixture object.

        This method is a coroutine that accepts a mixture object that dictates the size of each chunk and potentially
        the mixture itself. The coroutine also accepts a target index specifying which chunk should be yielded next.
        This latter parameter is useful when chunking in a multiprocessed environment and at most once visitation
        guarantees are required.
        """
        current_chunk_index = 0
        mixture_id = -1
        chunker_index_keys = list(self._chunker_index.keys())
        chunker_index_keys_idx = 0
        empty_key_idx: set[int] = set()
        # Here we shuffle the chunker index keys,
        # which determines the order of keys considered when two MixtureKeys are equal.
        # Hence, this depends on the hash function.
        seed_everything_from_list(chunker_index_keys)
        chunker_index_keys.sort()  # Otherwise, despite seeding, a shuffle is not reproducible.
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
        # We allow to re-query the latest chunk after not being able to generate one,
        # e.g., due to dynamic mixture changes.
        # However, after a certain limit, we raise a StopIteration to avoid deadlocks.
        no_success_counter = 0

        total_counts: dict[int] = defaultdict(int)

        while True:
            if no_success_counter > 10:
                logger.error("Hard-stopping chunk generation after 10 unsucessful tries.")
                return

            mixture = deepcopy(base_mixture.mixture_in_rows())
            is_strict = base_mixture.strict

            chunk_success = False
            if mixture:
                if previous_mixture != mixture:
                    logger.debug(f"Obtained new mixture: {mixture}")
                    mixture_id += 1
                    previous_mixture = deepcopy(mixture)

                    if len(self._mixture_log) > 0:
                        last_mixture = self._mixture_log[-1][1]
                        if isinstance(last_mixture, DynamicMixture) and hasattr(last_mixture, "_mixing_alg"):
                            logger.info("Cleaning up last mixing algorithm from mixture log.")
                            last_mixture._mixing_alg = None

                    self._mixture_log.append((current_chunk_index, deepcopy(base_mixture)))
                    self._persist_mixture_log()
                    self._update_key_id_map()

                chunk: ChunkerIndex = create_chunker_index()
                remaining_sizes: dict[MixtureKey, int] = {  # pylint: disable=unnecessary-comprehension
                    key: size for key, size in mixture.items()
                }
                original_sizes = remaining_sizes.copy()

                global_progress_made = True
                while global_progress_made and any(remaining_sizes.values()):
                    global_progress_made = False

                    # Sort to guarantee same handling for semantically same mixtures
                    for mixture_key in sorted(remaining_sizes.keys()):
                        # logger.debug(f"Handling key {mixture_key}, remaining sizes: {remaining_sizes}")

                        for component_key, iterator in sorted(component_iterators.items(), key=lambda x: x[0]):
                            # logger.debug(f"Checking component key {component_key}")
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

                                    # logger.debug(
                                    #    f"Received chunk size: {chunk_size} for {mixture_key} from {component_key}"
                                    # )

                                    # Merge the component chunk into the main chunk
                                    for dataset_id, files in component_chunk.items():
                                        for file_id, ranges in files.items():
                                            chunk[mixture_key][dataset_id][file_id] = (
                                                ranges
                                                if file_id not in chunk[mixture_key][dataset_id]
                                                else merge_sorted_lists(
                                                    chunk[mixture_key][dataset_id][file_id],
                                                    ranges,
                                                )
                                            )
                                            # If we extended the ranges of that file, we need to sort them since,
                                            # e.g., the JSONL file wrapper expects them in sorted order
                                            # Since we now ranges are sorted and the existing ranges
                                            # are sorted as well, we use a merge operation.

                                    global_progress_made = global_progress_made or chunk_size > 0

                                    if remaining_sizes[mixture_key] == 0:
                                        # logger.debug(f"Finished data for {mixture_key}: {remaining_sizes}")
                                        break  # Do not consider another iterator if we're done

                                except StopIteration:
                                    continue

                        # No matching components found or all are exhausted
                        if remaining_sizes[mixture_key] > 0:
                            logger.debug(f"No progress on key {mixture_key}.")
                            if is_strict:  # Unable to complete chunk
                                logger.debug("Did not make progress, unable to complete chunk.")
                            else:
                                # best-effort generation
                                num_missing_samples = remaining_sizes.pop(mixture_key)
                                # Remaining sizes contains all keys that have not been depleted so far,
                                # even ones with value 0.

                                assert num_missing_samples <= mixture[mixture_key], (
                                    f"missing samples = {num_missing_samples}"
                                    + f"\noriginal sizes = {original_sizes} \n mixture = {mixture}"
                                )
                                pre_best_effort_sum = sum(mixture.values())

                                # If we have not put any samples for this key into the chunk,
                                # remove it from mixture
                                if num_missing_samples == mixture[mixture_key]:
                                    mixture.pop(mixture_key)
                                else:
                                    logger.debug(
                                        f"This is the first chunk without finishing {mixture_key}. "
                                        + f"{num_missing_samples}/{mixture[mixture_key]} samples are missing."
                                    )
                                    mixture[mixture_key] -= num_missing_samples

                                if not remaining_sizes:
                                    logger.debug("Not enough data, ending chunk generation")
                                else:
                                    # Redistribute missing samples among other mixture keys that are not finished
                                    # Note that remaining_sizes includes keys with current value 0,
                                    # just not already empty ones (avoiding loops)
                                    target_keys = sorted(remaining_sizes.keys())
                                    # logger.debug(f"target keys = {target_keys}")

                                    total_mixture_size_remaining = sum(mixture[key] for key in target_keys)

                                    ratios = [mixture[key] / total_mixture_size_remaining for key in target_keys]

                                    samples_to_distribute = distribute_by_ratio(num_missing_samples, ratios)

                                    assert sum(samples_to_distribute) == num_missing_samples, (
                                        f"std = {samples_to_distribute}" + f"\nmissing = {num_missing_samples}"
                                    )

                                    for i, key in enumerate(target_keys):
                                        # logger.debug(f"Distributing {samples_to_distribute[i]} samples to {key}.")
                                        remaining_sizes[key] += samples_to_distribute[i]
                                        mixture[key] += samples_to_distribute[i]

                                    post_best_effort_sum = sum(mixture.values())

                                    assert pre_best_effort_sum == post_best_effort_sum, (
                                        "mixture sum changed: "
                                        + f"pre = {pre_best_effort_sum} post = {post_best_effort_sum}"
                                    )

                                    # Otherwise, if the first key runs out,
                                    # we will stop generating due to the break below.
                                    global_progress_made = True

                                break

                # Check if we have enough data for all mixture keys
                if remaining_sizes and all(size == 0 for size in remaining_sizes.values()):
                    chunk_success = True
                    no_success_counter = 0
                    if current_chunk_index == target_chunk_index:
                        logger.debug(f"Yielding chunk {current_chunk_index}.")
                        self._persist_chunk_idx(current_chunk_index)
                        result_chunk = ResultChunk(
                            defaultdict_to_dict(chunk),
                            self.dataset_type,
                            self.file_path,
                            self.parsing_func,
                            base_mixture.chunk_size,
                            self._key_id_map,
                            mixture_id,
                            mixture=mixture,
                            strict_mixture=is_strict,
                        )

                        chunk_counts = defaultdict(int)
                        for mixture_key, datasets in chunk.items():
                            dataset = mixture_key.properties["dataset"][0]
                            for files in datasets.values():
                                for ranges in files.values():
                                    for start, end in ranges:
                                        count = end - start
                                        chunk_counts[dataset] += count

                        for dataset, count in chunk_counts.items():
                            total_counts[dataset] += count

                        print(f"Chunk {current_chunk_index} counts: {total_counts}")
                        base_mixture, target_chunk_index = yield result_chunk

                    else:
                        logger.debug(
                            f"current_chunk_index = {current_chunk_index} != target_chunk_index = {target_chunk_index}"
                        )
                # Not enough data to complete the chunk, end generation
                else:
                    logger.debug("Not enough data, ending chunk generation")
                    no_success_counter += 1
                    # yield None
            else:
                if previous_mixture is not None or current_chunk_index == 0:
                    logger.debug("Obtained new None mixture.")
                    mixture_id += 1
                    previous_mixture = None
                    self._mixture_log.append((current_chunk_index, base_mixture))
                    self._persist_mixture_log()

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
                chunk_success = True
                if chunk is None:
                    return  # No need to yield None, if ArbitraryMixture is exhausted, we will never be able to continue

                # Chunk has been successfully generated
                if current_chunk_index == target_chunk_index:
                    chunk = {chunker_index_keys[chunker_index_keys_idx]: chunk}
                    self._persist_chunk_idx(current_chunk_index)
                    logger.debug(f"Yielding chunk {current_chunk_index}.")
                    result_chunk = ResultChunk(
                        chunk,
                        self.dataset_type,
                        self.file_path,
                        self.parsing_func,
                        base_mixture.chunk_size,
                        self._key_id_map,
                        mixture_id,
                        mixture=None,
                    )
                    chunk_counts = defaultdict(int)
                    for mixture_key, datasets in chunk.items():
                        dataset = mixture_key.properties["dataset"][0]
                        for files in datasets.values():
                            for ranges in files.values():
                                for start, end in ranges:
                                    count = end - start
                                    chunk_counts[dataset] += count

                    for dataset, count in chunk_counts.items():
                        total_counts[dataset] += count

                    logger.debug(f"Chunk {current_chunk_index} counts: {total_counts}")

                    base_mixture, target_chunk_index = yield result_chunk

            if chunk_success:
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
        with self._index.get_lock():
            chunk_target_index = self._index.get_obj().value
            self._index.get_obj().value += 1

        with self._lock:
            # Create the generator if it hasn't been created yet.
            if self._generator is None:
                self._generator = self._chunk_generator()
                next(self._generator)  # prime the generator
                self._num_returns_gen = 0

            try:
                # Try to get the next chunk.
                result = self._generator.send((self._mixture, chunk_target_index))
            except StopIteration:
                # The generator is exhausted.
                result = None

            # If a valid result is returned, cache it (if cycling is enabled) and return it.
            if result is not None:
                if self._infinite_cycle:
                    self._cached_chunks.append(result)
                return result

            # No new chunk was produced.
            # Roll back the index for this chunk.
            with self._index.get_lock():
                self._index.get_obj().value -= 1

            if not self._infinite_cycle:
                if self.stop_on_none:
                    raise StopIteration
                return None
            # If we haven't cached any chunks, we cannot cycle.
            if not self._cached_chunks:
                raise StopIteration("No chunks available to cycle through.")
            # Replace the generator with an infinite cycle over the cached chunks.
            if self._cycle_generator is None:
                self._cycle_generator = cycle(self._cached_chunks)
            return next(self._cycle_generator)

    # SERIALIZATION ##

    def __getstate__(self) -> dict:
        logger.debug("Starting to pickle a Queryresult.")
        state = self.__dict__.copy()

        # Remove the generator since it is not pickable (and will be recreated on __next__)
        del state["_generator"]

        # The following attributes are pickled using dill since they are not pickable by
        # the default pickler (used by torch)
        dill_pickled_attributes = {}
        for attrib in ["_meta", "_chunker_index", "_mixture_log"]:
            attrib_pickled = dill.dumps(state[attrib])
            del state[attrib]
            dill_pickled_attributes[attrib] = attrib_pickled

        if "_index" in state:
            logger.warning(
                "You're pickling a QueryResult without handling _index."
                + "We're deleting the _index attribute, but this might lead to unexpected behavior!"
            )
            del state["_index"]

        if "_lock" in state:
            logger.warning(
                "You're pickling a QueryResult without handling _lock."
                + "We're deleting the _lock attribute, but this might lead to unexpected behavior!"
            )
            del state["_lock"]

        logger.debug("QueryResult pickled.")

        # Return a dictionary with the pickled attribute and other picklable attributes
        return {"other": state, "dilled": dill_pickled_attributes}

    def __setstate__(self, state: dict) -> None:
        logger.debug("Starting to unpickle a Queryresult.")

        self.__dict__ = state["other"]
        self._generator = None

        self._lock = mp.Lock()
        self._index = mp.Value("i", 0)

        for attrib, attrib_pickled in state["dilled"].items():
            setattr(self, attrib, dill.loads(attrib_pickled))
        logger.debug("QueryResult unpickled.")

    def to_cache(self, path: Path) -> None:
        """
        Serialize the QueryResult object to a file at the given path.
        The _chunker_index is stored using klepto.dir_archive for efficient
        serialization.
        """
        if not os.path.isdir(path):
            raise RuntimeError("QueryResult::to_file is expected to be called with a directory path.")

        logger.info("Starting to cache QueryResult.")
        # Handle attributes that should not be stored via pickle/dill
        state = self.__dict__.copy()
        for attrib in ["_lock", "_index", "_generator", "_chunker_index"]:
            if attrib in state:
                del state[attrib]

        logger.debug("Removed unpickable attributed.")

        # Handle attributed that should be dilled (pickle is a bit faster, but pickling lambas needs dill)
        dilled = {}
        for attrib in ["_meta"]:
            dilled[attrib] = state[attrib]
            del state[attrib]

        logger.debug("Removed dillable attributes.")

        with open(path / "dilled.pkl", "wb") as f:
            dill.dump(dilled, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.debug("Stored dillable attributes.")

        with open(path / "pickled.pkl", "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.debug("Stored pickable attributes.")

        serialize_chunker_index(self._chunker_index, path / "chunker_index")

        logger.debug("Stored chunker index.")

    def replay(self, num_chunks_replay: int) -> None:
        if num_chunks_replay < 1:
            return

        logger.debug(f"Starting to replay {num_chunks_replay} chunks.")
        mixture_log = self._mixture_log

        mixture_log_index = 0
        num_mixture_changes = len(mixture_log)

        # Since there's always an entry at chunk index 0, set the initial mixture
        initial_mixture = mixture_log[0][1]
        assert initial_mixture is not None

        self.update_mixture(initial_mixture)

        mixture_log_index += 1

        # Initialize next mixture change, if any
        if mixture_log_index < num_mixture_changes:
            next_mixture_change_chunk_index = mixture_log[mixture_log_index][0]
            next_mixture = mixture_log[mixture_log_index][1]
        else:
            next_mixture_change_chunk_index = None
            next_mixture = None

        # Replay the chunks
        for i in range(num_chunks_replay):
            # Update mixture if the current chunk index matches a mixture change point
            if next_mixture_change_chunk_index is not None and i == next_mixture_change_chunk_index:
                assert next_mixture is not None
                self.update_mixture(next_mixture)
                mixture_log_index += 1
                if mixture_log_index < num_mixture_changes:
                    next_mixture_change_chunk_index = mixture_log[mixture_log_index][0]
                    next_mixture = mixture_log[mixture_log_index][1]
                else:
                    next_mixture_change_chunk_index = None

            try:
                _ = next(self)
            except StopIteration as e:
                raise RuntimeError(f"Generator exhausted during replay at chunk index {i} - should not happen!") from e

        logger.debug("Finished chunk replay.")
        assert self._num_returns_gen == num_chunks_replay
        assert self._index.get_obj().value == num_chunks_replay

    @classmethod
    def from_cache(cls, path: Path, replay: bool = True) -> "QueryResult":
        """
        Deserialize the QueryResult object from a file at the given path.
        The _chunker_index is loaded using klepto.dir_archive.
        """
        if not os.path.isdir(path):
            raise RuntimeError("QueryResult::from_cache expects a directory path.")
        logger.info("Loading QueryResult from cache.")

        # Load the pickled state
        with open(path / "pickled.pkl", "rb") as f:
            state = pickle.load(f)

        logger.debug("Loaded pickable attributes.")

        # Load the dilled attributes
        with open(path / "dilled.pkl", "rb") as f:
            dilled = dill.load(f)

        logger.debug("Loaded dillable attributes.")

        # Create a new instance without calling __init__
        query_result = cls.__new__(cls)

        # Set the state
        query_result.__dict__.update(state)

        # Set the dilled attributes
        for attrib, value in dilled.items():
            setattr(query_result, attrib, value)

        logger.debug("Instantiated QueryResult from pickle/dill.")

        # Initialize non-picklable attributes
        query_result._lock = mp.Lock()
        query_result._index = mp.Value("i", 0)

        num_chunks_replay = query_result._num_returns_gen

        query_result._num_returns_gen = 0  # reset for now, replay afterwards
        query_result._generator = None

        logger.debug("Instantiated non-pickable attributes.")

        query_result._chunker_index = deserialize_chunker_index(path / "chunker_index")

        logger.debug("Loaded chunker index.")

        if query_result._query_log_dir is not None:
            query_result._query_log_dir.mkdir(exist_ok=True)

        if num_chunks_replay > 0 and replay:
            query_result.replay(num_chunks_replay)

        return query_result
