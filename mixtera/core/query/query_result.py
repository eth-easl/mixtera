import json
import multiprocessing as mp
import os
import pickle
import random
from collections import defaultdict
from copy import deepcopy
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
                    "log:": {
                        chk_id: mixture.stringified_mixture()
                        for chk_id, mixture in self._mixture_log
                    },
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
        current_partition: dict[Any, dict[Any, list[tuple[int, int]]]] = defaultdict(
            lambda: defaultdict(list)
        )

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
                            current_partition[dataset_id][file_id].append(
                                (current_range[0], current_range[1] - diff)
                            )
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
            key: self._generate_per_mixture_component_chunks(self._chunker_index, key)
            for key in chunker_index_keys
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

        while True:
            if no_success_counter > 10:
                logger.error("Hard-stopping chunk generation after 10 unsucessful tries.")
                return

            mixture = base_mixture.mixture_in_rows()
            is_strict = base_mixture.strict  # best-effort mode if False
            chunk_success = False
            if mixture:
                if previous_mixture != mixture:
                    logger.debug(f"Obtained new mixture: {mixture}")
                    mixture_id += 1
                    previous_mixture = mixture
                    self._mixture_log.append((current_chunk_index, deepcopy(base_mixture)))
                    self._persist_mixture_log()
                    self._update_key_id_map()

                chunk: ChunkerIndex = create_chunker_index()
                remaining_sizes: dict[MixtureKey, int] = {  # pylint: disable=unnecessary-comprehension
                        key: size for key, size in mixture.items()
                }
                original_sizes = remaining_sizes.copy()

                while any(size > 0 for size in remaining_sizes.values()):
                    # Sort to guarantee same handling for semantically same mixtures
                    for mixture_key in sorted(remaining_sizes.keys()):
                        logger.debug(
                            f"Handling key {mixture_key}, remaining sizes: {remaining_sizes}"
                        )

                        progress_made = True
                        while remaining_sizes[mixture_key] > 0 and progress_made:
                            progress_made = False
                            for component_key, iterator in sorted(component_iterators.items(), key=lambda x: x[0]):
                                if mixture_key == component_key:
                                    try:
                                        component_chunk: ChunkerIndexDatasetEntries = (
                                            iterator.send(remaining_sizes[mixture_key])
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
                                        remaining_sizes[mixture_key] = (
                                            remaining_sizes[mixture_key] - chunk_size
                                        )

                                        # logger.debug(
                                        #    f"Received chunk size: {chunk_size} for {mixture_key} from {component_key}"
                                        # )

                                        # Merge the component chunk into the main chunk
                                        for dataset_id, files in component_chunk.items():
                                            for file_id, ranges in files.items():
                                                chunk[mixture_key][dataset_id][file_id] = (
                                                    ranges
                                                    if file_id
                                                    not in chunk[mixture_key][dataset_id]
                                                    else merge_sorted_lists(
                                                        chunk[mixture_key][dataset_id][file_id],
                                                        ranges,
                                                    )
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
                                        continue

                            # No matching components found or all are exhausted
                            if not progress_made:
                                if is_strict:  # Unable to complete chunk
                                    logger.debug(
                                        "Did not make progress, unable to complete chunk."
                                    )
                                else:
                                    # best-effort generation
                                    num_missing_samples = remaining_sizes.pop(mixture_key)

                                    if not remaining_sizes:
                                        logger.debug("All mixture components are exhausted.")
                                        return

                                    # redistribute missing samples among other mixture keys
                                    total_original_size_remaining = sum(
                                        original_sizes[key] for key in sorted(remaining_sizes.keys())
                                    )

                                    ratios = [
                                        original_sizes[key] / total_original_size_remaining
                                        for key in sorted(remaining_sizes.keys())
                                    ]

                                    samples_to_distribute = distribute_by_ratio(num_missing_samples, ratios)

                                    for i, key in enumerate(sorted(remaining_sizes.keys())):
                                        remaining_sizes[key] += samples_to_distribute[i]

                                    break

                # Check if we have enough data for all mixture keys
                if not remaining_sizes or all(size == 0 for size in remaining_sizes.values()):
                    chunk_success = True
                    no_success_counter = 0
                    if current_chunk_index == target_chunk_index:
                        logger.debug(f"Yielding chunk {current_chunk_index}.")
                        self._persist_chunk_idx(current_chunk_index)
                        base_mixture, target_chunk_index = yield ResultChunk(
                            defaultdict_to_dict(chunk),
                            self.dataset_type,
                            self.file_path,
                            self.parsing_func,
                            base_mixture.chunk_size,
                            self._key_id_map,
                            mixture_id,
                            mixture=mixture,
                        )
                    else:
                        logger.debug(
                            f"current_chunk_index = {current_chunk_index} != target_chunk_index = {target_chunk_index}"
                        )
                # Not enough data to complete the chunk, end generation
                else:
                    logger.debug("Not enough data, ending chunk generation")
                    no_success_counter += 1
                    yield None
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
                        chunker_index_keys_idx = (chunker_index_keys_idx + 1) % len(
                            chunker_index_keys
                        )
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
                    base_mixture, target_chunk_index = yield ResultChunk(
                        chunk,
                        self.dataset_type,
                        self.file_path,
                        self.parsing_func,
                        base_mixture.chunk_size,
                        self._key_id_map,
                        mixture_id,
                        mixture=None,
                    )

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
            result = self._generator.send((self._mixture, chunk_target_index))
            if result is None:
                # In case we reach end of chunks (for now),
                # next time (e.g. due to updated mixture) we try again to fetch the same ID.
                with self._index.get_lock():
                    self._index.get_obj().value -= 1

                if self.stop_on_none:
                    raise StopIteration

            return result

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
            raise RuntimeError(
                "QueryResult::to_file is expected to be called with a directory path."
            )

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
            if (
                next_mixture_change_chunk_index is not None
                and i == next_mixture_change_chunk_index
            ):
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
                raise RuntimeError(
                    f"Generator exhausted during replay at chunk index {i} - should not happen!"
                ) from e

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
