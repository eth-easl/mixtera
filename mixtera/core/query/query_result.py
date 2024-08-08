import multiprocessing as mp
import os
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Generator, Type

import dill
import portion
from loguru import logger
from mixtera.core.datacollection import MixteraDataCollection
from mixtera.core.datacollection.datasets import Dataset
from mixtera.core.datacollection.index import (
    ChunkerIndex,
    ChunkerIndexDatasetEntries,
    Index,
    IndexCommonType,
    InvertedIndex,
)
from mixtera.core.datacollection.index.index_collection import create_chunker_index, create_inverted_index_interval_dict
from mixtera.core.query.mixture import Mixture, MixtureKey
from mixtera.core.query.result_chunk import ResultChunk
from mixtera.utils.utils import defaultdict_to_dict, hash_list, merge_property_dicts, seed_everything

_NUM_CPU = os.cpu_count() or 1
INVERSION_POOL_SIZE = max(_NUM_CPU // 2, 1)  # TODO(#91): Make this configurable.


@dataclass
class InversionFileTask:
    dataset_id: str
    file_id: str
    ranges_per_property: (
        defaultdict[tuple[str, str], list[IndexCommonType]] | dict[tuple[str, str], list[IndexCommonType]]
    ) = field(
        default_factory=lambda: defaultdict(lambda: [])  # pylint: disable=unnecessary-lambda
    )  # The lambdas are necessary to satisfy mypy and defaultdict


@dataclass
class InversionFileTaskResult:
    dataset_id: str
    file_id: str
    interval_dict: portion.IntervalDict


def handle_inversion_task(task: InversionFileTask) -> InversionFileTaskResult:
    interval_dict = portion.IntervalDict()
    for (property_name, property_value), ranges in task.ranges_per_property.items():
        for row_range in ranges:
            range_interval = portion.closedopen(row_range[0], row_range[1])
            intersections = interval_dict[range_interval]
            interval_dict[range_interval] = {property_name: [property_value]}

            for intersection_range, intersection_properties in intersections.items():
                interval_dict[intersection_range] = merge_property_dicts(
                    interval_dict[intersection_range].values()[0], intersection_properties
                )

    return InversionFileTaskResult(dataset_id=task.dataset_id, file_id=task.file_id, interval_dict=interval_dict)


class QueryResult:
    """QueryResult is a class that represents the results of a query.
    The QueryResult object is iterable and yields the results in chunks of size `chunk_size`.

    The QueryResult object also has three meta properties: `dataset_type`,
    `file_path` and `parsing_func`, each of which is a dictionary that maps
    dataset/file ids to their respective types, paths and parsing functions.
    """

    def __init__(self, mdc: MixteraDataCollection, results: Index, mixture: Mixture) -> None:
        """
        Args:
            mdc (LocalDataCollection): The LocalDataCollection object.
            results (IndexType): The results of the query.
            mixture: A mixture object defining the mixture to be reflected in the chunks.
        """
        # Prepare structures for iterable chunking
        self._mixture = mixture
        self.results = results
        logger.debug("Instantiating QueryResult. Inverting index.")
        self._inverted_index: InvertedIndex = self._invert_result(self.results)
        logger.debug("Index inverted, creating chunker index.")
        self._chunker_index: ChunkerIndex = QueryResult._create_chunker_index(self._inverted_index)
        logger.debug("Chunker index created, informing mixture and parsing metadata.")

        self._mixture.inform(self._chunker_index)

        # Set up the auxiliary data structures
        self._meta = self._parse_meta(mdc)

        # A process holding a QueryResult might fork (e.g., for dataloaders).
        # Hence, we need to store the locks etc in shared memory.

        # Cross-process iterator state
        self._lock = mp.Lock()
        self._index = mp.Value("i", 0)

        #  The generator will be created lazily when calling __next__
        self._generator: Generator[ResultChunk, tuple[Mixture, int], None] | None = None
        self._num_returns_gen = 0
        logger.debug("QueryResult instantiated.")

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

    @staticmethod
    def _invert_result(index: Index) -> InvertedIndex:
        """
        Returns an InvertedIndex that points from files to an ordered dictionary
        of ranges (from portion) annotated with properties:
            {
                "dataset_id": {
                    "file_id": {
                        portion.Interval: {
                            feature_1_name: [feature_1_value_0, ...],
                            ...
                            feature_n_name: [feature_n_value_0, ...],
                        },
                        ...
                    },
                    ...
                },
                ...
            }

        Args:
            index: an IndexType object that will be inverted.

        Returns:
            An InvertedIndex
        """
        if INVERSION_POOL_SIZE == 1:
            logger.debug("Using single-threaded inversion.")
            return QueryResult._invert_result_st(index)

        logger.debug("Using multi-threaded inversion.")
        return QueryResult._invert_result_mt(index)

    @staticmethod
    def _invert_result_mt(index: Index) -> InvertedIndex:
        """
        Returns an InvertedIndex that points from files to an ordered dictionary
        of ranges (from portion) annotated with properties:
            {
                "dataset_id": {
                    "file_id": {
                        portion.Interval: {
                            feature_1_name: [feature_1_value_0, ...],
                            ...
                            feature_n_name: [feature_n_value_0, ...],
                        },
                        ...
                    },
                    ...
                },
                ...
            }

        Args:
            index: an IndexType object that will be inverted.

        Returns:
            An InvertedIndex
        """
        raw_index = index.get_full_dict_index(copy=False)
        inverted_dictionary: InvertedIndex = create_inverted_index_interval_dict()

        # Build tasks
        tasks: defaultdict[str, dict[str, InversionFileTask]] = defaultdict(
            lambda: {}  # pylint: disable=unnecessary-lambda
        )
        for property_name, property_values in raw_index.items():
            for property_value, datasets in property_values.items():
                for dataset_id, files in datasets.items():
                    for file_id, ranges in files.items():
                        if file_id not in tasks[dataset_id]:
                            tasks[dataset_id][file_id] = InversionFileTask(dataset_id=dataset_id, file_id=file_id)
                        tasks[dataset_id][file_id].ranges_per_property[(property_name, property_value)].extend(ranges)

        # Flattening the tasks dictionary into a list of task objects and converting defaultdict to dict
        # We ignore the type here since mypy complains that setattr does not return, which we fix using "or task"
        task_list = [
            setattr(task, "ranges_per_property", defaultdict_to_dict(task.ranges_per_property)) or task  # type: ignore
            for dataset_tasks in tasks.values()
            for task in dataset_tasks.values()
        ]

        logger.debug(f"Prepared {len(task_list)} parsing tasks. Execution with pool of size {INVERSION_POOL_SIZE}.")

        # Execute tasks
        with mp.Pool(INVERSION_POOL_SIZE) as pool:
            results = pool.map(handle_inversion_task, task_list)

        # Collect results
        for inversion_result in results:
            inverted_dictionary[inversion_result.dataset_id][inversion_result.file_id] = inversion_result.interval_dict

        return inverted_dictionary

    @staticmethod
    def _invert_result_st(index: Index) -> InvertedIndex:
        """
        Returns an InvertedIndex that points from files to an ordered dictionary
        of ranges (from portion) annotated with properties:
            {
                "dataset_id": {
                    "file_id": {
                        portion.Interval: {
                            feature_1_name: [feature_1_value_0, ...],
                            ...
                            feature_n_name: [feature_n_value_0, ...],
                        },
                        ...
                    },
                    ...
                },
                ...
            }

        Args:
            index: an IndexType object that will be inverted.

        Returns:
            An InvertedIndex
        """
        raw_index = index.get_full_dict_index(copy=False)
        inverted_dictionary: InvertedIndex = create_inverted_index_interval_dict()

        for property_name, property_values in raw_index.items():  # pylint: disable=too-many-nested-blocks
            for property_value, datasets in property_values.items():
                for dataset_id, files in datasets.items():
                    for file_id, ranges in files.items():
                        interval_dict = inverted_dictionary[dataset_id][file_id]
                        for row_range in ranges:
                            range_interval = portion.closedopen(row_range[0], row_range[1])
                            intersections = interval_dict[range_interval]
                            interval_dict[range_interval] = {property_name: [property_value]}
                            for intersection_range, intersection_properties in intersections.items():
                                interval_dict[intersection_range] = merge_property_dicts(
                                    interval_dict[intersection_range].values()[0], intersection_properties
                                )

        return inverted_dictionary

    @staticmethod
    def _create_chunker_index(inverted_index: InvertedIndex) -> ChunkerIndex:
        """
        Create a ChunkerIndex object from an InvertedIndex object. A ChunkerIndex has the following structure:
        {
            "prop_0_name:prop_0_value;prop_1_name:prop_1_value...": {
                dataset_0_id: {
                    file_0_id: [
                        [lo_bound_0, hi_bound_0],
                        ...
                        [lo_bound_n, hi_bound_n],
                    ],
                    ...
                },
                ...
            },
            ...
        }

        The ChunkerIndex object is useful for creating mixture chunks. It is guaranteed that all generated intervals
        are disjoint, and only represent one possible combination of properties and their values.

        Args:
            inverted_index: an InvertedIndex object.

        Returns: a ChunkerIndex object.

        """
        chunker_index: ChunkerIndex = create_chunker_index()

        for document_id, document_entries in inverted_index.items():
            for file_id, file_entries in document_entries.items():
                for intervals, interval_properties in file_entries.items():
                    # Intervals can be a simplified interval (e.g. '[-7,-2) | [11,15)') so we need a for loop
                    for interval in intervals:
                        mixture_key = MixtureKey(interval_properties)

                        # Add the interval to the chunk index based on the generated key
                        chunker_index[mixture_key][document_id][file_id].append([interval.lower, interval.upper])

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
        Updates the mixture to be used for future chunks

        Args:
            mixture: the new mixture object
        """
        with self._lock:
            self._mixture = mixture

    def _chunk_generator(self) -> Generator[ResultChunk, tuple[Mixture, int], None]:
        """
        Implements the chunking logic. This method yields chunks relative to  a mixture object.

        This method is a coroutine that accepts a mixture object that dictates the size of each chunk and potentially
        the mixture itself. The coroutine also accepts a target index specifying which chunk should be yielded next.
        This latter parameter is useful when chunking in a multiprocessed environment and at most once visitation
        guarantees are required.
        """
        # Variables for an arbitrary mixture
        current_chunk_index = 0
        chunker_index_keys_idx = 0
        chunker_index_keys = list(self._chunker_index.keys())
        empty_key_idx: set[int] = set()
        seed_everything(hash_list([str(key) for key in chunker_index_keys]))
        random.shuffle(chunker_index_keys)

        # Create coroutines for component iterators and advance them to the first yield
        component_iterators = {
            key: self._generate_per_mixture_component_chunks(self._chunker_index, key)
            for key in self._chunker_index.keys()
        }
        for _, component_iterator in component_iterators.items():
            try:
                next(component_iterator)
            except StopIteration:
                return

        base_mixture, target_chunk_index = yield
        while True:  # pylint: disable=too-many-nested-blocks
            # Get the mixture from the caller as it might have changed
            mixture = base_mixture.mixture_in_rows()

            if mixture:
                # Try to fetch a chunk part from each of the components. Here one of two things will happen:
                #   1. All the chunk's components can yield --> we will be able to build a chunk, or
                #   2. At least one of the chunk's components cannot yield --> StopIteration will be implicitly raised
                #      and the coroutine will pass the exception upstream to __next__
                chunk = {}
                number_of_keys_yielded = 0
                for mixture_key in mixture.keys():
                    for key in sorted(self._chunker_index.keys()):
                        try:
                            if mixture_key == key:
                                chunk[key] = component_iterators[key].send(mixture[mixture_key])
                                number_of_keys_yielded += 1
                                break
                        except StopIteration:
                            continue

                if number_of_keys_yielded != len(mixture):
                    # One of the components could not yield; we will not be able to build a chunk
                    return

                if current_chunk_index == target_chunk_index:
                    base_mixture, target_chunk_index = yield ResultChunk(
                        chunk, self.dataset_type, self.file_path, self.parsing_func, base_mixture.chunk_size, mixture
                    )
            else:
                chunk = None
                while len(empty_key_idx) < len(chunker_index_keys) and chunk is None:
                    chunker_index_keys_idx = (chunker_index_keys_idx + 1) % len(chunker_index_keys)
                    if chunker_index_keys_idx in empty_key_idx:
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
                        chunk, self.dataset_type, self.file_path, self.parsing_func, base_mixture.chunk_size, mixture
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
            #  The generator is created lazily since the QueryResult object might be pickled
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

        #  The following attributes are pickled using dill since they are not pickable by
        # the default pickler (used by torch)
        dill_pickled_attributes = {}
        for attrib in ["_meta", "_chunker_index", "_inverted_index"]:
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
