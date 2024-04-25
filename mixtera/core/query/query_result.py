import multiprocessing as mp
from collections import defaultdict
from typing import Any, Callable, Optional, Type

import dill
import portion
from loguru import logger
from mixtera.core.datacollection import MixteraDataCollection
from mixtera.core.datacollection.datasets import Dataset
from mixtera.core.datacollection.index import ChunkerIndex, ChunkerIndexDatasetEntries, Index, IndexType, InvertedIndex
from mixtera.core.datacollection.index.index_collection import create_chunker_index, create_inverted_index_interval_dict
from mixtera.utils.utils import defaultdict_to_dict, generate_hashable_search_key, merge_property_dicts

from .mixture import Mixture


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
        results: Index,
        chunk_size: Optional[int] = 1,
        mixture: Optional[Mixture] = None,
    ) -> None:
        """
        Args:
            mdc (LocalDataCollection): The LocalDataCollection object.
            results (IndexType): The results of the query.
            chunk_size (int): The chunk size of the results. This parameter is mutually exclusive with
                the mixture parameter.
            mixture: A mixture object defining the mixture to be reflected in the chunks. This parameter is mutually
                exclusive with the chunk_size parameter.
        """
        if mixture is not None and chunk_size is not None:
            raise AttributeError(
                "Both mixture and chunk_size are specified! Only one of these parameter can be specified!"
            )

        self._chunk_size = chunk_size
        self._mixture = mixture

        self.results = results
        self._meta = self._parse_meta(mdc)
        self._chunks: list[ChunkerIndexDatasetEntries] = []
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

    @staticmethod
    def _invert_result(index: Index, enable_parallelism: bool = False) -> InvertedIndex:
        """
        Returns an InvertedIndex that points from files to an ordered dictionary
        of ranges (from portion) annotated with properties:
            {
                "dataset_id": {
                    "file_id": {
                        portion.Interval: {
                            feature_1_name: [feature_1_value, ...],
                            ...
                        },
                        ...
                    },
                    ...
                },
                ...
            }

        Args:
            index: an IndexType object that will be inverted.
            enable_parallelism: if True, the inversion is done in a multi-processed fashion

        Returns:
            An InvertedIndex
        """
        raw_index = index.get_full_dict_index(copy=False)

        if not enable_parallelism:  # pylint: disable=too-many-nested-blocks
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

        # Parallel approach
        range_workloads = []
        for property_name, property_values in raw_index.items():
            for property_value, datasets in property_values.items():
                for dataset_id, files in datasets.items():
                    for file_id, _ in files.items():
                        range_workloads.append((property_name, property_value, dataset_id, file_id))

        def _first_map_stage(payload: tuple[int, int]) -> dict[Any, Any]:
            """
            The first step in the inversion process. In this step, an inverted index is created on a per property,
            dataset and file id basis.

            Args:
                payload: A tuple consisting of the [low, hi) intervals that should be processed in range_workloads

            Returns:
                An inverted dictionary generated from the assigned range
            """
            lo_bound, hi_bound = payload
            partial_inverted_dictionary: InvertedIndex = create_inverted_index_interval_dict()
            for i in range(lo_bound, hi_bound):
                prop_name, prop_val, did, fid = range_workloads[i]
                interval_dictionary = partial_inverted_dictionary[did][fid]
                for rrange in raw_index[prop_name][prop_val][did][fid]:
                    interval_dictionary[
                        portion.closedopen(rrange[0], rrange[1])] = {property_name: [property_value]}
            return partial_inverted_dictionary

        with mp.Pool() as pool:
            work_partitions = list(range(0, len(range_workloads) + 1, len(range_workloads) // mp.cpu_count()))
            work_partitions[-1] = len(range_workloads)
            partial_results = pool.map(_first_map_stage, zip(work_partitions, work_partitions[1:]))

        # Results will need to be merged
        unique_keys = set()
        merged_partial_results: dict[Any, dict[Any, list]] = defaultdict(lambda: defaultdict(list))
        for partial_result in partial_results:
            for document_id, document_entry in partial_result.items():
                for file_id, interval_entries in document_entry.items():
                    unique_keys.add((document_id, file_id))
                    merged_partial_results[document_id][file_id].append(interval_entries)
        unique_keys = list(unique_keys)

        # Convert unique keys to list and start 2nd stage of map reduce
        def _second_map_stage(payload: tuple[int, int]) -> dict[Any, Any]:
            """
            The second step in the inversion process. In this step, an inverted index is created on
            a per dataset and file id basis.

            Args:
                payload: A tuple consisting of the [low, hi) intervals that should be processed in range_workloads

            Returns:
                An inverted dictionary generated from the assigned range
            """
            lo_bound, hi_bound = payload
            partial_inverted_dictionary: InvertedIndex = create_inverted_index_interval_dict()
            for i in range(lo_bound, hi_bound):
                document_id, file_id = unique_keys[i]
                interval_dictionary = partial_inverted_dictionary[document_id][file_id]
                for d in merged_partial_results[document_id][file_id]:
                    for intervals, props in d.items():
                        intersections = interval_dictionary[intervals]
                        interval_dictionary[intervals] = props
                        for intersection_range, intersection_properties in intersections.items():
                            interval_dictionary[intersection_range] = merge_property_dicts(
                                interval_dictionary[intersection_range].values()[0], intersection_properties
                            )
            return partial_inverted_dictionary

        with mp.Pool() as pool:
            work_partitions = list(range(0, len(unique_keys) + 1, len(unique_keys) // mp.cpu_count()))
            work_partitions[-1] = len(unique_keys)
            partial_results = pool.map(_second_map_stage, zip(work_partitions, work_partitions[1:]))

        inverted_dictionary: InvertedIndex = create_inverted_index_interval_dict()
        for d in partial_results:
            inverted_dictionary |= d

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
                    ],
                    ...
                },
                ...
            },
            ...
        }

        The ChunkerIndex object is useful for creating mixture chunks.

        Args:
            inverted_index: an InvertedIndex object.

        Returns: a ChunkerIndex object.

        """
        chunker_index: ChunkerIndex = create_chunker_index()

        for document_id, document_entries in inverted_index.items():
            for file_id, file_entries in document_entries.items():
                for intervals, interval_properties in file_entries.items():
                    # intervals can be a simplified interval (e.g. '[-7,-2) | [11,15)') so we need a for loop
                    for interval in intervals:
                        # Create a key for this interval; only the first value for a property is considered here
                        property_names = list(interval_properties.keys())
                        property_values = [interval_properties[property_name] for property_name in property_names]
                        hashable_key = generate_hashable_search_key(property_names, property_values)

                        # Add the interval to the chunk index
                        chunker_index[hashable_key][document_id][file_id].append([interval.lower, interval.upper])

        # TODO(DanGraur): Add option to parallelize: (1) count number of files (2) disseminate to procs equally;
        #                 each proc handles the list reduction and converts to a chunker index (3) chunker indexes are
        #                 returned and merged by main proc
        return chunker_index

    @staticmethod
    def _generate_per_mixture_component_chunks(
        chunker_index: ChunkerIndex, component_key: str, component_cardinality: int
    ) -> list[ChunkerIndexDatasetEntries]:
        """
        This method per chunks for each component of a mixture (e.g. 25% medicine + english) given a key into
        the chunking index and the cardinality (in number of instances) for the component
        (e.g. 25% --> 25 instances).

        Args:
            chunker_index: The chunking index
            component_key: chunking index key
            component_cardinality: number of instances required for the component of the mixture

        Returns:
            A list of component chunks. The list has the following format:
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
        """
        component_chunks = []
        target_ranges = chunker_index[component_key]

        current_cardinality = 0
        current_partition: dict[Any, dict[Any, list[tuple[int, int]]]] = defaultdict(lambda: defaultdict(list))

        for dataset_id, document_entries in target_ranges.items():  # pylint: disable=too-many-nested-blocks
            for file_id, ranges in document_entries.items():
                for base_range in ranges:
                    current_range = (base_range[0], base_range[1])
                    range_cardinality = current_range[1] - current_range[0]
                    if current_cardinality + range_cardinality < component_cardinality:
                        current_partition[dataset_id][file_id].append(current_range)
                        current_cardinality += range_cardinality
                    else:
                        # Add the partial range and the full component
                        diff = current_cardinality + range_cardinality - component_cardinality
                        current_partition[dataset_id][file_id].append((current_range[0], current_range[1] - diff))
                        component_chunks.append(defaultdict_to_dict(current_partition))

                        # Prepare the rest of the range and new component
                        current_range = (current_range[1] - diff, current_range[1])
                        current_partition = defaultdict(lambda: defaultdict(list))
                        current_cardinality = 0

                        # Process the remaining range
                        continue_processing = current_range[1] > current_range[0]
                        while continue_processing:
                            range_cardinality = current_range[1] - current_range[0]
                            if current_cardinality + range_cardinality < component_cardinality:
                                current_partition[dataset_id][file_id].append(current_range)
                                current_cardinality += range_cardinality
                                continue_processing = False
                            else:
                                diff = current_cardinality + range_cardinality - component_cardinality
                                current_partition[dataset_id][file_id].append(
                                    (current_range[0], current_range[1] - diff)
                                )
                                component_chunks.append(defaultdict_to_dict(current_partition))

                                # Prepare the rest of the range and new component
                                current_range = (current_range[1] - diff, current_range[1])
                                current_partition = defaultdict(lambda: defaultdict(list))
                                current_cardinality = 0

                                # Stop if range has been exhausted perfectly
                                continue_processing = current_range[1] > current_range[0]

        if current_cardinality > 0:
            component_chunks.append(defaultdict_to_dict(current_partition))

        return component_chunks

    def _temp_chunker(self) -> list[ChunkerIndex]:
        """
        This is a dummy method for implementing chunking added here to not break the unit tests.
        """
        inverted_index: InvertedIndex = self._invert_result(self.results)
        chunker_index: ChunkerIndex = self._create_chunker_index(inverted_index)
        # TODO(DanGraur): (1) add logic that stores some mixture data structure (2) add logic that can generate chunks
        #                 (3) add unit test for it (4) [separately of this] add multiprocessing to inverted/chunk index

        mixture = self._mixture.get_mixture()  # type: ignore[union-attr]
        mixture_keys = mixture.keys()
        component_chunks = [
            self._generate_per_mixture_component_chunks(chunker_index, key, mixture[key]) for key in mixture_keys
        ]

        chunks = []
        for components in zip(*component_chunks):
            chunk = {}
            for component in zip(mixture_keys, components):
                chunk[component[0]] = component[1]
            chunks.append(chunk)

        return chunks

    def chunking(self) -> None:
        """
        Implements the chunking logic. This method populates the self._chunks variable with chunks fulfilling the
        requirements specified by the self._mixture object (if it exists). Otherwise, simply produces chunks of given
        chunk size using any data.
        """
        inverted_index: InvertedIndex = self._invert_result(self.results)
        chunker_index: ChunkerIndex = self._create_chunker_index(inverted_index)
        # TODO(DanGraur): (1) add logic that stores some mixture data structure (2) add logic that can generate chunks
        #                 (3) add unit test for it (4) [separately of this] add multiprocessing to inverted/chunk index

        if self._mixture is not None:
            # If we have a mixture, we use that to generate the chunks
            mixture = self._mixture.get_mixture()
            mixture_keys = mixture.keys()
            component_chunks = [
                self._generate_per_mixture_component_chunks(chunker_index, key, mixture[key]) for key in mixture_keys
            ]

            # Chunks will be composed using multiple properties
            for components in zip(*component_chunks):
                chunk = {}
                for component in zip(mixture_keys, components):
                    chunk[component[0]] = component[1]
                self._chunks.append(chunk)
        else:
            # Each chunk stems from a single property
            # Aliasing has to be done to avoid formatters complaining of line lengths
            f_alias = self._generate_per_mixture_component_chunks
            for key in chunker_index.keys():
                temp = [{key: x} for x in f_alias(chunker_index, key, self._chunk_size)]  # type: ignore[arg-type]
                self._chunks.extend(temp)

    @property
    def chunk_size(self) -> int:
        if self._mixture is not None:
            return self._mixture.chunk_size
        return self._chunk_size  # type: ignore[return-value]

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
