from collections.abc import Generator
from typing import Any, Callable, Optional, Type

from mixtera.core.datacollection import MixteraDataCollection
from mixtera.core.datacollection.datasets import Dataset
from mixtera.core.datacollection.index import IndexType
from mixtera.core.datacollection.index.index_collection import IndexFactory, IndexTypes
from mixtera.core.query.operators._base import Operator


class QueryPlan:
    """
    QueryPlan is a tree structure that represents the execution plan of a query.
    """

    def __init__(self) -> None:
        # The root should be None only when the query plan is empty
        # (i.e., when initializing).
        self.root: Optional[Operator] = None

    def display(self) -> None:
        if self.root:
            self.root.display(0)

    def is_empty(self) -> bool:
        return self.root is None

    def add(self, operator: "Operator") -> None:
        """
        This method adds an operator to the QueryPlan.
        By default, the new operator becomes the new root of the QueryPlan.
        However, each operator could have its own logic to insert
        itself into the QueryPlan (e.g., for `Select` it may create a new
        Intersection Operator).

        Args:
            operator (Operator): The operator to add.
        """
        if self.is_empty():
            self.root = operator
        else:
            self.root = operator.insert(self)

    def __str__(self) -> str:
        # The differnce between __str__ and display is that
        # __str__ returns the string representation of the query plan
        # while display prints the query plan in a tree format.
        if self.root:
            # there is a trailing newline, so we strip it
            return self.root.string(level=0).strip("\n")
        return "<empty>"


class Query:
    def __init__(self, mdc: MixteraDataCollection) -> None:
        self.mdc = mdc
        self.query_plan = QueryPlan()
        self.results: QueryResult

    def is_empty(self) -> bool:
        return self.query_plan.is_empty()

    @classmethod
    def register(cls, operator: Operator) -> None:
        """
        This method registers operators for the query.
        By default, all built-in operators (under ./operators) are registered.

        Args:
            operator (Operator): The operator to register.
        """
        op_name = operator.__name__.lower()

        def process_op(self, *args: Any, **kwargs: Any) -> "Query":  # type: ignore[no-untyped-def]
            op: Operator = operator(*args, **kwargs)
            # (todo: create issue) optimally we only set this if it is a leaf.
            op.datacollection = self.mdc
            self.query_plan.add(op)
            return self

        setattr(cls, op_name, process_op)

    @classmethod
    def from_datacollection(cls, mdc: MixteraDataCollection) -> "Query":
        """
        This method creates a Query object from a MixteraDataCollection object.

        Args:
            mdc (MixteraDataCollection): The MixteraDataCollection object.

        Returns:
            Query: The Query object.
        """
        return cls(mdc)

    @property
    def root(self) -> Operator:
        return self.query_plan.root

    def display(self) -> None:
        """
        This method displays the query plan in a tree
        format. For example:

        .. code-block:: python

            union<>()
            -> select<>(language == Go)
            -> select<>(language == CSS)
        """
        self.query_plan.display()

    def __str__(self) -> str:
        return str(self.query_plan)

    def execute(self, chunk_size: int = 1) -> "QueryResult":
        """
        This method executes the query and returns the resulting indices, in the form of a QueryResult object.

        Args:
            chunk_size (int): chunk_size is used to set the size of `subresults` in the QueryResult object.
                Defaults to 1. When iterating over a :py:class:`QueryResult`
                object, the results are yielded in chunks of size `chunk_size`.

        Returns:
            res (QueryResult): The results of the query. See :py:class:`QueryResult` for more details.
        """
        self.root.post_order_traverse()
        self.results = QueryResult(self.mdc, self.root.results, chunk_size=chunk_size)
        return self.results


class QueryResult:
    """QueryResult is a class that represents the results of a query.
    When constructing, it takes a list of indices (from the root of
    the query plan), a chunk size and a MixteraDataCollection object.

    The QueryResult object is iterable and yields the results in chunks of size `chunk_size`.

    The QueryResult object also has three meta properties: `dataset_type`,
    `file_path` and `parsing_func`, each of which is a dictionary that maps
    dataset/file ids to their respective types, paths and parsing functions.
    """

    def __init__(self, mdc: MixteraDataCollection, results: IndexType, chunk_size: int = 1) -> None:
        """
        Args:
            mdc (MixteraDataCollection): The MixteraDataCollection object.
            results (list): The list of results of the query.
            chunk_size (int): The chunk size of the results.
        """
        self.mdc = mdc
        self.chunk_size = chunk_size
        self.results = results
        self.chunks: list[IndexType] = []
        self.chunking()
        self._meta = self._parse_meta()

    def _parse_meta(self) -> dict:
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
            "dataset_type": {did: self.mdc._get_dataset_type_by_id(did) for did in dataset_ids},
            "parsing_func": {did: self.mdc._get_dataset_func_by_id(did) for did in dataset_ids},
            "file_path": {fid: self.mdc._get_file_path_by_id(fid) for fid in file_ids},
            "total_length": total_length,
        }

    @property
    def dataset_type(self) -> dict[str, Type[Dataset]]:
        return self._meta["dataset_type"]

    @property
    def file_path(self) -> dict[str, str]:
        return self._meta["file_path"]

    @property
    def parsing_func(self) -> dict[str, Callable[[str], str]]:
        return self._meta["parsing_func"]

    def chunking(self) -> None:
        # structure of self.chunks: [index_type, index_type, ...]
        # each index_type is an index, but contains exactly
        # chunk_size items, except the last one.
        chunks: list[dict] = []
        # here we chunk the self.results from a large index_type
        # into smaller index_types.
        current_chunk: IndexType = {}
        current_chunk_length = 0
        # this is likely not a good impl, optimize it later.
        # now just ensure correct chunk_size and each chunk is an index.
        # pylint: disable-next=too-many-nested-blocks
        for property_name in self.results.get_all_features():
            for property_value, property_dict in self.results.get_dict_index_by_feature(property_name).items():
                for did, files in property_dict.items():
                    for fid, file_ranges in files.items():
                        for start, end in file_ranges:
                            # TODO(create issue): we may want to optimize this later together with mixture.
                            # for now this is a simple implementaion (for testing)
                            if end - start <= self.chunk_size - current_chunk_length:
                                current_chunk[property_name] = {property_value: {did: {fid: [(start, end)]}}}
                                current_chunk_length += end - start
                                chunks.append(current_chunk)
                            else:
                                if self.chunk_size < current_chunk_length:
                                    raise RuntimeError("This should not happen.")
                                if self.chunk_size == current_chunk_length:
                                    chunks.append(current_chunk)
                                    current_chunk = {}
                                    current_chunk_length = 0
                                if self.chunk_size > current_chunk_length:
                                    # --> we need to split the range
                                    # --> and add the first part to the current chunk
                                    # --> and add the second part to a new chunk
                                    # let's drop the remainder for now
                                    num_chunks_needed = (end - start) // self.chunk_size
                                    for _ in range(num_chunks_needed):
                                        current_chunk[property_name] = {
                                            property_value: {did: {fid: [(start, start + self.chunk_size)]}}
                                        }
                                        chunks.append(current_chunk)
                                        current_chunk = {}
                                        current_chunk_length = 0
                                        start += self.chunk_size

        for chunk in chunks:
            chunk_index = IndexFactory.create_index(IndexTypes.IN_MEMORY_DICT_RANGE)
            chunk_index._index = chunk
            self.chunks.append(chunk_index)

    def __iter__(self) -> Generator[IndexType, None, None]:
        """Iterate over the results of the query with a chunk size
        (except the last chunk).

        """
        for chunk in self.chunks:
            yield chunk
