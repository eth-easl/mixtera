from collections.abc import Generator
from typing import Any, Dict, List

from mixtera.core.datacollection import MixteraDataCollection
from mixtera.core.query.operators._base import Operator


class QueryPlan:
    """
    QueryPlan is a tree structure that represents the execution plan of a query.
    """

    def __init__(self) -> None:
        self.root = None

    def add(self, node: Operator) -> None:
        self.root = node.insert(self.root)

    def display(self) -> None:
        if self.root:
            self.root.display(0)


class QueryResult:
    """QueryResult is a class that represents the results of a query.

    Args:
        mdc (MixteraDataCollection): The MixteraDataCollection object.
        results (list): The list of results of the query.
        chunk_size (int): The chunk size of the results.
    """

    def __init__(self, mdc: MixteraDataCollection, results: List, chunk_size: int = 1) -> None:
        self.mdc = mdc
        self.chunk_size = chunk_size
        self._meta = self._parse_meta(results)
        self.results = results

    def _parse_meta(self, indices: List) -> Dict:
        dataset_ids = set()
        file_ids = set()
        for idx in indices:
            dataset_ids.update(idx.keys())
            for val in idx.values():
                file_ids.update(val)
        return {
            "dataset_type": {did: self.mdc._get_dataset_type_by_id(did) for did in dataset_ids},
            "file_path": {fid: self.mdc._get_file_path_by_id(fid) for fid in file_ids},
        }

    @property
    def dataset_type(self) -> Dict:
        return self._meta["dataset_type"]

    @property
    def file_path(self) -> Dict:
        return self._meta["file_path"]

    def __iter__(self) -> Generator[List, None, None]:
        """Iterate over the results of the query with a chunk size.

        This method is very dummy right now without ensuring the correct mixture.
        """
        for i in range(0, len(self.results), self.chunk_size):
            yield self.results[i : i + self.chunk_size]


class Query:

    def __init__(self, mdc: MixteraDataCollection) -> None:
        self.mdc = mdc
        self.query_plan = QueryPlan()
        self.results: QueryResult

    @classmethod
    def register(cls, operator: Operator) -> None:
        """
        This method registers operators for the query.
        By default, all built-in operators (under ./operators) are registered.

        Args:
            operator (_type_): _description_
        """
        op_name = operator.__name__.lower()

        def process_op(self, *args: Any, **kwargs: Any) -> "Query":  # type: ignore[no-untyped-def]
            op: Operator = operator(*args, **kwargs)
            op.set_datacollection(self.mdc)
            self.query_plan.add(op)
            return self

        setattr(cls, op_name, process_op)

    @classmethod
    def from_datacollection(cls, mdc: MixteraDataCollection) -> "Query":
        return cls(mdc)

    @property
    def root(self) -> Operator:
        return self.query_plan.root

    def display(self) -> None:
        self.query_plan.display()

    def execute(self, chunk_size: int = 1) -> QueryResult:
        """
        This method executes the query and returns the resulting indices, in the form of a QueryResult object.

        Args:
            chunk_size (int): chunk_size is used to
            set the size of `subresults` in the
            QueryResult object. Defaults to 1.
            When iterating over a QueryResult object,
            the results are yielded in chunks of size `chunk_size`.

        Returns:
            QueryResult: _description_
        """
        self.root.post_order_traverse()
        self.results = QueryResult(self.mdc, self.root.results, chunk_size=chunk_size)
        return self.results
