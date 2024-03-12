from typing import Any, Optional

from loguru import logger
from mixtera.core.datacollection import MixteraDataCollection
from mixtera.core.query.operators._base import Operator
from mixtera.core.query.query_plan import QueryPlan
from mixtera.core.query.query_result import LocalQueryResult, QueryResult


class Query:
    def __init__(self, training_id: str, num_workers_per_node: int, num_nodes: int) -> None:
        self.query_plan = QueryPlan()
        self.results: Optional[LocalQueryResult] = None
        self.training_id = training_id
        self.num_workers_per_node = num_workers_per_node
        self.num_nodes = num_nodes
        self.query_id: Optional[int] = None

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
            self.query_plan.add(op)
            return self

        setattr(cls, op_name, process_op)

    @classmethod
    def for_training(cls, training_id: str, num_workers_per_node: int, num_nodes: int = 1) -> "Query":
        """
        TODO
        Args:
            mdc (MixteraDataCollection): The MixteraDataCollection object.
        Returns:
            Query: The Query object.
        """
        return cls(training_id, num_workers_per_node, num_nodes)

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

    def execute(self, mdc: MixteraDataCollection, chunk_size: int = 1) -> "QueryResult":
        """
        This method executes the query and returns the resulting indices, in the form of a QueryResult object.
        Args:
            chunk_size (int): chunk_size is used to set the size of `subresults` in the QueryResult object.
                Defaults to 1. When iterating over a :py:class:`QueryResult`
                object, the results are yielded in chunks of size `chunk_size`.
        Returns:
            res (QueryResult): The results of the query. See :py:class:`QueryResult` for more details.
        """
        if mdc.is_remote():
            logger.debug(f"Executing query remotely with chunk size {chunk_size}")
            return mdc.execute_query_at_server(self, chunk_size)

        logger.debug(f"Executing query locally with chunk size {chunk_size}")
        self.root.post_order_traverse(mdc)
        # TODO(#31): Use num_nodes and workers to guarantee correct mixture.
        self.results = LocalQueryResult(mdc, self.root.results, chunk_size=chunk_size)
        # We first execute the query, and then register it.
        self.query_id = mdc.register_query(self, chunk_size)

        return self.results
