from typing import Any, Optional

from loguru import logger
from mixtera.core.datacollection import MixteraDataCollection
from mixtera.core.query.operators._base import Operator
from mixtera.core.query.query_plan import QueryPlan
from mixtera.core.query.query_result import QueryResult


class Query:
    def __init__(self, training_id: str) -> None:
        self.query_plan = QueryPlan()
        self.results: Optional[QueryResult] = None
        self.training_id = training_id

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
    def for_training(cls, training_id: str) -> "Query":
        """
        Factory method to instantiate a new query for a given job id.

        Args:
            training_id (str): The training_id to instantiate a query for.
        Returns:
            Query: The Query object.
        """
        return cls(training_id)

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

    def execute(self, mdc: MixteraDataCollection, chunk_size: int = 1) -> None:
        """
        This method executes the query and returns the resulting indices, in the form of a QueryResult object.
        Args:
            chunk_size (int): chunk_size is used to set the size of `subresults` in the QueryResult object.
                Defaults to 1. When iterating over a :py:class:`QueryResult`
                object, the results are yielded in chunks of size `chunk_size`.
        """
        logger.debug(f"Executing query locally with chunk size {chunk_size}")
        self.root.post_order_traverse(mdc)
        self.results = QueryResult(mdc, self.root.results, chunk_size=chunk_size)
        logger.debug("Query executed.")
