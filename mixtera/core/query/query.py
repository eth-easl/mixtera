from typing import Any

from loguru import logger
from mixtera.core.datacollection import MixteraDataCollection
from mixtera.core.query.operators._base import Operator
from mixtera.core.query.query_plan import QueryPlan
from mixtera.core.query.query_result import QueryResult

from .mixture import Mixture


class Query:
    def __init__(self, job_id: str) -> None:
        self.query_plan = QueryPlan()
        self.results: Any | None = None
        self.job_id = job_id

    def is_empty(self) -> bool:
        return self.query_plan.is_empty()

    @classmethod
    def register(cls, operator: type[Operator]) -> None:
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
    def for_job(cls, job_id: str) -> "Query":
        """
        Factory method to instantiate a new query for a given job id.

        Args:
            job_id (str): The job_id to instantiate a query for.
        Returns:
            Query: The Query object.
        """
        return cls(job_id)

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

    def execute(self, mdc: MixteraDataCollection, mixture: Mixture) -> None:
        """
        This method executes the query and returns the resulting indices, in the form of a QueryResult object.
        Args:
            mdc: The MixteraDataCollection object required to execute the query
            mixture: A mixture object defining the mixture to be reflected in the chunks.
        """
        logger.debug(f"Executing query locally with chunk size {mixture.chunk_size}")
        sql_query, parameters = self.root.generate_sql(mdc._connection)
        logger.debug(f"SQL:\n{sql_query}\nParameters:\n{parameters}")
        self.results = QueryResult(mdc, mdc._connection.execute(sql_query, parameters).pl(), mixture)

        logger.debug(f"Results:\n{self.results}")
        logger.debug("Query executed.")
