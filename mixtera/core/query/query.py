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
        conn = mdc._connection

        # -- BEGIN INNER SELECTION QUERY GENERATION  -- #
        # Fetch the table schema
        schema_query = "PRAGMA table_info('samples');"
        schema_info = conn.execute(schema_query).fetchall()
        # schema_info is a list of tuples with the following structure:
        # (cid, name, type, notnull, dflt_value, pk)

        # Build a schema mapping: column_name -> {type: ..., multiple: bool}
        schema = {}
        for _, name, col_type, notnull, _, _ in schema_info:
            # Determine if the column is an array type (ends with '[]')
            is_array = col_type.strip().endswith("[]")
            base_type = col_type.strip("[]").strip()
            schema[name] = {
                "type": base_type,
                "multiple": is_array,
                "nullable": not notnull == 1,  # notnull == 1 means NOT NULL constraint
            }

        logger.debug(schema)
        base_query, parameters = self.root.generate_sql(schema)
        logger.debug(f"SQL:\n{base_query}\nParameters:\n{parameters}")

        # -- BEGIN OF OUTER QUERY GENERATION -- #

        # First, we need to get the column names from the base query
        columns_query = f"SELECT * FROM ({base_query}) LIMIT 0"
        columns = conn.execute(columns_query, parameters).fetch_arrow_table().column_names

        # Determine group columns (all columns except 'sample_id')
        group_cols = ["dataset_id", "file_id"] + sorted(
            [col for col in columns if col not in ["sample_id", "dataset_id", "file_id"]]
        )

        # Create the partition by clause for the window functions
        partition_clause = ", ".join(group_cols)

        # Wrap the base query in a CTE and add the chunking logic
        full_query = f"""
        WITH base_data AS (
            {base_query}
        ),
        grouped_samples AS (
            SELECT
                *,
                sample_id - LAG(sample_id, 1, sample_id)
                    OVER (PARTITION BY {partition_clause} ORDER BY sample_id) AS diff
            FROM base_data
        ),
        intervals AS (
            SELECT
                {', '.join(group_cols)},
                SUM(CASE WHEN diff != 1 THEN 1 ELSE 0 END)
                    OVER (PARTITION BY {partition_clause} ORDER BY sample_id) AS group_id,
                MIN(sample_id) as interval_start,
                MAX(sample_id) + 1 as interval_end
            FROM grouped_samples
            GROUP BY {partition_clause}, diff, sample_id
        )
        SELECT
            {', '.join(group_cols)},
            group_id,
            MIN(interval_start) as interval_start,
            MAX(interval_end) as interval_end
        FROM intervals
        GROUP BY {partition_clause}, group_id
        ORDER BY {', '.join(group_cols)}, interval_start
        """

        self.results = QueryResult(mdc, mdc._connection.execute(full_query, parameters).pl(), mixture)

        logger.debug(f"Results:\n{self.results}")
        logger.debug("Query executed.")
