from mixtera.core.query.operators.intersect import Intersection
from mixtera.core.query.operators.materialize import Materialize
from mixtera.core.query.operators.select import Select
from mixtera.core.query.operators.union import Union

from .operators._base import Operator
from .query import Query
from .query_plan import QueryPlan
from .query_result import LocalQueryResult, QueryResult, RemoteQueryResult

Query.register(Select)
Query.register(Union)
Query.register(Intersection)

__all__ = [
    "Query",
    "Operator",
    "QueryPlan",
    "Select",
    "Union",
    "Materialize",
    "LocalQueryResult",
    "Intersection",
    "RemoteQueryResult",
    "QueryResult",
]
