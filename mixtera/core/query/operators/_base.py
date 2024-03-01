from typing import List, Optional

from mixtera.core.datacollection import IndexType, MixteraDataCollection
from mixtera.core.query import query


class Operator:
    """
    Operator is a single node in the query plan tree. It has two main attributes:

    * children: List of child nodes (Operator).
        Child nodes are executed before the current node,
        The orders of the children are irrelevant - they can be executed in parallel.
    * results: List of results of the current node. Each item in the list is an index/pointer to a data sample.
        Each item in the list is an index/pointer to a data sample (if not materialized).
    """

    def __init__(self) -> None:
        self.children: List[Operator] = []
        self.results: List[IndexType] = []
        # for leaf nodes, the data collection needs to be set
        # for other nodes, we don't set the data collection as
        # they will not touch the underlying data collection.
        self.mdc: Optional[MixteraDataCollection] = None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"

    @property
    def datacollection(self) -> Optional[MixteraDataCollection]:
        return self.mdc

    @datacollection.setter
    def datacollection(self, value: MixteraDataCollection) -> None:
        self.mdc = value

    def insert(self, query_plan: "query.QueryPlan") -> "Operator":
        """
        For most operators, the insert function (insert this node into the query)
            adds the current root node as a child of this operator.
            For example: with "select().union()", when execute .union(),
            it inserts the "select" operator as a child of the "union" operator.

        Args:
            query (Query): The query to insert into the current operator.

        Returns:
            root (Operator): The new root of the query plan.

        """
        if query_plan.is_empty():
            return self
        self.children.append(query_plan.root)
        return self

    def display(self, level: int) -> None:
        print(f"{'-'*level}{'> ' if level > 0 else ''}{str(self)}")
        for child in self.children:
            child.display(level + 1)

    def post_order_traverse(self) -> None:
        for child in self.children:
            child.post_order_traverse()
        self.apply()

    def apply(self) -> None:
        raise NotImplementedError("apply method must be implemented in the child class")
