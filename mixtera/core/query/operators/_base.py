from itertools import chain
from typing import List, Union

from mixtera.core.datacollection import MixteraDataCollection
import mixtera.core.query.query as query

class Operator:
    """
    Operator is a single node in the query plan tree. It has two main attributes:

    * children: List of child nodes.
        Child nodes are executed before the current node,
        The orders of the children are irrelevant - they can be executed in parallel.
    * results: List of results of the current node.
        Each item in the list is an index/pointer to a data sample (if not materialized).
    """

    def __init__(self) -> None:
        self.children: List[Operator] = []
        self.results: Union[List, chain] = []
        self._materialized = False
        self.mdc: MixteraDataCollection = None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"

    def set_datacollection(self, data_collection: MixteraDataCollection) -> None:
        self.mdc = data_collection

    def insert(self, query: "query.QueryPlan") -> "Operator":
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
        if query.is_empty():
            return self
        self.children.append(query.root)
        return self

    def display(self, level: int) -> None:
        print(f"{'-'*level}{'> ' if level > 0 else ''}{str(self)}")
        for child in self.children:
            if child:
                child.display(level + 1)

    def post_order_traverse(self) -> None:
        for child in self.children:
            if child:
                child.post_order_traverse()
        self.apply()

    def apply(self) -> None:
        raise NotImplementedError("apply method must be implemented in the child class")
