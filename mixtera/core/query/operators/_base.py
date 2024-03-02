from typing import List, Optional

from mixtera.core.datacollection import IndexType, MixteraDataCollection
from mixtera.core.query import query


class Operator:
    """
    Operator is a single node in the query plan tree. It has two main attributes:

    * children: List of child nodes (Operator).
        Child nodes are executed before the current node,
        The orders of the children are irrelevant - they can be executed in parallel.
    * results: List of results of the current node.
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

        Each operator can override this method to customize the insertion logic.

        Args:
            query (Query): The query to insert into the current operator.

        Returns:
            root (Operator): The new root of the query plan.

        """
        if not query_plan.is_empty():
            self.children.append(query_plan.root)
        return self

    def display(self, level: int) -> None:
        """
        Prints the query plan in a tree format.

        Args:
            level (int): The level of the current node in the tree.

        """
        print(f"{'-'*level}{'> ' if level > 0 else ''}{str(self)}")
        for child in self.children:
            child.display(level + 1)

    def string(self, level: int) -> str:
        """
        Returns the string representation of this node (including its children)
        in a tree format.

        Args:
            level (int): The level of the current node in the tree.

        Returns:
            str: The string representation of the current node and its children.
        """

        # we may want to customize this method in the future so it can be
        # parsed and used in other places (e.g., maybe a web interface)
        node_string = f"{'-'*level}{'> ' if level > 0 else ''}{str(self)}\n"
        for child in self.children:
            node_string += child.string(level + 1)
        return node_string

    def post_order_traverse(self) -> None:
        for child in self.children:
            child.post_order_traverse()
        self.apply()

    def apply(self) -> None:
        raise NotImplementedError("apply method must be implemented in the child class")
