from typing import Optional

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
