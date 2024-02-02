"""Query Planner is a tree of operators"""

from ._base import Operator
from loguru import logger

class QueryPlan:
    def __init__(self) -> None:
        self.root = None

    def add(self, node: Operator):
        self.root = node.insert(self.root)

    def display(self):
        self.root.display(0)