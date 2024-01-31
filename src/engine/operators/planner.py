"""Query Planner is a tree of operators"""
from ._base import Operator

class QueryPlan:
    def __init__(self) -> None:
        self.root = None
        self.nodes = []

    def insert(self, node:Operator, parent:Operator=None):
        if parent is None:
            self.root = node
        else:
            parent.children.append(node)
            node.parent = parent
        self.nodes.append(node)
        
    def display(self, level=0):
        # traverse the tree from bottom to top
        for node in self.nodes:
            print("-"*level + str(node))
            if node.children:
                self.display(level+1)
        
