from src.engine.datasets import MixteraDataset
from ._base import SelectOperator, UnionOperator
from typing import List
from .planner import QueryPlan

class Query:
    dataset: MixteraDataset
    query_plan: QueryPlan

    @classmethod
    def from_dataset(cls, dataset:MixteraDataset):
        cls.dataset = dataset
        cls.query_plan = QueryPlan()
        return cls()

    def display(self):
        self.query_plan.display()
    
    def register(self, operator):
        pass