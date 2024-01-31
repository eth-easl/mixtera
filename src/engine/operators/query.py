from src.engine.datasets import MixteraDataset
from typing import List
from .planner import QueryPlan
from loguru import logger


class Query:
    dataset: MixteraDataset
    query_plan: QueryPlan

    def __init__(self, ds) -> None:
        self.dataset = ds
        self.query_plan = QueryPlan()
        
    @classmethod
    def from_dataset(cls, dataset: MixteraDataset):
        return cls(dataset)

    @classmethod
    def register(cls, operator):
        op_name = operator.__name__.lower()
        logger.info(f"Registering operator {op_name}")

        def process_op(self, *args, **kwargs):
            op = operator(*args, **kwargs)
            print("args")
            if isinstance(args[0], Query):
                args[0].root.display(0)
            op.set_ds(self.dataset)
            logger.info(f"Processing operator {op_name}")
            """add op to the query plan"""
            self.query_plan.add(op)
            return self

        setattr(cls, op_name, process_op)

    def display(self):
        self.query_plan.display()

    @property
    def root(self):
        return self.query_plan.root


def register(operator):
    Query.register(operator)
