from src.engine.datasets import MixteraDataset
from typing import List
from .planner import QueryPlan
from loguru import logger
from .builtins import Select, Union, Materialize
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
        def process_op(self, *args, **kwargs):
            op = operator(*args, **kwargs)
            if isinstance(args[0], Query):
                args[0].root.display(0)
            op.set_ds(self.dataset)
            self.query_plan.add(op)
            return self
        setattr(cls, op_name, process_op)

    def display(self):
        self.query_plan.display()

    @property
    def root(self):
        return self.query_plan.root

    def execute(self, materialize=True):
        if materialize:
            mat_op = Materialize()
            mat_op.set_ds(self.dataset)
            self.query_plan.add(mat_op)
        self.root.cleanup()
        self.root.post_order_traverse()
        logger.info(f"Query returned {len(self.root.results)} samples")
        return self.root.results

def register(operator):
    Query.register(operator)

register(Select)
register(Union)
register(Materialize)