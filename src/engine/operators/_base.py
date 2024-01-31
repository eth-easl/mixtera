from src.engine.datasets import MixteraDataset


class Operator:
    def __init__(self) -> None:
        self.children = []

    def __repr__(self):
        return f"{self.__class__.__name__}"

    def display(self):
        print(self)

    def set_ds(self, ds):
        self.ds = ds

    def display(self, level):
        print(f"{'-'*level}{'> ' if level > 0 else ''}{str(self)}")
        for child in self.children:
            if child:
                child.display(level+1)


def childOp(original_class):

    return original_class
