from src.engine.datasets import MixteraDataset


class Operator:
    def __init__(self) -> None:
        self.children = []
        self.results = []

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

    def post_order_traverse(self):
        for child in self.children:
            if child:
                child.post_order_traverse()
        self.apply()

    def cleanup(self):
        #todo(xiaozhe): we'd better remove 'none' while constructing...
        self.children = [x for x in self.children if x]
        for child in self.children:
            child.cleanup()

def parent_op(original_class):
    def parent_op_insert(self, parent):
        self.children.append(parent)
        return self
    original_class.insert = parent_op_insert
    return original_class