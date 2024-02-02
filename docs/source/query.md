# Query and Query Plan

`Query` is a class that represents a query plan, that can be chained with supported operators to form a query. The query plan is a tree of operators, where each operator is a node in the tree. The query plan is not executed until the `execute` method is called.

For example, to construct a query plan that finds documents in either `Python` or `C`, you can create a query as follows:

```python
from src.engine.datasets import MixteraDataset
from src.engine.operators import Query

mds = MixteraDataset.from_folder(".cache/datasets/rpj_sample")
query = Query.from_dataset(mds).select("language=Python").union(Query.from_dataset(mds).select("language=C"))
query.display()
```

The constructed query plan will look like this:

```
union<>()
-> select<MixteraDataset(.cache/datasets/rpj_sample)>(language=C)
-> select<MixteraDataset(.cache/datasets/rpj_sample)>(language=Python)
```

To execute the query, use the `execute` method:

```python
res = query.execute()
```

When the query plan is executed, it performs a post order traverse of the tree, i.e., it first executes the leaves of the tree, and then their parents and so on until the root. In the above example, the two `select` operators are executed first, and then the `union` operator is executed. The `union` operator is executed by merging the results of its children.