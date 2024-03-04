# Query

In Mixtera, a {py:class}`Query <mixtera.core.query.Query>` is a representation of how to retrieve data from the system. Each query is composed of a set of operations (called {py:class}`Operator <mixtera.core.query.Operator>`) that are executed in a specific order. The result of a query is the indexes of the data that match the query.

## High Level User Interface

The high level user interface for Mixtera is the {py:class}`Query <mixtera.core.query.Query>` class. This class is used to create and execute queries from a {py:class}`DataCollection <mixtera.core.datacollection.MixteraDataCollection>`. The following is an example of how to use the {py:class}`Query <mixtera.core.query.Query>` class on a {py:class}`LocalDataCollection <mixtera.core.datacollection.local.LocalDataCollection>`:

```python
from mixtera.core.datacollection.local import LocalDataCollection
from mixtera.core.query import Query

mdc = LocalDataCollection.from_directory(directory)
query = Query.from_datacollection(mdc)
    .select(("language", "==", "en"))
    .select(("year", ">", 2010))
# Print the query structure
query.display()
# now execute gives a QueryResult object
query_result = query.execute()

# access the metadata in the QueryResult object
# {1: <class 'mixtera.core.datacollection.datasets.jsonl_dataset.JSONLDataset'>}
print(query_result.dataset_type)
# {1: 'tmp/data/test.jsonl'}
print(query_result.file_path)
# {1: <function <lambda> at 0x7f759b4e9d00>}
print(query_result.parsing_func)

# and access the query results
for x in query_result:
    # this gives the returned index
    # [{1: {1: [(0, 2)]}}]
    # client is supposed to use this index, 
    # together with the metadata to access the actual data
    print(x)
```

## Query Processing

Each {py:class}`Query <mixtera.core.query.Query>` object holds a {py:class}`QueryPlan <mixtera.core.query.QueryPlan>`, which is a tree of {py:class}`Operator <mixtera.core.query.Operator>` objects.

For example, with the following code:

```python
query = Query.from_datacollection(mdc)
    .select(("language", "==", "en"))
```

Mixtera will create a {py:class}`QueryPlan <mixtera.core.query.QueryPlan>` with a single {py:class}`SelectOperator <mixtera.core.query.Select>`. If you add more operations to the query, Mixtera will create a tree of {py:class}`Operator <mixtera.core.query.Operator>` objects. You can use {py:meth}`query.display() <mixtera.core.query.Query.display()>` to visualize the tree.

For example, with the following chained operations:

```python
query_1 = Query.from_datacollection(mdc).select(("language", "==", "en"))
query_2 = Query.from_datacollection(mdc).select(("year", ">", 2010))
query = query_1.union(query_2)
query.display()
```

The corresponding tree will be:
    
```python
union<>()
-> select<<...mdc...>>(year > 2010)
-> select<<...mdc...>>(language == en)
```

When executing a query, Mixtera will traverse the tree in a post-order fashion, i.e., it will execute the children of a node before executing the node itself. For example, in the above tree, Mixtera will first execute the two select operations and then the union operation. Once the traverse is done, the result of the query will be the result of the root node.

:::{note}
Now, we always assume that the indices can reside in memory, and thus all operations are done in memory. In the future we may need to consider the case where the indices are too large to fit in memory.
:::

:::{note}
Now, we enforce that only the leaf nodes can access the indices. In other words, the indices are not accessible to the internal nodes. Internal nodes can only operate the indices returned by their children.

This also affects how we construct the query tree for chaining {py:class}`Select <mixtera.core.query.Select>`. Now if we have a query like:

```python
query_1 = Query.from_datacollection(mdc)
    .select(("language", "==", "en"))
    .select(("year", ">", 2010))
```

This will **not** create a tree like:

```python
select<...mdc...>(language == en)
-> select<...mdc...>(year > 2010)
```

Instead, it will create a tree like:
```python
Intersection()
-> select<...mdc...>(language == en)
-> select<...mdc...>(year > 2010)
```
:::


:::{note}
Right now, we assume that the entire query processing process does not require materializing the indices in the middle. We may want to rethink this assumption in the future.
:::

## Query Result

When executing a query with `query.execute(chunk_size=1)`, Mixtera will return a {py:class}`QueryResult <mixtera.core.query.QueryResult>` object. This object contains two parts:
1. The metadata of the data collection that the query is executed on.
2. The indices of the data that match the query.

The metadata, at this point, contains three dictionaries: `dataset_type`, `file_path` and `parsing_func`, each of which is a dictionary that maps dataset/file ids to their respective types, paths and parsing functions. For example:

```python
query_result = query.execute()

# access the metadata in the QueryResult object
print(query_result.dataset_type)
# {1: <class 'mixtera.core.datacollection.datasets.jsonl_dataset.JSONLDataset'>}
print(query_result.file_path)
# {1: 'tmp/data/test.jsonl'}
print(query_result.parsing_func)
# {1: <function <lambda> at 0x7f759b4e9d00>}
```

The {py:class}`QueryResult <mixtera.core.query.QueryResult>` is also an iterable object. Each element of the iterable is a chunk of indices that match the query (i.e., it is a fixed-length list of indices, where the `length==chunk_size`, except the last iteration). For example:

```python
query_result = query.execute(chunk_size=1)
for x in query_result:
    print(x)
    # this gives the returned index
    # [{1: {1: [(0, 2)]}}]
    # client is supposed to use this index, 
    # together with the metadata to access the actual data
```

:::{warning}
The implementation of the {py:class}`QueryResult <mixtera.core.query.QueryResult>` is still dummy. In the future, Once we have a `mixture` operator, we need to ensure the correct mixture in each chunk.
:::