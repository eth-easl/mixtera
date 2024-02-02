# Getting Started

## Indexing Dataset

Mixtera supports indexing datasets in an inverted index format. The indices are categories, and the values are the documents that belong to the category. To index a dataset (which, as of now, is a list of `.jsonl` files.), use the `prepare` function and provide your own processing method. For example:

```python
import os
import json
from src.engine.datasets import MixteraDataset
mds = MixteraDataset.from_folder(".cache/datasets/rpj_sample")
mds.prepare(process_fn=process_rpj_meta)
```

The `process_fn` is a function that takes a document and yields category. For example, the `process_rpj_meta` function in our example is defined as:

```python
# function yields (filename, line_id, key) tuples
def process_rpj_meta(dataset_path):
    def extract(example):
        if "language" in example and len(example["language"]) > 0:
            return f"language={example['language'][0]['name']}"
        elif "language" in example and len(example["language"]) == 0:
            return "language=unknown"
        elif "publication_date" in example:
            return f"publication_date={example['publication_date']}"
        else:
            return "unknown"

    files = [x for x in os.listdir(dataset_path) if x.endswith(".jsonl")]
    for file in files:
        with open(os.path.join(dataset_path, file)) as f:
            for line_id, line in enumerate(f):
                yield file, line_id, extract(json.loads(line)["meta"])
```

## Query Dataset

Once the dataset is indexed, you can query the dataset using the `Query` class. For example:

```python
from src.engine.datasets import MixteraDataset
from src.engine.operators import Query

mds = MixteraDataset.from_folder(".cache/datasets/rpj_sample")
query = Query.from_dataset(mds).select("language=Python")
query.display()
```

The above code will not actually execute the query, but only show the query plan, as shown below:

```
select<MixteraDataset(.cache/datasets/rpj_sample)>(language=Python)
```

To execute the query, use the `execute` method:

```python
from src.engine.datasets import MixteraDataset
from src.engine.operators import Query

mds = MixteraDataset.from_folder(".cache/datasets/rpj_sample")
query = Query.from_dataset(mds).select("language=Python")
query.display()
res = query.execute()
```

The `res` variable will contain the results of the query. The example output is shown below:

```
...
Applying select<MixteraDataset(.cache/datasets/rpj_sample)>(language=Python)
Applying materialize<>()
2024-02-02 16:10:41.981 | INFO     | src.engine.operators.builtins:apply:53 - Going to materialize 662 results...
2024-02-02 16:10:42.133 | INFO     | src.engine.operators.query:execute:44 - Query returned 662 samples
>>> print(res[0])
{'text': 'import pyxb.binding.generate\nimport pyxb.utils.domutils\nfrom...', ...}
```

Read more about the `Query` class in the [Query](query.md) section.