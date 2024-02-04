from src.engine.datasets import MixteraDataset, RemoteMixteraDataset
from src.engine.operators import Query

mds = RemoteMixteraDataset.from_url("http://localhost:8000")

query = Query.from_dataset(mds).select("language=HTML")
res = query.execute(materialize=True, streaming=True)

print(res)