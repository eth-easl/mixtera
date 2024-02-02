from src.engine.datasets import MixteraDataset
from src.engine.operators import Query

mds = MixteraDataset.from_folder(".cache/datasets/rpj_sample")
query_1 = Query.from_dataset(mds).select("language=C").select("language=Python").select("language=Java")
# query_1.display()


# query_2 = Query.from_dataset(mds).select("language=Python").select("language=C").union(query_1)
# query_2.display()

query_2 = Query.from_dataset(mds).select("language=HTML")
res = query_2.execute(materialize=True)

print(res[0])