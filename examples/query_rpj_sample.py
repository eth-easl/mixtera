from src.engine.datasets import MixteraDataset
from src.engine.operators import Query

mds = MixteraDataset.from_folder(".cache/datasets/rpj_sample")
query = Query.from_dataset(mds).select("language=C").select("language=Python").select("language=Java")


query.display()
