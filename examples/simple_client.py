import time
import json

from mixtera.core.datacollection import MixteraDataCollection
from mixtera.core.datacollection.datasets import JSONLDataset
from mixtera.core.filesystem import LocalFilesystem
from mixtera.core.query import Query

register_dataset = False # only need to do once
TRAINING_ID = str(round(time.time() * 1000)) # Each node should have the same TRAINING_ID, such that they can ask the server for the query_id (can be passed, e.g., via environment variable)
prefetch_buffer_size = 50000
num_workers_per_node = 1
parsing_func = lambda sample: json.loads(sample)["text"]


### LOCAL CASE
ldc = MixteraDataCollection.from_directory("/Users/mboether/phd/mixtera")
if register_dataset:
    ldc.register_dataset("test_dataset", "/Users/mboether/phd/mixtera/test_dataset", JSONLDataset, LocalFilesystem, parsing_func, "RED_PAJAMA")

query = Query.for_training(TRAINING_ID, num_workers_per_node).select(("language", "==", "JavaScript")) # num_nodes = 1 default
_ = query.execute(ldc, chunk_size=2) # -> LocalQueryResult

### FORK ###

## TODO: Ensure that we fork here and not duplicate between processes the data. Should be unique.

query_result = ldc.get_query_result(TRAINING_ID) # -> LocalQueryResult
for sample in ldc.stream_query_results(query_result):
    print(sample)


print("\n\nRemote\n\n")
## Remote case (without streaming)
rdc = MixteraDataCollection.from_remote("127.0.0.1", 8888, prefetch_buffer_size)

# Pre-fork on primary node
query = Query.for_training(TRAINING_ID, num_workers_per_node, num_nodes=2).select(("language", "==", "JavaScript"))
_ = query.execute(rdc, chunk_size=2) # -> RemoteQueryResult, most likely ignored

### FORK ###

query_result = rdc.get_query_result(TRAINING_ID) # -> RemoteQueryResult
for sample in rdc.stream_query_results(query_result, tunnel_via_server=False):
    print(sample)

## Remote case (with streaming)
TRAINING_ID = str(round(time.time() * 1000)) # Need a new training ID
# Pre-fork on primary node
query = Query.for_training(TRAINING_ID, num_workers_per_node, num_nodes=2).select(("language", "==", "JavaScript"))
_ = query.execute(rdc, chunk_size=2) # -> RemoteQueryResult, most likely ignored

### FORK ###

query_result = rdc.get_query_result(TRAINING_ID) # -> RemoteQueryResult
for sample in rdc.stream_query_results(query_result, tunnel_via_server=True):
    print(sample)



raise RuntimeError("TODO: Torch Dataset")

### Torch Test

def processing_func(sample: str) -> str:
    return "processed_" + sample # actually used for tokenization or sth

torch_ds = MixteraTorchDataset(ldc/mdc, processing_func)

# test torch ds