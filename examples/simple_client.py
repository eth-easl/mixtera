import json
import time

from loguru import logger
from mixtera.core.datacollection import MixteraDataCollection
from mixtera.core.datacollection.datasets import JSONLDataset
from mixtera.core.filesystem import LocalFilesystem
from mixtera.core.query import Query

register_dataset = True # only need to do once
TRAINING_ID = str(round(time.time() * 1000)) # Each node should have the same TRAINING_ID, such that they can ask the server for the query_id (can be passed, e.g., via environment variable)
num_workers_per_node = 1

def parsing_func(sample):
    try:
        return json.loads(sample)["text"]
    except Exception:
        logger.error("empty")
        logger.error(sample)
        logger.error("empty")


### LOCAL CASE
ldc = MixteraDataCollection.from_directory("/Users/mboether/phd/mixtera")
if register_dataset:
    ldc.register_dataset("test_dataset", "/Users/mboether/phd/mixtera/test_dataset", JSONLDataset, LocalFilesystem, parsing_func, "RED_PAJAMA")

query = Query.for_training(TRAINING_ID, num_workers_per_node).select(("language", "==", "JavaScript")) # num_nodes = 1 default
_ = query.execute(ldc, chunk_size=2) # -> LocalQueryResult

### FORK ###

## TODO: Ensure that we fork here and not duplicate between processes the data. Should be unique.

local_result = []
query_result = ldc.get_query_result(TRAINING_ID) # -> LocalQueryResult
for sample in ldc.stream_query_results(query_result):
    local_result.append(sample)


print("\n\nRemote\n\n")
## Remote case (without streaming)
rdc = MixteraDataCollection.from_remote("127.0.0.1", 8888)

# Pre-fork on primary node
query = Query.for_training(TRAINING_ID, num_workers_per_node, num_nodes=2).select(("language", "==", "JavaScript"))
_ = query.execute(rdc, chunk_size=2) # -> RemoteQueryResult, most likely ignored

### FORK ###
remote_non_tunnel_result = []
query_result = rdc.get_query_result(TRAINING_ID) # -> RemoteQueryResult
for sample in rdc.stream_query_results(query_result, tunnel_via_server=False):
    remote_non_tunnel_result.append(sample)

if local_result != remote_non_tunnel_result:
    raise RuntimeError("Local does not equal remote non tunnel result!")

## Remote case (with streaming)
TRAINING_ID = str(round(time.time() * 1000)) # Need a new training ID
# Pre-fork on primary node
query = Query.for_training(TRAINING_ID, num_workers_per_node, num_nodes=2).select(("language", "==", "JavaScript"))
_ = query.execute(rdc, chunk_size=2) # -> RemoteQueryResult, most likely ignored

### FORK ###
remote_tunnel_result = []
query_result = rdc.get_query_result(TRAINING_ID) # -> RemoteQueryResult
for sample in rdc.stream_query_results(query_result, tunnel_via_server=True):
    remote_tunnel_result.append(sample)

if local_result != remote_tunnel_result:
    raise RuntimeError("Local does not equal remote tunnel result!")


raise RuntimeError("TODO: Torch Dataset")

### Torch Test

def processing_func(sample: str) -> str:
    return "processed_" + sample # actually used for tokenization or sth

torch_ds = MixteraTorchDataset(ldc/mdc, processing_func)

# test torch ds