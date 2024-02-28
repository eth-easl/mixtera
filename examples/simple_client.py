import time

from mixtera.core.datacollection import MixteraDataCollection

TRAINING_ID = str(round(time.time() * 1000)) # Each node should have the same TRAINING_ID, such that they can ask the server for the query_id (can be passed, e.g., via environment variable)
prefetch_buffer_size = 50000
transfer_chunk_size = 16384
node_id = 0
worker_id = 0
num_nodes = 1
num_workers_per_node = 1

rdc = MixteraDataCollection.from_remote("127.0.0.1", 8888, prefetch_buffer_size)

# On primary node, do this
query = { 2: { 2: [(0,2)]}} # TODO(#): Use actual query instead of dict of ranges
query_id = rdc.register_query(query, TRAINING_ID, num_workers_per_node, num_nodes=num_nodes)
# This also means that the query is not created `from_mdc` but instead a mdc registers a query
# The main reason is that we somehow have to handle multi-node/multi-worker data loading

# On other nodes we can do:
query_id2 = rdc.get_query_id(TRAINING_ID)
assert query_id == query_id2

server_results = []
for sample in rdc.stream_query_results(query_id, worker_id, node_id=node_id):
    print(sample)
    server_results.append(sample)


### LOCAL CASE
num_workers = 1
ldc = MixteraDataCollection.from_directory("/Users/mboether/phd/mixtera")

# Internally, the server should do this and forward the results.
query_id = ldc.register_query(query, TRAINING_ID, num_workers) # num_nodes=1 is default - but when we start executing is still unclear

# Here, we fork (e.g., in PyTorch DataLoader) and instantiate the workers
worker_id = 0 # We would get this from Torch/TF/...

local_results = []
for sample in ldc.stream_query_results(query_id, worker_id): # node_id = 0 is default
    print(sample)
    local_results.append(sample)

if local_results == server_results:
    print("Success!")
else:
    raise AssertionError("Non equal results.")

### Torch Test

def processing_func(sample: str) -> str:
    return "processed_" + sample # actually used for tokenization or sth

torch_ds = MixteraTorchDataset(ldc/mdc, processing_func)

# test torch ds