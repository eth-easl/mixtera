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

query = { 2: { 2: [(0,2)]}} # replace this with an actual query
query_id = rdc.register_query(query, TRAINING_ID, num_nodes, num_workers_per_node) # When do we execute the query at the server - already here or when first client starts streaming? What is our execution model?
# This also means that the query is not created `from_mdc` but instead a mdc registers a query
# The main reason is that we somehow have to handle multi-node/multi-worker data loading

# On other nodes we can do:
query_id2 = rdc.get_query_id(TRAINING_ID)
assert query_id == query_id2

for sample in rdc.stream_query_results(query_id, node_id, worker_id):
    print(sample)
