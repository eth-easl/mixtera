import time

import torch
from loguru import logger
from mixtera.core.datacollection import MixteraClient
from mixtera.core.datacollection.datasets import JSONLDataset
from mixtera.core.query import Query
from mixtera.torch.mixtera_torch_dataset import MixteraTorchDataset
from tqdm import tqdm


def parsing_func(sample):
    import json
    return json.loads(sample)["text"]


def main():
    register_dataset = True # only need to do once
    TRAINING_ID = str(round(time.time() * 1000)) # Each node should have the same TRAINING_ID, such that they can ask the server for the query_id (can be passed, e.g., via environment variable)
    num_workers_per_node = 1

    ### LOCAL CASE
    mdc = MixteraClient.from_directory("/Users/mboether/phd/mixtera")
    if register_dataset:
        mdc.register_dataset("test_dataset", "/Users/mboether/phd/mixtera/test_dataset", JSONLDataset, parsing_func, "RED_PAJAMA")

    query = Query.for_job(TRAINING_ID, num_workers_per_node).select(("language", "==", "HTML")) # num_nodes = 1 default
    _ = query.execute(mdc, chunk_size=100) # -> LocalQueryResult

    ### FORK ###

    local_result = []
    query_result = mdc.get_query_result(TRAINING_ID) # -> LocalQueryResult
    for sample in tqdm(mdc.stream_query_results(query_result)):
        local_result.append(sample)


    print("\n\nRemote\n\n")
    ## Remote case (without streaming)
    rdc = MixteraClient.from_remote("127.0.0.1", 8888)

    # Pre-fork on primary node
    query = Query.for_job(TRAINING_ID, num_workers_per_node, num_nodes=2).select(("language", "==", "HTML"))
    _ = query.execute(rdc, chunk_size=100) # -> RemoteQueryResult, most likely ignored

    ### FORK ###
    remote_non_tunnel_result = []
    query_result = rdc.get_query_result(TRAINING_ID) # -> RemoteQueryResult
    for sample in tqdm(rdc.stream_query_results(query_result, tunnel_via_server=False)):
        remote_non_tunnel_result.append(sample)

    if local_result != remote_non_tunnel_result:
        raise RuntimeError("Local does not equal remote non tunnel result!")

    ## Remote case (with streaming)
    TRAINING_ID = str(round(time.time() * 1000)) # Need a new job ID
    # Pre-fork on primary node
    query = Query.for_job(TRAINING_ID, num_workers_per_node, num_nodes=2).select(("language", "==", "HTML"))
    _ = query.execute(rdc, chunk_size=100) # -> RemoteQueryResult, most likely ignored

    ### FORK ###
    remote_tunnel_result = []
    query_result = rdc.get_query_result(TRAINING_ID) # -> RemoteQueryResult
    for sample in tqdm(rdc.stream_query_results(query_result, tunnel_via_server=True)):
        remote_tunnel_result.append(sample)

    if local_result != remote_tunnel_result:
        raise RuntimeError("Local does not equal remote tunnel result!")

    ### Torch Test
    TRAINING_ID = str(round(time.time() * 1000)) # Need a new job ID
    query = Query.for_job(TRAINING_ID, num_workers_per_node).select(("language", "==", "HTML"))
    torch_ds = MixteraTorchDataset(rdc, query, TRAINING_ID, 100, tunnel_via_server=False)
    dl = torch.utils.data.DataLoader(torch_ds, batch_size=2, num_workers=16)

    dataset_result = []
    for batch in tqdm(dl):
        dataset_result.extend(batch)

    if sorted(local_result) != sorted(dataset_result):
        raise RuntimeError("Local does not equal dataset tunnel result!")

    print(f"All had {len(local_result)} samples! (dataset = {len(dataset_result)})")


if __name__ == '__main__':
    main()
