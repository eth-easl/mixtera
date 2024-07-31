import json
import os
import time
from typing import Any, Optional

from mixtera.core.client import MixteraClient
from mixtera.core.client.mixtera_client import QueryExecutionArgs, ResultStreamingArgs
from mixtera.core.datacollection.datasets import JSONLDataset
from mixtera.core.datacollection.index.parser import MetadataParser
from mixtera.core.query import ArbitraryMixture, Query
from timeit import default_timer as timer
import multiprocessing as mp


if __name__ == "__main__":
    # mp.set_start_method('spawn')
    
    directory = "/mnt/scratch/xiayao/cache/datasets/slimpajamas/slimpajama"
    datasets = [x for x in os.listdir(directory) if x.endswith(".jsonl")]
    client = MixteraClient.from_directory(directory)

    def run_query(client: MixteraClient, chunk_size: int):
        job_id = str(round(time.time() * 1000))
        query = Query.for_job(job_id).select(("setname", "==", "RedPajamaCommonCrawl"))
        mixture = ArbitraryMixture(chunk_size=chunk_size)
        client.execute_query(query, QueryExecutionArgs(mixture))
        return client.stream_results(ResultStreamingArgs(job_id=job_id))

    class TestMetadataParser(MetadataParser):
        def parse(self, line_number: int, payload: Any, **kwargs: Optional[dict[Any, Any]]) -> None:
            metadata = payload["meta"]
            self._index.append_entry("setname", metadata["redpajama_set_name"], self.dataset_id, self.file_id, line_number)

    def parsing_func(sample):
        return json.loads(sample)["text"]

    # client.register_metadata_parser("TEST_PARSER", TestMetadataParser)

    # client.register_dataset(f"slimpajama",directory, JSONLDataset, parsing_func,"TEST_PARSER")
    start = timer()
    result = run_query(client, 1000)

    for res in result:
        print('streaming result...')
        break

    end = timer()
    print(f"Time elapsed: {end - start:.2f}")