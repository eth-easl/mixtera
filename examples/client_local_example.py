"""
This script demonstrates the use of Mixtera's local client interface to interact with a simulated dataset. The dataset is
composed of 1000 samples where each sample is a JSONL line with a text field and metadata indicating whether the language
is JavaScript or HTML. This example sets up a local client, registers a dataset, and runs a query to filter samples based
on the language metadata.

A custom metadata parser (TestMetadataParser) is used to index the metadata fields, which are then queried using Mixtera's
query interface. The setup showcases the full lifecycle of data preparation, registration, and querying within the Mixtera
ecosystem in a local environment.

The result of the query is a collection of samples containing text from JavaScript-coded entries, which is verified in the
assertion check. This example gives a basic understanding of creating and querying datasets using Mixtera locally.
"""

import tempfile
import time
from pathlib import Path
from typing import Any, Optional

from mixtera.core.client import MixteraClient, QueryExecutionArgs, ResultStreamingArgs
from mixtera.core.datacollection.datasets import JSONLDataset
from mixtera.core.datacollection.index.parser import MetadataParser
from mixtera.core.query import ArbitraryMixture, Query


def write_jsonl(path: Path) -> None:
    data = ""
    for i in range(1000):
        data = (
            data
            + '{ "text": "'
            + str(i)
            + '", "meta": { "language": "'
            + ("JavaScript" if i % 2 == 0 else "HTML")
            + '", "license": "CC"}}\n'
        )

    with open(path, "w") as text_file:
        text_file.write(data)


class TestMetadataParser(MetadataParser):
    def parse(self, line_number: int, payload: Any, **kwargs: Optional[dict[Any, Any]]) -> None:
        metadata = payload["meta"]
        self._index.append_entry("language", metadata["language"], self.dataset_id, self.file_id, line_number)
        self._index.append_entry("license", metadata["license"], self.dataset_id, self.file_id, line_number)


def parsing_func(sample):
    import json
    return json.loads(sample)["text"]

def setup_local_client(directory: Path):
    # Writing JSONL data to the directory, which simulates the dataset.
    write_jsonl(directory / "testd.jsonl")
    
    # Instantiating a client from a local directory to interact with the datasets locally.
    client = MixteraClient.from_directory(directory)
    
    # Register the metadata parser.
    client.register_metadata_parser("TEST_PARSER", TestMetadataParser)
    
    # Registering the dataset with the client.
    client.register_dataset(
        "local_integrationtest_dataset", directory / "testd.jsonl", JSONLDataset, parsing_func, "TEST_PARSER"
    )
    
    return client

def run_query(client: MixteraClient, chunk_size: int):
    job_id = str(round(time.time() * 1000)) # Get some job ID based on current timestamp
    query = Query.for_job(job_id).select(("language", "==", "JavaScript")) # In our example, we want to query all samples tagged JavaScript
    mixture = ArbitraryMixture(chunk_size=chunk_size)

    query_args = QueryExecutionArgs(mixture=mixture)
    client.execute_query(query, query_args)

    result_args = ResultStreamingArgs(job_id)
    result_samples = list(client.stream_results(result_args))
    
    # Checking the number of results and their validity.
    assert len(result_samples) == 500, f"Got {len(result_samples)} samples instead of the expected 500!"
    for sample in result_samples:
        assert int(sample) % 2 == 0, f"Sample {sample} should not appear for JavaScript"

def main():
    with tempfile.TemporaryDirectory() as temp_dir:
        # Setup the local client with a temporary directory.
        # This also populates the database with a dummy dataset, where 50% of data is tagged HTML and 50% is tagged JavaScript.
        client = setup_local_client(Path(temp_dir))
        chunk_size = 42 # Size of the result chunks of the query
        run_query(client, chunk_size)
        

if __name__ == "__main__":
    main()