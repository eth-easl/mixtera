"""
This script demonstrates integrating Mixtera's data loading capabilities with PyTorch's data processing pipeline by using
the MixteraTorchDataset. This class allows seamless integration with PyTorch's DataLoader.

The example first creates a local client with a Mixtera dataset as done in the local client example. It then sets up a
MixteraTorchDataset using a specified query. The DataLoader from PyTorch is then used to iterate over the dataset in
batches, simulating a scenario where one would train a machine learning model on the streamed data. This is also to showcase
that Mixtera works correctly without data duplication for multiple data loader workers.

This example is particularly useful for users who are looking to incorporate Mixtera's data querying and streaming
capabilities into a machine learning workflow with PyTorch. It showcases how Mixtera can be used to feed data directly
into a model training loop.
"""

import tempfile
import time
from pathlib import Path
from typing import Any, Optional

import torch
from mixtera.core.client import MixteraClient
from mixtera.core.client.mixtera_client import QueryExecutionArgs, ResultStreamingArgs
from mixtera.core.datacollection.datasets import JSONLDataset
from mixtera.core.datacollection.index.parser import MetadataParser
from mixtera.core.datacollection.index.parser.metadata_parser import MetadataProperty
from mixtera.core.query import Query
from mixtera.core.query.mixture import ArbitraryMixture, Mixture
from mixtera.torch import MixteraTorchDataset


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
    @classmethod
    def get_properties(cls) -> list[MetadataProperty]:
        return [
            MetadataProperty(
                name="language",
                dtype="ENUM",
                multiple=False,
                nullable=False,
                enum_options={"JavaScript", "HTML"},
            ),
            MetadataProperty(
                name="license",
                dtype="STRING",
                multiple=False,
                nullable=False,
                enum_options={"CC", "MIT"},
            ),  # Could be ENUM but we are using string to test
            MetadataProperty(
                name="doublelanguage",
                dtype="ENUM",
                multiple=True,
                nullable=False,
                enum_options={"JavaScript", "HTML"},
            ),
        ]

    def parse(
        self, line_number: int, payload: Any, **kwargs: Optional[dict[Any, Any]]
    ) -> None:
        metadata = payload["meta"]
        self.add_metadata(
            sample_id=line_number,
            language=metadata["language"],
            license=metadata["license"],
            doublelanguage=[metadata["language"], metadata["language"]],
        )


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
        "local_integrationtest_dataset",
        directory / "testd.jsonl",
        JSONLDataset,
        parsing_func,
        "TEST_PARSER",
    )

    return client


def setup_torch_dataset(
    client: MixteraClient,
    job_id: str,
    query: Query,
    num_workers: int,
    mixture: Mixture,
    tunnel: bool,
):
    # Creating a torch dataset.
    qea = QueryExecutionArgs(mixture=mixture, num_workers=num_workers)
    rsa = ResultStreamingArgs(job_id=job_id, tunnel_via_server=tunnel)
    torch_ds = MixteraTorchDataset(client, query, qea, rsa)
    return torch_ds


def main():
    with tempfile.TemporaryDirectory() as temp_dir:
        # Setup the local client with a temporary directory.
        # This also populates the database with a dummy dataset, where 50% of data is tagged HTML and 50% is tagged JavaScript.
        client = setup_local_client(Path(temp_dir))
        job_id = str(round(time.time() * 1000))
        mixture = ArbitraryMixture(chunk_size=42)
        query = Query.for_job(job_id).select(("language", "==", "JavaScript"))

        num_workers = 2

        torch_ds = setup_torch_dataset(
            client, job_id, query, num_workers, mixture=mixture, tunnel=False
        )
        dataloader = torch.utils.data.DataLoader(
            torch_ds, batch_size=10, num_workers=num_workers
        )

        for batch in dataloader:
            for sample in batch:
                assert (
                    int(sample) % 2 == 0
                ), f"Sample {sample} should not appear for JavaScript"
            print(batch)  # Here, you could process the batch as needed.

        print("Successfully ran torch dataset wrapper example!")


if __name__ == "__main__":
    main()
