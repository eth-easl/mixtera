"""
This script illustrates how to interact with a dataset using Mixtera's server interface. The example first sets up a
temporary directory with a simulated dataset in the same format as the local client example. However, this time the
dataset is intended to be used with a server client.

After starting the server, the script connects to the server and runs queries in two modes: with and without tunneling.
Tunneling is a feature that, when enabled, streams the results via the server instead of directly from the file system.
This is useful in distributed environments where the client may not have direct access to the data location.

The script also demonstrates the same assertion checks as in the local client example, ensuring the correctness of the
query results.

This example helps understanding the server-client architecture and how to perform similar operations as the
local client but with the additional server layer.
"""

import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

from mixtera.core.client import MixteraClient
from mixtera.core.client.mixtera_client import QueryExecutionArgs, ResultStreamingArgs
from mixtera.core.datacollection.datasets import JSONLDataset
from mixtera.core.datacollection.index.parser import MetadataParser
from mixtera.core.datacollection.index.parser.metadata_parser import MetadataProperty
from mixtera.core.query import Query
from mixtera.core.query.mixture import ArbitraryMixture


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


def run_query(client: MixteraClient, chunk_size: int, tunnel: bool):
    """
    This runs the example query on the client. As you might notice, this is almost the same
    as for the local client. In the server case, we are able to tunnel the file reads via the
    server as well. In the local case, we use the default of no tunneling.
    """
    job_id = str(
        round(time.time() * 1000)
    )  # Get some job ID based on current timestamp
    query = Query.for_job(job_id).select(
        ("language", "==", "JavaScript")
    )  # In our example, we want to query all samples tagged JavaScript
    mixture = ArbitraryMixture(chunk_size=chunk_size)
    qea = QueryExecutionArgs(mixture=mixture)
    client.execute_query(query, qea)
    client.wait_for_execution(job_id)

    rsa = ResultStreamingArgs(job_id=job_id)
    result_samples = list(client.stream_results(rsa))

    # Checking the number of results and their validity.
    assert (
        len(result_samples) == 500
    ), f"Got {len(result_samples)} samples instead of the expected 500!"
    for (
        _,
        _,
        sample,
    ) in result_samples:  # The first argument is the index in the current chunk, needed for state recovery. The second argument is the domain id.
        assert int(sample) % 2 == 0, f"Sample {sample} should not appear for JavaScript"


def main(server_host: str, server_port: int):
    """
    Note that as the server interface is not fully implemented yet,
    we first prepare the database at the server using a local client.
    Afterwards, we can run the query on the server.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # Prepares the directory for the server, similar to the local client example.
        server_dir = Path(temp_dir)

        # Writing JSONL data to the directory, which simulates the dataset.
        write_jsonl(server_dir / "testd.jsonl")

        # Instructing the user to start the server.
        print(
            f"Please start the Mixtera server in another terminal with the following command:\n\n\tmixtera-server {server_dir}\n"
        )
        print("After starting the server, press Enter to continue...")
        input()

        # Connecting to the Mixtera server.
        client = MixteraClient.from_remote(server_host, server_port)

        client.register_metadata_parser("TEST_PARSER", TestMetadataParser)

        # Registering the dataset with the client.
        if not client.register_dataset(
            "server_integrationtest_dataset",
            server_dir / "testd.jsonl",
            JSONLDataset,
            parsing_func,
            "TEST_PARSER",
        ):
            raise RuntimeError("Error while registering dataset.")

        # Run queries on server
        chunk_size = 42
        for tunnel in [False, True]:
            run_query(client, chunk_size, tunnel)

        print("Successfully ran server client example!")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(
            "Usage: python client_server_example.py <server_host> <server_port> (e.g. localhost 8888)"
        )
        sys.exit(1)

    host = sys.argv[1]
    port = int(sys.argv[2])
    main(host, port)
