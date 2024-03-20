import sys
from pathlib import Path

from mixtera.core.client import MixteraClient
from mixtera.core.datacollection.datasets import JSONLDataset


def parsing_func(sample):
    import json

    return json.loads(sample)["text"]


def main() -> None:
    from utils import TestMetadataParser, write_jsonl

    server_dir = Path(sys.argv[1])
    print(f"Prepping directory {server_dir}.")
    write_jsonl(server_dir / "testd.jsonl")
    client = MixteraClient.from_directory(server_dir)
    client.register_metadata_parser("TEST_PARSER", TestMetadataParser)
    client.register_dataset(
        "ldc_integrationtest_dataset", server_dir / "testd.jsonl", JSONLDataset, parsing_func, "TEST_PARSER"
    )
    print("Directory prepped.")


if __name__ == "__main__":
    main()
