import tempfile
import time
from pathlib import Path

import torch
from integrationtests.utils import TestMetadataParser, write_jsonl
from mixtera.core.datacollection import MixteraDataCollection
from mixtera.core.datacollection.datasets import JSONLDataset
from mixtera.core.query import Query
from mixtera.torch import MixteraTorchDataset


def sample_parsing_func(sample):
    import json

    return json.loads(sample)["text"]


def test_filter_javascript(
    mdc: MixteraDataCollection, chunk_size: int, num_workers: int, batch_size: int, tunnel: bool
):
    training_id = str(round(time.time() * 1000))
    query = Query.for_training(training_id, 1).select(("language", "==", "JavaScript"))
    torch_ds = MixteraTorchDataset(mdc, query, training_id, chunk_size, tunnel_via_server=tunnel)
    dl = torch.utils.data.DataLoader(torch_ds, batch_size=batch_size, num_workers=num_workers)
    result_samples = []
    for batch in dl:
        result_samples.extend(batch)

    assert len(result_samples) == 500, f"Got {len(result_samples)} samples instead of the expected 500!"
    for sample in result_samples:
        assert int(sample) % 2 == 0, f"Sample {sample} should not appear for JavaScript"


def test_filter_html(mdc: MixteraDataCollection, chunk_size: int, num_workers: int, batch_size: int, tunnel: bool):
    training_id = str(round(time.time() * 1000))
    query = Query.for_training(training_id, 1).select(("language", "==", "HTML"))
    torch_ds = MixteraTorchDataset(mdc, query, training_id, chunk_size, tunnel_via_server=tunnel)
    dl = torch.utils.data.DataLoader(torch_ds, batch_size=batch_size, num_workers=num_workers)
    result_samples = []
    for batch in dl:
        result_samples.extend(batch)

    assert len(result_samples) == 500, f"Got {len(result_samples)} samples instead of the expected 500!"
    for sample in result_samples:
        assert int(sample) % 2 == 1, f"Sample {sample} should not appear for HTML"


def test_filter_both(mdc: MixteraDataCollection, chunk_size: int, num_workers: int, batch_size: int, tunnel: bool):
    training_id = str(round(time.time() * 1000))
    query = (
        Query.for_training(training_id, 1)
        .select(("language", "==", "HTML"))
        .union(Query.for_training(training_id, 1).select(("language", "==", "JavaScript")))
    )
    torch_ds = MixteraTorchDataset(mdc, query, training_id, chunk_size, tunnel_via_server=tunnel)
    dl = torch.utils.data.DataLoader(torch_ds, batch_size=batch_size, num_workers=num_workers)
    result_samples = []
    for batch in dl:
        result_samples.extend(batch)

    assert len(result_samples) == 1000, f"Got {len(result_samples)} samples instead of the expected 1000!"
    for sample in result_samples:
        assert 0 <= int(sample) < 1000, f"Sample {sample} should not appear"


def test_filter_license(mdc: MixteraDataCollection, chunk_size: int, num_workers: int, batch_size: int, tunnel: bool):
    training_id = str(round(time.time() * 1000))
    query = Query.for_training(training_id, 1).select(("license", "==", "CC"))
    torch_ds = MixteraTorchDataset(mdc, query, training_id, chunk_size, tunnel_via_server=tunnel)
    dl = torch.utils.data.DataLoader(torch_ds, batch_size=batch_size, num_workers=num_workers)
    result_samples = []
    for batch in dl:
        result_samples.extend(batch)

    assert len(result_samples) == 1000, f"Got {len(result_samples)} samples instead of the expected 1000!"
    for sample in result_samples:
        assert 0 <= int(sample) < 1000, f"Sample {sample} should not appear"


def test_filter_unknown_license(
    mdc: MixteraDataCollection, chunk_size: int, num_workers: int, batch_size: int, tunnel: bool
):
    training_id = str(round(time.time() * 1000))
    query = Query.for_training(training_id, 1).select(("license", "==", "All rights reserved."))
    torch_ds = MixteraTorchDataset(mdc, query, training_id, chunk_size, tunnel_via_server=tunnel)
    dl = torch.utils.data.DataLoader(torch_ds, batch_size=batch_size, num_workers=num_workers)
    result_samples = []
    for batch in dl:
        result_samples.extend(batch)
    assert len(result_samples) == 0, "Got results back for expected empty results."


def test_filter_license_and_html(
    mdc: MixteraDataCollection, chunk_size: int, num_workers: int, batch_size: int, tunnel: bool
):
    # TODO(41): This test currently tests unexpected behavior - we want to deduplicate!
    training_id = str(round(time.time() * 1000))
    query = (
        Query.for_training(training_id, 1)
        .select(("language", "==", "HTML"))
        .union(Query.for_training(training_id, 1).select(("license", "==", "CC")))
    )
    torch_ds = MixteraTorchDataset(mdc, query, training_id, chunk_size, tunnel_via_server=tunnel)
    dl = torch.utils.data.DataLoader(torch_ds, batch_size=batch_size, num_workers=num_workers)
    result_samples = []
    for batch in dl:
        result_samples.extend(batch)

    assert len(result_samples) == 1500, f"Got {len(result_samples)} samples instead of the expected 1500!"
    for sample in result_samples:
        assert 0 <= int(sample) < 1000, f"Sample {sample} should not appear"


def test_torchds(mdc: MixteraDataCollection, chunk_size: int, num_workers: int, batch_size: int, tunnel: bool):
    test_filter_javascript(mdc, chunk_size, num_workers, batch_size, tunnel)
    test_filter_html(mdc, chunk_size, num_workers, batch_size, tunnel)
    test_filter_both(mdc, chunk_size, num_workers, batch_size, tunnel)
    test_filter_license(mdc, chunk_size, num_workers, batch_size, tunnel)
    test_filter_unknown_license(mdc, chunk_size, num_workers, batch_size, tunnel)
    test_filter_license_and_html(mdc, chunk_size, num_workers, batch_size, tunnel)


def test_tds(dir: Path) -> None:
    write_jsonl(dir / "testd.jsonl")
    ldc = MixteraDataCollection.from_directory(dir)
    ldc._metadata_factory.add_parser("TEST_PARSER", TestMetadataParser)
    ldc.register_dataset(
        "ldc_integrationtest_dataset", dir / "testd.jsonl", JSONLDataset, sample_parsing_func, "TEST_PARSER"
    )

    rdc = MixteraDataCollection.from_remote("127.0.0.1", 6666)

    for mdc in [ldc, rdc]:
        for chunk_size in [1, 3, 250, 500, 750, 1000, 2000]:
            for num_workers in [0, 1, 3, 8]:
                for batch_size in [1, 2, 500]:
                    for tunnel in [False, True]:
                        if tunnel and not mdc.is_remote():
                            continue
                        try:
                            test_torchds(mdc, chunk_size, num_workers, batch_size, tunnel)
                        except Exception as e:
                            print(
                                f"Error with mdc.is_remote = {mdc.is_remote()},"
                                + f"chunk_size = {chunk_size}, num_workers = {num_workers},"
                                + f"batch_size = {batch_size}, tunnel = {tunnel}"
                            )
                            raise e

    print("Successfully ran MixteraTorchDataset test!")


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        test_tds(Path(directory))


if __name__ == "__main__":
    main()
