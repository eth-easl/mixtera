import sys
import tempfile
from pathlib import Path

import torch
from integrationtests.utils import TestMetadataParser, setup_test_dataset
from mixtera.core.client import MixteraClient
from mixtera.core.datacollection.datasets import JSONLDataset
from mixtera.core.query import ArbitraryMixture, Mixture, Query
from mixtera.torch import MixteraTorchDataset


def sample_parsing_func(sample):
    import json

    return json.loads(sample)["text"]


def test_filter_javascript(
    client: MixteraClient,
    mixture: Mixture,
    dp_groups: int,
    nodes_per_group: int,
    num_workers: int,
    batch_size: int,
    tunnel: bool,
):
    job_id = f"0_{mixture.chunk_size}_{batch_size}_{dp_groups}_{nodes_per_group}_{num_workers}_{tunnel}"
    query = Query.for_job(job_id).select(("language", "==", "JavaScript"))
    torch_ds = MixteraTorchDataset(
        client,
        query,
        job_id,
        mixture,
        dp_groups=dp_groups,
        nodes_per_group=nodes_per_group,
        node_id=0,
        dp_group_id=0,
        tunnel_via_server=tunnel,
    )
    dl = torch.utils.data.DataLoader(torch_ds, batch_size=batch_size, num_workers=num_workers)
    result_samples = []
    for batch in dl:
        result_samples.extend(batch)

    assert len(result_samples) == 500, f"Got {len(result_samples)} samples instead of the expected 500!"
    for sample in result_samples:
        assert int(sample) % 2 == 0, f"Sample {sample} should not appear for JavaScript"


def test_filter_html(
    client: MixteraClient,
    mixture: Mixture,
    dp_groups: int,
    nodes_per_group: int,
    num_workers: int,
    batch_size: int,
    tunnel: bool,
):
    job_id = f"1_{mixture.chunk_size}_{batch_size}_{dp_groups}_{nodes_per_group}_{num_workers}_{tunnel}"
    query = Query.for_job(job_id).select(("language", "==", "HTML"))
    torch_ds = MixteraTorchDataset(client, query, job_id, mixture, tunnel_via_server=tunnel)
    dl = torch.utils.data.DataLoader(torch_ds, batch_size=batch_size, num_workers=num_workers)
    result_samples = []
    for batch in dl:
        result_samples.extend(batch)

    assert len(result_samples) == 500, f"Got {len(result_samples)} samples instead of the expected 500!"
    for sample in result_samples:
        assert int(sample) % 2 == 1, f"Sample {sample} should not appear for HTML"


def test_filter_both(
    client: MixteraClient,
    mixture: Mixture,
    dp_groups: int,
    nodes_per_group: int,
    num_workers: int,
    batch_size: int,
    tunnel: bool,
):
    job_id = f"2_{mixture.chunk_size}_{batch_size}_{dp_groups}_{nodes_per_group}_{num_workers}_{tunnel}"
    query = (
        Query.for_job(job_id)
        .select(("language", "==", "HTML"))
        .union(Query.for_job(job_id).select(("language", "==", "JavaScript")))
    )
    torch_ds = MixteraTorchDataset(client, query, job_id, mixture, tunnel_via_server=tunnel)
    dl = torch.utils.data.DataLoader(torch_ds, batch_size=batch_size, num_workers=num_workers)
    result_samples = []
    for batch in dl:
        result_samples.extend(batch)

    assert len(result_samples) == 1000, f"Got {len(result_samples)} samples instead of the expected 1000!"
    for sample in result_samples:
        assert 0 <= int(sample) < 1000, f"Sample {sample} should not appear"


def test_filter_license(
    client: MixteraClient,
    mixture: Mixture,
    dp_groups: int,
    nodes_per_group: int,
    num_workers: int,
    batch_size: int,
    tunnel: bool,
):
    job_id = f"3_{mixture.chunk_size}_{batch_size}_{dp_groups}_{nodes_per_group}_{num_workers}_{tunnel}"
    query = Query.for_job(job_id).select(("license", "==", "CC"))
    torch_ds = MixteraTorchDataset(client, query, job_id, mixture, tunnel_via_server=tunnel)
    dl = torch.utils.data.DataLoader(torch_ds, batch_size=batch_size, num_workers=num_workers)
    result_samples = []
    for batch in dl:
        result_samples.extend(batch)

    assert len(result_samples) == 1000, f"Got {len(result_samples)} samples instead of the expected 1000!"
    for sample in result_samples:
        assert 0 <= int(sample) < 1000, f"Sample {sample} should not appear"


def test_filter_unknown_license(
    client: MixteraClient,
    mixture: Mixture,
    dp_groups: int,
    nodes_per_group: int,
    num_workers: int,
    batch_size: int,
    tunnel: bool,
):
    job_id = f"4_{mixture.chunk_size}_{batch_size}_{dp_groups}_{nodes_per_group}_{num_workers}_{tunnel}"
    query = Query.for_job(job_id).select(("license", "==", "All rights reserved."))
    torch_ds = MixteraTorchDataset(client, query, job_id, mixture, tunnel_via_server=tunnel)
    dl = torch.utils.data.DataLoader(torch_ds, batch_size=batch_size, num_workers=num_workers)
    result_samples = []
    for batch in dl:
        result_samples.extend(batch)
    assert len(result_samples) == 0, "Got results back for expected empty results."


def test_filter_license_and_html(
    client: MixteraClient,
    mixture: Mixture,
    dp_groups: int,
    nodes_per_group: int,
    num_workers: int,
    batch_size: int,
    tunnel: bool,
):
    job_id = f"5_{mixture.chunk_size}_{batch_size}_{dp_groups}_{nodes_per_group}_{num_workers}_{tunnel}"
    query = (
        Query.for_job(job_id)
        .select(("language", "==", "HTML"))
        .union(Query.for_job(job_id).select(("license", "==", "CC")))
    )
    torch_ds = MixteraTorchDataset(client, query, job_id, mixture, tunnel_via_server=tunnel)
    dl = torch.utils.data.DataLoader(torch_ds, batch_size=batch_size, num_workers=num_workers)
    result_samples = []
    for batch in dl:
        result_samples.extend(batch)

    assert len(result_samples) == 1000, f"Got {len(result_samples)} samples instead of the expected 1000!"
    for sample in result_samples:
        assert 0 <= int(sample) < 1000, f"Sample {sample} should not appear"


def test_torchds(
    client: MixteraClient,
    mixture: Mixture,
    dp_groups: int,
    nodes_per_group: int,
    num_workers: int,
    batch_size: int,
    tunnel: bool,
):
    test_filter_javascript(client, mixture, dp_groups, nodes_per_group, num_workers, batch_size, tunnel)
    test_filter_html(client, mixture, dp_groups, nodes_per_group, num_workers, batch_size, tunnel)
    test_filter_both(client, mixture, dp_groups, nodes_per_group, num_workers, batch_size, tunnel)
    test_filter_license(client, mixture, dp_groups, nodes_per_group, num_workers, batch_size, tunnel)
    test_filter_unknown_license(client, mixture, dp_groups, nodes_per_group, num_workers, batch_size, tunnel)
    test_filter_license_and_html(client, mixture, dp_groups, nodes_per_group, num_workers, batch_size, tunnel)


def test_tds(local_dir: Path, server_dir: Path) -> None:
    local_file = setup_test_dataset(local_dir)

    # local tests
    local_client = MixteraClient(local_dir)
    local_client.register_metadata_parser("TEST_PARSER", TestMetadataParser)
    local_client.register_dataset(
        "ldc_torch_integrationtest_dataset", local_file, JSONLDataset, sample_parsing_func, "TEST_PARSER"
    )

    for mixture in [ArbitraryMixture(x) for x in [1, 3, 500, 750, 2000]]:
        for num_workers in [0, 3, 8]:
            for batch_size in [1, 2, 500]:
                try:
                    test_torchds(local_client, mixture, 1, 1, num_workers, batch_size, False)
                except Exception as e:
                    print(
                        "Error with "
                        + f"chunk_size = {mixture.chunk_size}, num_workers = {num_workers},"
                        + f"batch_size = {batch_size}"
                    )
                    raise e

    # server tests (smaller matrix)
    server_file = setup_test_dataset(server_dir)
    server_client = MixteraClient("127.0.0.1", 6666)

    assert server_client.register_metadata_parser("TEST_PARSER_TORCH", TestMetadataParser)
    assert server_client.register_dataset(
        "ldc_torch_integrationtest_dataset", server_file, JSONLDataset, sample_parsing_func, "TEST_PARSER_TORCH"
    )

    assert server_client.check_dataset_exists("ldc_torch_integrationtest_dataset"), "Dataset does not exist!"

    for mixture in [ArbitraryMixture(x) for x in [1, 2000]]:
        for num_workers in [0, 8]:
            for batch_size in [1, 500]:
                for tunnel in [False, True]:
                    if tunnel and (batch_size > 1 or num_workers > 0 or mixture.chunk_size > 1):
                        continue
                    try:
                        test_torchds(server_client, mixture, 1, 1, num_workers, batch_size, tunnel)
                    except Exception as e:
                        print(
                            "Error with "
                            + f"chunk_size = {mixture.chunk_size}, num_workers = {num_workers},"
                            + f"batch_size = {batch_size}, tunnel = {tunnel}"
                        )
                        raise e

    print("Successfully ran MixteraTorchDataset test!")


def main() -> None:
    server_dir = Path(sys.argv[1])

    with tempfile.TemporaryDirectory() as directory:
        test_tds(Path(directory), server_dir)


if __name__ == "__main__":
    main()
