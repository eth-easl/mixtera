import sys
import tempfile
from copy import deepcopy
from pathlib import Path

import torch
from integrationtests.utils import REPRODUCIBILITY_ITERATIONS, TestMetadataParser, setup_test_dataset
from loguru import logger
from mixtera.core.client import MixteraClient
from mixtera.core.client.mixtera_client import QueryExecutionArgs, ResultStreamingArgs
from mixtera.core.datacollection.datasets import JSONLDataset
from mixtera.core.query import Query
from mixtera.core.query.mixture import ArbitraryMixture, InferringMixture
from mixtera.torch import MixteraTorchDataset

TEST_PYTORCH_INSTANCE_COUNT = 1000
TEST_PYTORCH_FILE_COUNT = 5
TEST_PYTORCH_FRACTION_MULTIPLIER = 2


def sample_parsing_func(sample):
    import json

    return json.loads(sample)["text"]


def create_and_iterate_dataloaders(
    client, query, query_execution_args: QueryExecutionArgs, result_streaming_args: ResultStreamingArgs, batch_size
):
    # This iterates in a round robin fashion across the dp nodes and groups:
    # It first fetches a batch from group 0 / node 0, then group 0 / node 1, etc
    # Then group 1 / node 0.
    # This is to somewhat simulate the behavior of nodes requesting batches in parallel.
    # Otherwise, if we were to iterate over all batches of group 0 / node 0 first,
    # then we'd completely cache all batches, and all other data parallel groups won't
    # have any chunks left.
    result_samples = []
    data_loaders = []
    for node_id in range(query_execution_args.nodes_per_group):
        for dp_group_id in range(query_execution_args.dp_groups):
            node_args = deepcopy(result_streaming_args)
            node_args.node_id = node_id
            node_args.dp_group_id = dp_group_id
            torch_ds = MixteraTorchDataset(client, query, query_execution_args, node_args)
            dl = torch.utils.data.DataLoader(
                torch_ds, batch_size=batch_size, num_workers=query_execution_args.num_workers
            )
            data_loaders.append(dl)

    iterators = [iter(dl) for dl in data_loaders]
    active = True
    while active:
        active = False
        for it in iterators:
            try:
                batch = next(it)
                result_samples.extend(batch)
                active = True  # As long as one iterator returns a batch, keep looping
            except StopIteration:
                continue  # This iterator is exhausted, move to the next

    return result_samples


def test_filter_javascript(
    client: MixteraClient, query_exec_args: QueryExecutionArgs, batch_size: int, tunnel: bool, mixture_str: str
):
    job_id = (
        f"0_{query_exec_args.mixture.chunk_size}_{batch_size}_{query_exec_args.dp_groups}"
        + f"_{query_exec_args.nodes_per_group}_{query_exec_args.num_workers}_{tunnel}_{mixture_str}"
    )
    query = Query.for_job(job_id).select(("language", "==", "JavaScript"))
    result_samples = create_and_iterate_dataloaders(
        client, query, query_exec_args, ResultStreamingArgs(job_id=job_id, tunnel_via_server=tunnel), batch_size
    )
    expected_samples = 500 * query_exec_args.nodes_per_group

    assert (
        len(result_samples) == expected_samples
    ), f"Got {len(result_samples)} samples instead of the expected 500 * {query_exec_args.nodes_per_group} = {expected_samples}!"
    for sample in result_samples:
        assert int(sample) % 2 == 0, f"Sample {sample} should not appear for JavaScript"


def test_filter_html(
    client: MixteraClient, query_exec_args: QueryExecutionArgs, batch_size: int, tunnel: bool, mixture_str: str
):
    job_id = (
        f"1_{query_exec_args.mixture.chunk_size}_{batch_size}_{query_exec_args.dp_groups}"
        + f"_{query_exec_args.nodes_per_group}_{query_exec_args.num_workers}_{tunnel}_{mixture_str}"
    )
    query = Query.for_job(job_id).select(("language", "==", "HTML"))
    result_samples = create_and_iterate_dataloaders(
        client, query, query_exec_args, ResultStreamingArgs(job_id=job_id, tunnel_via_server=tunnel), batch_size
    )
    expected_samples = 500 * query_exec_args.nodes_per_group

    assert (
        len(result_samples) == expected_samples
    ), f"Got {len(result_samples)} samples instead of the expected 500 * {query_exec_args.nodes_per_group} = {expected_samples}!"
    for sample in result_samples:
        assert int(sample) % 2 == 1, f"Sample {sample} should not appear for HTML"


def test_filter_both(
    client: MixteraClient, query_exec_args: QueryExecutionArgs, batch_size: int, tunnel: bool, mixture_str: str
):
    job_id = (
        f"2_{query_exec_args.mixture.chunk_size}_{batch_size}_{query_exec_args.dp_groups}"
        + f"_{query_exec_args.nodes_per_group}_{query_exec_args.num_workers}_{tunnel}_{mixture_str}"
    )
    query = Query.for_job(job_id).select(("language", "==", "HTML")).select(("language", "==", "JavaScript"))
    result_samples = create_and_iterate_dataloaders(
        client, query, query_exec_args, ResultStreamingArgs(job_id=job_id, tunnel_via_server=tunnel), batch_size
    )
    expected_samples = 1000 * query_exec_args.nodes_per_group

    assert (
        len(result_samples) == expected_samples
    ), f"Got {len(result_samples)} samples instead of the expected 1000 * {query_exec_args.nodes_per_group} = {expected_samples}!"
    for sample in result_samples:
        assert 0 <= int(sample) < 1000, f"Sample {sample} should not appear"


def test_filter_license(
    client: MixteraClient, query_exec_args: QueryExecutionArgs, batch_size: int, tunnel: bool, mixture_str: str
):
    job_id = (
        f"3_{query_exec_args.mixture.chunk_size}_{batch_size}_{query_exec_args.dp_groups}"
        + f"_{query_exec_args.nodes_per_group}_{query_exec_args.num_workers}_{tunnel}_{mixture_str}"
    )
    query = Query.for_job(job_id).select(("license", "==", "CC"))
    result_samples = create_and_iterate_dataloaders(
        client, query, query_exec_args, ResultStreamingArgs(job_id=job_id, tunnel_via_server=tunnel), batch_size
    )

    expected_samples = 500 * query_exec_args.nodes_per_group

    assert (
        len(result_samples) == expected_samples
    ), f"Got {len(result_samples)} samples instead of the expected 500 * {query_exec_args.nodes_per_group} = {expected_samples}!"
    for sample in result_samples:
        assert 0 <= int(sample) < 1000, f"Sample {sample} should not appear"


def test_filter_unknown_license(
    client: MixteraClient, query_exec_args: QueryExecutionArgs, batch_size: int, tunnel: bool, mixture_str: str
):
    job_id = (
        f"4_{query_exec_args.mixture.chunk_size}_{batch_size}_{query_exec_args.dp_groups}"
        + f"_{query_exec_args.nodes_per_group}_{query_exec_args.num_workers}_{tunnel}_{mixture_str}"
    )
    query = Query.for_job(job_id).select(("license", "==", "All rights reserved."))
    result_samples = create_and_iterate_dataloaders(
        client, query, query_exec_args, ResultStreamingArgs(job_id=job_id, tunnel_via_server=tunnel), batch_size
    )

    assert len(result_samples) == 0, f"Got {len(result_samples)} samples for expected empty results"


def test_filter_license_and_html(
    client: MixteraClient, query_exec_args: QueryExecutionArgs, batch_size: int, tunnel: bool, mixture_str: str
):
    job_id = (
        f"5_{query_exec_args.mixture.chunk_size}_{batch_size}_{query_exec_args.dp_groups}"
        + f"_{query_exec_args.nodes_per_group}_{query_exec_args.num_workers}_{tunnel}_{mixture_str}"
    )
    query = Query.for_job(job_id).select(("language", "==", "HTML")).select(("license", "==", "CC"))
    result_samples = create_and_iterate_dataloaders(
        client, query, query_exec_args, ResultStreamingArgs(job_id=job_id, tunnel_via_server=tunnel), batch_size
    )
    expected_samples = 750 * query_exec_args.nodes_per_group

    assert (
        len(result_samples) == expected_samples
    ), f"Got {len(result_samples)} samples instead of the expected 750 * {query_exec_args.nodes_per_group} = {expected_samples}!"
    for sample in result_samples:
        assert 0 <= int(sample) < 750, f"Sample {sample} should not appear"


def test_filter_both_with_order_validation(
    client: MixteraClient, query_exec_args: QueryExecutionArgs, batch_size: int, tunnel: bool, mixture_str: str
):
    if query_exec_args.nodes_per_group <= 1:
        return  # This test is only relevant if there are multiple nodes per group

    job_id = (
        f"6_{query_exec_args.mixture.chunk_size}_{batch_size}_{query_exec_args.dp_groups}"
        + f"_{query_exec_args.nodes_per_group}_{query_exec_args.num_workers}_{tunnel}_{mixture_str}"
    )
    query = Query.for_job(job_id).select(("language", "==", "HTML")).select(("language", "==", "JavaScript"))

    # Collect batches for each node in each group
    group_batches = {}
    for dp_group_id in range(query_exec_args.dp_groups):
        node_batches = {}
        for node_id in range(query_exec_args.nodes_per_group):
            torch_ds = MixteraTorchDataset(
                client,
                query,
                query_exec_args,
                ResultStreamingArgs(job_id=job_id, dp_group_id=dp_group_id, node_id=node_id, tunnel_via_server=tunnel),
            )
            dl = torch.utils.data.DataLoader(torch_ds, batch_size=batch_size, num_workers=query_exec_args.num_workers)
            batches = list(dl)
            node_batches[node_id] = batches

        group_batches[dp_group_id] = node_batches

    for group_id, node_batches in group_batches.items():
        reference_batches = node_batches[0]  # Use the first node's batches as the reference
        for node_id, batches in node_batches.items():
            if node_id == 0:
                continue  # Skip the reference node itself
            assert batches == reference_batches, f"Mismatch in batch order for group {group_id}, node {node_id}"


def test_reader_reproducibility(
    client: MixteraClient, query_exec_args: QueryExecutionArgs, batch_size: int, tunnel: bool, mixture_str: str
):
    if (
        (query_exec_args.dp_groups > 1)
        or (query_exec_args.nodes_per_group > 1)
        or query_exec_args.num_workers > 3
        or (batch_size not in [1, 500])
        or (query_exec_args.mixture.chunk_size > 750)
    ):
        return

    reader_degrees_of_parallelisms = [1, 4]
    per_window_mixtures = [False, True]
    window_sizes = [64, 256]

    for reader_degree_of_parallelism in reader_degrees_of_parallelisms:
        for per_window_mixture in per_window_mixtures:
            for window_size in window_sizes:
                if reader_degree_of_parallelism > 1 and (query_exec_args.mixture.chunk_size < 500):
                    continue

                result_list = []
                logger.info("Running iterations.")
                for i in range(REPRODUCIBILITY_ITERATIONS):
                    group_batches = {}
                    for dp_group_id in range(query_exec_args.dp_groups):
                        node_batches = {}
                        for node_id in range(query_exec_args.nodes_per_group):
                            job_id = (
                                f"7_{query_exec_args.mixture.chunk_size}_{batch_size}_{query_exec_args.dp_groups}"
                                + f"_{query_exec_args.nodes_per_group}_{query_exec_args.num_workers}_{tunnel}_{reader_degree_of_parallelism}"
                                + f"_{per_window_mixture}_{window_size}_{i}_{dp_group_id}_{node_id}_{mixture_str}"
                            )
                            query = (
                                Query.for_job(job_id)
                                .select(("language", "==", "HTML"))
                                .select(("language", "==", "JavaScript"))
                            )
                            torch_ds = MixteraTorchDataset(
                                client,
                                query,
                                query_exec_args,
                                ResultStreamingArgs(
                                    job_id=job_id,
                                    dp_group_id=dp_group_id,
                                    node_id=node_id,
                                    tunnel_via_server=tunnel,
                                    chunk_reading_degree_of_parallelism=reader_degree_of_parallelism,
                                    chunk_reading_mixture_type=per_window_mixture,
                                    chunk_reading_window_size=window_size,
                                ),
                            )
                            dl = torch.utils.data.DataLoader(
                                torch_ds, batch_size=batch_size, num_workers=query_exec_args.num_workers
                            )
                            batches = list(dl)
                            node_batches[node_id] = batches
                        group_batches[dp_group_id] = node_batches
                    result_list.append(group_batches)

                logger.info("Iterations done, running comparisons.")
                reference_batches = result_list[0][0][0]  # Use the first node's batches as the reference

                for i in range(1, REPRODUCIBILITY_ITERATIONS):
                    for dp_group_id, node_batches in result_list[i].items():
                        if dp_group_id == 0:
                            continue  #  Skip the reference dp group itself
                        for node_id, batches in node_batches.items():
                            if node_id == 0:
                                continue  # Skip the reference node itself
                            assert (
                                batches == reference_batches
                            ), f"Mismatch in batch order for group {dp_group_id}, node {node_id}"

                logger.info("Comparisons done.")


def test_torchds(
    client: MixteraClient,
    query_exec_args: QueryExecutionArgs,
    batch_size: int,
    tunnel: bool,
):
    mixture_str = "arbitrary" if isinstance(query_exec_args.mixture, ArbitraryMixture) else "inferring"
    test_filter_javascript(client, query_exec_args, batch_size, tunnel, mixture_str)
    test_filter_html(client, query_exec_args, batch_size, tunnel, mixture_str)
    test_filter_both(client, query_exec_args, batch_size, tunnel, mixture_str)
    test_filter_license(client, query_exec_args, batch_size, tunnel, mixture_str)
    test_filter_unknown_license(client, query_exec_args, batch_size, tunnel, mixture_str)
    test_filter_license_and_html(client, query_exec_args, batch_size, tunnel, mixture_str)
    test_filter_both_with_order_validation(client, query_exec_args, batch_size, tunnel, mixture_str)
    test_reader_reproducibility(client, query_exec_args, batch_size, tunnel, mixture_str)


def test_tds(local_dir: Path, server_dir: Path) -> None:
    setup_test_dataset(
        local_dir,
        total_instance_count=TEST_PYTORCH_INSTANCE_COUNT,
        file_count=TEST_PYTORCH_FILE_COUNT,
        fraction_multiplier=TEST_PYTORCH_FRACTION_MULTIPLIER,
    )

    # local tests
    local_client = MixteraClient.from_directory(local_dir)
    local_client.register_metadata_parser("TEST_PARSER", TestMetadataParser)
    local_client.register_dataset(
        "ldc_torch_integrationtest_dataset", local_dir, JSONLDataset, sample_parsing_func, "TEST_PARSER"
    )

    # TODO(#111): InferringMixture currently fails the test because we do not support best effort mixture.
    # Without best effort mixture, the last chunk cannot be generated due to rounding issues
    # (the first key needs more items to account for rounding issues, e.g., if we have chunk size 250,
    #  we cannot have 4 properties with equal weight because 250 % 4 != 0)
    # :   + [InferringMixture(x) for x in [2, 500]]
    for mixture in [ArbitraryMixture(x) for x in [1, 3, 500, 750, 2000]]:
        for num_workers in [0, 3, 8]:
            for batch_size in [1, 2, 500]:
                try:
                    query_exec_args = QueryExecutionArgs(mixture=mixture, num_workers=num_workers)
                    test_torchds(local_client, query_exec_args, batch_size, False)
                except Exception as e:
                    print(
                        "Error with "
                        + f"chunk_size = {mixture.chunk_size}, num_workers = {num_workers},"
                        + f"batch_size = {batch_size}"
                    )
                    raise e

    # server tests (smaller matrix)
    setup_test_dataset(
        server_dir,
        total_instance_count=TEST_PYTORCH_INSTANCE_COUNT,
        file_count=TEST_PYTORCH_FILE_COUNT,
        fraction_multiplier=TEST_PYTORCH_FRACTION_MULTIPLIER,
    )
    server_client = MixteraClient("127.0.0.1", 6666)

    assert server_client.register_metadata_parser("TEST_PARSER_TORCH", TestMetadataParser)
    assert server_client.register_dataset(
        "ldc_torch_integrationtest_dataset", server_dir, JSONLDataset, sample_parsing_func, "TEST_PARSER_TORCH"
    )

    assert server_client.check_dataset_exists("ldc_torch_integrationtest_dataset"), "Dataset does not exist!"

    # TODO(#111): See above.
    # :    + [InferringMixture(2)]
    for mixture in [ArbitraryMixture(x) for x in [1, 2000]]:
        for dp_groups, num_nodes_per_group in [(1, 1), (1, 2), (2, 1), (2, 2), (4, 4)]:
            for num_workers in [0, 3, 8]:
                for batch_size in [1, 500]:
                    for tunnel in [False, True]:
                        if tunnel and (batch_size > 1 or num_workers > 0 or mixture.chunk_size > 1):
                            continue

                        if dp_groups > 2 and num_workers > 2:
                            continue  # we cant spawn num workers * dp groups * num_nodes processes that's too much

                        if dp_groups <= 2 and num_workers == 2:
                            continue  # just test 8 workers for the smaller number of nodes

                        try:
                            query_exec_args = QueryExecutionArgs(
                                mixture=mixture,
                                dp_groups=dp_groups,
                                nodes_per_group=num_nodes_per_group,
                                num_workers=num_workers,
                            )
                            test_torchds(server_client, query_exec_args, batch_size, tunnel)
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
