import tempfile
import time
from pathlib import Path
from typing import Any

from integrationtests.utils import TestMetadataParser, get_expected_js_and_html_samples, get_job_id, setup_test_dataset
from loguru import logger
from mixtera.core.client import ChunkReaderType, MixteraClient
from mixtera.core.datacollection.datasets import JSONLDataset
from mixtera.core.query import ArbitraryMixture, Mixture, Query

TEST_LOCAL_INSTANCE_COUNT = 1000
TEST_LOCAL_FILE_COUNT = 5
TEST_LOCAL_FRACTION_MULTIPLIER = 2

EXPECTED_JS_SAMPLES, EXPECTED_HTML_SAMPLES = get_expected_js_and_html_samples(
    TEST_LOCAL_INSTANCE_COUNT, TEST_LOCAL_FRACTION_MULTIPLIER
)


def parsing_func(sample):
    import json

    return json.loads(sample)["text"]


def test_filter_javascript(
    client: MixteraClient,
    mixture: Mixture,
    **args: Any,
) -> None:
    job_id = get_job_id()
    query = Query.for_job(job_id).select(("language", "==", "JavaScript"))
    client.execute_query(query, mixture)
    result_samples = []
    for sample in client.stream_results(job_id, False, **args):
        result_samples.append(sample)

    assert (
        len(result_samples) == EXPECTED_JS_SAMPLES
    ), f"Got {len(result_samples)} samples instead of the expected {EXPECTED_JS_SAMPLES}!"
    for sample in result_samples:
        assert int(sample) % TEST_LOCAL_FRACTION_MULTIPLIER == 0, f"Sample {sample} should not appear for JavaScript"


def test_filter_html(
    client: MixteraClient,
    mixture: Mixture,
    **args: Any,
):
    job_id = get_job_id()
    query = Query.for_job(job_id).select(("language", "==", "HTML"))
    client.execute_query(query, mixture)
    result_samples = []

    for sample in client.stream_results(job_id, False, **args):
        result_samples.append(sample)

    assert (
        len(result_samples) == EXPECTED_HTML_SAMPLES
    ), f"Got {len(result_samples)} samples instead of the expected {EXPECTED_HTML_SAMPLES}!"
    for sample in result_samples:
        assert int(sample) % TEST_LOCAL_FRACTION_MULTIPLIER == 1, f"Sample {sample} should not appear for HTML"


def test_filter_both(
    client: MixteraClient,
    mixture: Mixture,
    **args: Any,
):
    job_id = get_job_id()
    query = (
        Query.for_job(job_id)
        .select(("language", "==", "HTML"))
        .union(Query.for_job(job_id).select(("language", "==", "JavaScript")))
    )
    client.execute_query(query, mixture)
    result_samples = []

    for sample in client.stream_results(job_id, False, **args):
        result_samples.append(sample)

    assert (
        len(result_samples) == TEST_LOCAL_INSTANCE_COUNT
    ), f"Got {len(result_samples)} samples instead of the expected {TEST_LOCAL_INSTANCE_COUNT}!"
    for sample in result_samples:
        assert 0 <= int(sample) < TEST_LOCAL_INSTANCE_COUNT, f"Sample {sample} should not appear"


def test_filter_license(
    client: MixteraClient,
    mixture: Mixture,
    **args: Any,
):
    job_id = get_job_id()
    query = Query.for_job(job_id).select(("license", "==", "CC"))
    client.execute_query(query, mixture)
    result_samples = []

    for sample in client.stream_results(job_id, False, **args):
        result_samples.append(sample)

    assert (
        len(result_samples) == TEST_LOCAL_INSTANCE_COUNT
    ), f"Got {len(result_samples)} samples instead of the expected {TEST_LOCAL_INSTANCE_COUNT}!"
    for sample in result_samples:
        assert 0 <= int(sample) < TEST_LOCAL_INSTANCE_COUNT, f"Sample {sample} should not appear"


def test_filter_unknown_license(
    client: MixteraClient,
    mixture: Mixture,
    **args: Any,
):
    job_id = get_job_id()
    query = Query.for_job(job_id).select(("license", "==", "All rights reserved."))
    client.execute_query(query, mixture)
    assert len(list(client.stream_results(job_id, False, **args))) == 0, "Got results back for expected empty results."


def test_filter_license_and_html(
    client: MixteraClient,
    mixture: Mixture,
    **args: Any,
):
    job_id = get_job_id()
    query = (
        Query.for_job(job_id)
        .select(("language", "==", "HTML"))
        .union(Query.for_job(job_id).select(("license", "==", "CC")))
    )
    client.execute_query(query, mixture)
    result_samples = []

    for sample in client.stream_results(job_id, False, **args):
        result_samples.append(sample)

    assert (
        len(result_samples) == TEST_LOCAL_INSTANCE_COUNT
    ), f"Got {len(result_samples)} samples instead of the expected {TEST_LOCAL_INSTANCE_COUNT}!"
    for sample in result_samples:
        assert 0 <= int(sample) < TEST_LOCAL_INSTANCE_COUNT, f"Sample {sample} should not appear"


def test_client_chunksize(
    client: MixteraClient,
    mixture: Mixture,
    **args: Any,
):
    test_filter_javascript(client, mixture, **args)
    test_filter_html(client, mixture, **args)
    test_filter_both(client, mixture, **args)
    test_filter_license(client, mixture, **args)
    test_filter_unknown_license(client, mixture, **args)
    test_filter_license_and_html(client, mixture, **args)


def test_chunk_readers(dir: Path) -> None:
    setup_test_dataset(dir, TEST_LOCAL_INSTANCE_COUNT, TEST_LOCAL_FILE_COUNT, TEST_LOCAL_FRACTION_MULTIPLIER)
    client = MixteraClient.from_directory(dir)
    client.register_metadata_parser("TEST_PARSER", TestMetadataParser)
    client.register_dataset(
        "client_integrationtest_chunk_reader_dataset", dir, JSONLDataset, parsing_func, "TEST_PARSER"
    )

    degrees_of_parallelisms = [1, 2, 4]
    per_window_mixtures = [False, True]
    window_sizes = [64, 128, 256]

    for chunk_size in [100, 250, 500, 750, 1000]:
        for degree_of_parallelism in degrees_of_parallelisms:
            for per_window_mixture in per_window_mixtures:
                for window_size in window_sizes:
                    args = {
                        "degree_of_parallelism": degree_of_parallelism,
                        "per_window_mixture": per_window_mixture,
                        "window_size": window_size,
                    }
                    logger.debug(
                        f"Running chunk reader tests with chunk_size={chunk_size}, degree_of_parallelism={degree_of_parallelism}, "
                        f"per_window_mixture={per_window_mixture}, window_size={window_size}"
                    )
                    test_client_chunksize(client, ArbitraryMixture(chunk_size), **args)

    print("Successfully ran chunk reader tests!")

    client.remove_dataset("client_integrationtest_chunk_reader_dataset")


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        test_chunk_readers(Path(directory))


if __name__ == "__main__":
    main()
