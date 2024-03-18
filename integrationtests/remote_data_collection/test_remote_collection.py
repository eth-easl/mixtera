import time

from mixtera.core.client.server import ServerStub
from mixtera.core.datacollection import MixteraClient
from mixtera.core.query import Query


def test_filter_javascript(rdc: ServerStub, chunk_size: int, tunnel: bool):
    training_id = str(round(time.time() * 1000))
    query = Query.for_training(training_id, 1).select(("language", "==", "JavaScript"))
    query.execute(rdc, chunk_size=chunk_size)
    query_result = rdc.get_query_result(training_id)
    result_samples = []

    for sample in rdc.stream_query_results(query_result, tunnel_via_server=tunnel):
        result_samples.append(sample)

    assert len(result_samples) == 500, f"Got {len(result_samples)} samples instead of the expected 500!"
    for sample in result_samples:
        assert int(sample) % 2 == 0, f"Sample {sample} should not appear for JavaScript"


def test_filter_html(rdc: ServerStub, chunk_size: int, tunnel: bool):
    training_id = str(round(time.time() * 1000))
    query = Query.for_training(training_id, 1).select(("language", "==", "HTML"))
    query_result = query.execute(rdc, chunk_size=chunk_size)
    result_samples = []

    for sample in rdc.stream_query_results(query_result, tunnel_via_server=tunnel):
        result_samples.append(sample)

    assert len(result_samples) == 500, f"Got {len(result_samples)} samples instead of the expected 500!"
    for sample in result_samples:
        assert int(sample) % 2 == 1, f"Sample {sample} should not appear for HTML"


def test_filter_both(rdc: ServerStub, chunk_size: int, tunnel: bool):
    training_id = str(round(time.time() * 1000))
    query = (
        Query.for_training(training_id, 1)
        .select(("language", "==", "HTML"))
        .union(Query.for_training(training_id, 1).select(("language", "==", "JavaScript")))
    )
    query_result = query.execute(rdc, chunk_size=chunk_size)
    result_samples = []

    for sample in rdc.stream_query_results(query_result, tunnel_via_server=tunnel):
        result_samples.append(sample)

    assert len(result_samples) == 1000, f"Got {len(result_samples)} samples instead of 1000!"
    for sample in result_samples:
        assert 0 <= int(sample) < 1000, f"Sample {sample} should not appear"


def test_filter_license(rdc: ServerStub, chunk_size: int, tunnel: bool):
    training_id = str(round(time.time() * 1000))
    query = Query.for_training(training_id, 1).select(("license", "==", "CC"))
    query_result = query.execute(rdc, chunk_size=chunk_size)
    result_samples = []

    for sample in rdc.stream_query_results(query_result, tunnel_via_server=tunnel):
        result_samples.append(sample)

    assert len(result_samples) == 1000, f"Got {len(result_samples)} samples instead of the expected 1000!"
    for sample in result_samples:
        assert 0 <= int(sample) < 1000, f"Sample {sample} should not appear"


def test_filter_unknown_license(rdc: ServerStub, chunk_size: int, tunnel: bool):
    training_id = str(round(time.time() * 1000))
    query = Query.for_training(training_id, 1).select(("license", "==", "All rights reserved."))
    query_result = query.execute(rdc, chunk_size=chunk_size)
    assert (
        len(list(rdc.stream_query_results(query_result, tunnel_via_server=tunnel))) == 0
    ), "Got results back for expected empty results."


def test_filter_license_and_html(rdc: ServerStub, chunk_size: int, tunnel: bool):
    # TODO(41): This test currently tests unexpected behavior - we want to deduplicate!
    training_id = str(round(time.time() * 1000))
    query = (
        Query.for_training(training_id, 1)
        .select(("language", "==", "HTML"))
        .union(Query.for_training(training_id, 1).select(("license", "==", "CC")))
    )
    query_result = query.execute(rdc, chunk_size=chunk_size)
    result_samples = []

    for sample in rdc.stream_query_results(query_result, tunnel_via_server=tunnel):
        result_samples.append(sample)

    assert len(result_samples) == 1500, f"Got {len(result_samples)} samples instead of the expected 1500!"
    for sample in result_samples:
        assert 0 <= int(sample) < 1000, f"Sample {sample} should not appear"


def test_rdc_chunksize_tunnel(rdc: ServerStub, chunk_size: int, tunnel: bool):
    test_filter_javascript(rdc, chunk_size, tunnel)
    test_filter_html(rdc, chunk_size, tunnel)
    test_filter_both(rdc, chunk_size, tunnel)
    test_filter_license(rdc, chunk_size, tunnel)
    test_filter_unknown_license(rdc, chunk_size, tunnel)
    test_filter_license_and_html(rdc, chunk_size, tunnel)


def test_rdc() -> None:
    rdc = MixteraClient.from_remote("127.0.0.1", 6666)
    for chunk_size in [1, 3, 250, 500, 750, 1000, 2000]:
        for tunnel in [False, True]:
            test_rdc_chunksize_tunnel(rdc, chunk_size, tunnel)

    print("Successfully ran RDC test!")


def main() -> None:
    test_rdc()


if __name__ == "__main__":
    main()
