import time

from mixtera.core.client import MixteraClient
from mixtera.core.client.server import ServerStub
from mixtera.core.query import Query


def test_filter_javascript(client: ServerStub, chunk_size: int, tunnel: bool):
    training_id = str(round(time.time() * 1000))
    query = Query.for_training(training_id).select(("language", "==", "JavaScript"))
    assert client.execute_query(query, chunk_size)
    result_samples = []

    for sample in client.stream_results(training_id, tunnel_via_server=tunnel):
        result_samples.append(sample)

    assert len(result_samples) == 500, f"Got {len(result_samples)} samples instead of the expected 500!"
    for sample in result_samples:
        assert int(sample) % 2 == 0, f"Sample {sample} should not appear for JavaScript"


def test_filter_html(client: ServerStub, chunk_size: int, tunnel: bool):
    training_id = str(round(time.time() * 1000))
    query = Query.for_training(training_id).select(("language", "==", "HTML"))
    assert client.execute_query(query, chunk_size)
    result_samples = []

    for sample in client.stream_results(training_id, tunnel_via_server=tunnel):
        result_samples.append(sample)

    assert len(result_samples) == 500, f"Got {len(result_samples)} samples instead of the expected 500!"
    for sample in result_samples:
        assert int(sample) % 2 == 1, f"Sample {sample} should not appear for HTML"


def test_filter_both(client: ServerStub, chunk_size: int, tunnel: bool):
    training_id = str(round(time.time() * 1000))
    query = (
        Query.for_training(training_id)
        .select(("language", "==", "HTML"))
        .union(Query.for_training(training_id).select(("language", "==", "JavaScript")))
    )
    assert client.execute_query(query, chunk_size)
    result_samples = []

    for sample in client.stream_results(training_id, tunnel_via_server=tunnel):
        result_samples.append(sample)

    assert len(result_samples) == 1000, f"Got {len(result_samples)} samples instead of 1000!"
    for sample in result_samples:
        assert 0 <= int(sample) < 1000, f"Sample {sample} should not appear"


def test_filter_license(client: ServerStub, chunk_size: int, tunnel: bool):
    training_id = str(round(time.time() * 1000))
    query = Query.for_training(training_id).select(("license", "==", "CC"))
    assert client.execute_query(query, chunk_size)
    result_samples = []

    for sample in client.stream_results(training_id, tunnel_via_server=tunnel):
        result_samples.append(sample)

    assert len(result_samples) == 1000, f"Got {len(result_samples)} samples instead of the expected 1000!"
    for sample in result_samples:
        assert 0 <= int(sample) < 1000, f"Sample {sample} should not appear"


def test_filter_unknown_license(client: ServerStub, chunk_size: int, tunnel: bool):
    training_id = str(round(time.time() * 1000))
    query = Query.for_training(training_id).select(("license", "==", "All rights reserved."))
    assert client.execute_query(query, chunk_size)
    assert (
        len(list(client.stream_results(training_id, tunnel_via_server=tunnel))) == 0
    ), "Got results back for expected empty results."


def test_filter_license_and_html(client: ServerStub, chunk_size: int, tunnel: bool):
    # TODO(41): This test currently tests unexpected behavior - we want to deduplicate!
    training_id = str(round(time.time() * 1000))
    query = (
        Query.for_training(training_id)
        .select(("language", "==", "HTML"))
        .union(Query.for_training(training_id).select(("license", "==", "CC")))
    )
    assert client.execute_query(query, chunk_size)
    result_samples = []

    for sample in client.stream_results(training_id, tunnel_via_server=tunnel):
        result_samples.append(sample)

    assert len(result_samples) == 1500, f"Got {len(result_samples)} samples instead of the expected 1500!"
    for sample in result_samples:
        assert 0 <= int(sample) < 1000, f"Sample {sample} should not appear"


def test_rdc_chunksize_tunnel(client: ServerStub, chunk_size: int, tunnel: bool):
    test_filter_javascript(client, chunk_size, tunnel)
    test_filter_html(client, chunk_size, tunnel)
    test_filter_both(client, chunk_size, tunnel)
    test_filter_license(client, chunk_size, tunnel)
    test_filter_unknown_license(client, chunk_size, tunnel)
    test_filter_license_and_html(client, chunk_size, tunnel)


def test_server() -> None:
    client = MixteraClient.from_remote("127.0.0.1", 6666)
    for chunk_size in [1, 3, 250, 500, 750, 1000, 2000]:
        for tunnel in [False, True]:
            test_rdc_chunksize_tunnel(client, chunk_size, tunnel)

    print("Successfully ran server tests!")


def main() -> None:
    test_server()


if __name__ == "__main__":
    main()
