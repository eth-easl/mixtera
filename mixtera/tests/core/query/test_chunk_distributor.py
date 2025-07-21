import pytest

from mixtera.core.query.chunk_distributor import ChunkDistributor


class MockQueryResult:
    def __init__(self):
        self.stop_on_none = False
        self.generator = self.result_generator()

    def result_generator(self):
        yield from range(100)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.generator)


@pytest.fixture(name="query_result")
def fixture_query_result():
    return MockQueryResult()


@pytest.fixture(name="chunk_distributor")
def fixture_chunk_distributor(query_result):
    return ChunkDistributor(dp_groups=2, nodes_per_group=2, num_workers=3, query_result=query_result, job_id="test_job")


def test_initialization(chunk_distributor):
    assert len(chunk_distributor._chunk_cache) == 2


def test_chunk_distribution_within_dp(chunk_distributor):
    chunk_0_0_0 = chunk_distributor.next_chunk_for(0, 0, 0, True)
    chunk_0_0_1 = chunk_distributor.next_chunk_for(0, 0, 1, True)
    chunk_0_0_2 = chunk_distributor.next_chunk_for(0, 0, 2, True)
    assert chunk_0_0_0 == 0
    assert chunk_0_0_1 == 1
    assert chunk_0_0_2 == 2

    chunk_0_1_0 = chunk_distributor.next_chunk_for(0, 1, 0, True)
    chunk_0_1_1 = chunk_distributor.next_chunk_for(0, 1, 1, True)
    chunk_0_1_2 = chunk_distributor.next_chunk_for(0, 1, 2, True)
    assert chunk_0_1_0 == 0
    assert chunk_0_1_1 == 1
    assert chunk_0_1_2 == 2

    chunk_0_0_0 = chunk_distributor.next_chunk_for(0, 0, 0, True)
    chunk_0_0_1 = chunk_distributor.next_chunk_for(0, 0, 1, True)
    assert chunk_0_0_0 == 3
    assert chunk_0_0_1 == 4

    chunk_0_1_0 = chunk_distributor.next_chunk_for(0, 1, 0, True)
    chunk_0_1_2 = chunk_distributor.next_chunk_for(0, 1, 2, True)
    assert chunk_0_1_0 == 3
    assert chunk_0_1_2 == 5

    chunk_0_0_2 = chunk_distributor.next_chunk_for(0, 0, 2, True)
    assert chunk_0_0_2 == 5


def test_chunk_distribution_across_dps(chunk_distributor):
    chunk_0_0_0 = chunk_distributor.next_chunk_for(0, 0, 0, True)
    chunk_0_0_1 = chunk_distributor.next_chunk_for(0, 0, 1, True)
    chunk_0_0_2 = chunk_distributor.next_chunk_for(0, 0, 2, True)
    assert chunk_0_0_0 == 0
    assert chunk_0_0_1 == 1
    assert chunk_0_0_2 == 2

    chunk_1_0_0 = chunk_distributor.next_chunk_for(1, 0, 0, True)
    chunk_1_0_1 = chunk_distributor.next_chunk_for(1, 0, 1, True)
    chunk_1_0_2 = chunk_distributor.next_chunk_for(1, 0, 2, True)
    assert chunk_1_0_0 == 3
    assert chunk_1_0_1 == 4
    assert chunk_1_0_2 == 5

    chunk_0_1_0 = chunk_distributor.next_chunk_for(0, 1, 0, True)
    chunk_0_1_1 = chunk_distributor.next_chunk_for(0, 1, 1, True)
    chunk_0_1_2 = chunk_distributor.next_chunk_for(0, 1, 2, True)
    assert chunk_0_1_0 == 0
    assert chunk_0_1_1 == 1
    assert chunk_0_1_2 == 2

    chunk_1_1_0 = chunk_distributor.next_chunk_for(1, 1, 0, True)
    chunk_1_1_1 = chunk_distributor.next_chunk_for(1, 1, 1, True)
    chunk_1_1_2 = chunk_distributor.next_chunk_for(1, 1, 2, True)
    assert chunk_1_1_0 == 3
    assert chunk_1_1_1 == 4
    assert chunk_1_1_2 == 5


def test_cache_management(chunk_distributor):
    # Access the same chunk multiple times and check cache behavior
    _ = chunk_distributor.next_chunk_for(0, 0, 0, True)
    _ = chunk_distributor.next_chunk_for(0, 1, 0, True)
    # After both nodes have accessed chunk 0, it should still be in cache due to deferred eviction
    assert 0 in chunk_distributor._chunk_cache[0]

    # Access further chunks to trigger eviction
    _ = chunk_distributor.next_chunk_for(0, 0, 0, True)  # Worker 0 on node 0 accesses chunk 3
    _ = chunk_distributor.next_chunk_for(0, 1, 0, True)  # Worker 0 on node 1 accesses chunk 3

    # Now chunk 0 should be evicted from the cache
    assert 0 not in chunk_distributor._chunk_cache[0]


def test_chunk_cache_eviction_multiple_chunks(chunk_distributor):
    # Test chunk cache eviction with multiple chunks
    chunk_0_0_0 = chunk_distributor.next_chunk_for(0, 0, 0, True)
    chunk_0_1_0 = chunk_distributor.next_chunk_for(0, 1, 0, True)
    chunk_0_0_1 = chunk_distributor.next_chunk_for(0, 0, 1, True)
    chunk_0_1_1 = chunk_distributor.next_chunk_for(0, 1, 1, True)
    assert chunk_0_0_0 == 0
    assert chunk_0_1_0 == 0
    assert chunk_0_0_1 == 1
    assert chunk_0_1_1 == 1

    # Chunks 0 and 1 should still be in cache due to deferred eviction
    assert 0 in chunk_distributor._chunk_cache[0]
    assert 1 in chunk_distributor._chunk_cache[0]

    # Access further chunks to trigger eviction
    chunk_distributor.next_chunk_for(0, 0, 0, True)  # Worker 0 accesses chunk 3
    chunk_distributor.next_chunk_for(0, 1, 0, True)
    chunk_distributor.next_chunk_for(0, 0, 1, True)  # Worker 1 accesses chunk 4
    chunk_distributor.next_chunk_for(0, 1, 1, True)

    # Now chunks 0 and 1 should be evicted from the cache
    assert 0 not in chunk_distributor._chunk_cache[0]
    assert 1 not in chunk_distributor._chunk_cache[0]


def test_chunk_reuse_across_nodes(chunk_distributor):
    # Test chunk reuse across different nodes
    chunk_0_0_0 = chunk_distributor.next_chunk_for(0, 0, 0, True)
    chunk_0_1_0 = chunk_distributor.next_chunk_for(0, 1, 0, True)
    assert chunk_0_0_0 == 0
    assert chunk_0_1_0 == 0

    chunk_0_0_1 = chunk_distributor.next_chunk_for(0, 0, 1, True)
    chunk_0_1_1 = chunk_distributor.next_chunk_for(0, 1, 1, True)
    assert chunk_0_0_1 == 1
    assert chunk_0_1_1 == 1


def test_end_of_generator(chunk_distributor):
    with pytest.raises(StopIteration):
        for _ in range(101):
            chunk_distributor.next_chunk_for(0, 0, 0, True)


def test_chunk_exhaustion(chunk_distributor):
    # Test behavior when chunks are exhausted
    for _ in range(33):
        chunk_distributor.next_chunk_for(0, 0, 0, True)
        chunk_distributor.next_chunk_for(0, 0, 1, True)
        chunk_distributor.next_chunk_for(0, 0, 2, True)

    chunk_distributor.next_chunk_for(0, 0, 0, True)

    with pytest.raises(StopIteration):
        chunk_distributor.next_chunk_for(0, 0, 1, True)
