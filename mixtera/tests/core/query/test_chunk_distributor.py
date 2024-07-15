from multiprocessing.managers import SyncManager

import pytest
from mixtera.core.query.chunk_distributor import ChunkDistributor


@pytest.fixture(name="query_result")
def fixture_query_result():
    # Mocking QueryResult to return a generator of indices
    def result_generator():
        yield from range(100)

    return result_generator()


@pytest.fixture(name="chunk_distributor")
def fixture_chunk_distributor(query_result):
    return ChunkDistributor(dp_groups=2, nodes_per_group=2, num_workers=3, query_result=query_result)


def test_initialization(chunk_distributor):
    assert isinstance(chunk_distributor._manager, SyncManager)
    assert len(chunk_distributor._chunk_cache) == 2
    assert len(chunk_distributor._dp_locks) == 2


def test_chunk_distribution_within_dp(chunk_distributor):
    chunk_0_0_0 = chunk_distributor.next_chunk_for(0, 0, 0)
    chunk_0_0_1 = chunk_distributor.next_chunk_for(0, 0, 1)
    chunk_0_0_2 = chunk_distributor.next_chunk_for(0, 0, 2)
    assert chunk_0_0_0 == 0
    assert chunk_0_0_1 == 1
    assert chunk_0_0_2 == 2

    chunk_0_1_0 = chunk_distributor.next_chunk_for(0, 1, 0)
    chunk_0_1_1 = chunk_distributor.next_chunk_for(0, 1, 1)
    chunk_0_1_2 = chunk_distributor.next_chunk_for(0, 1, 2)
    assert chunk_0_1_0 == 0
    assert chunk_0_1_1 == 1
    assert chunk_0_1_2 == 2

    chunk_0_0_0 = chunk_distributor.next_chunk_for(0, 0, 0)
    chunk_0_0_1 = chunk_distributor.next_chunk_for(0, 0, 1)
    assert chunk_0_0_0 == 3
    assert chunk_0_0_1 == 4

    chunk_0_1_0 = chunk_distributor.next_chunk_for(0, 1, 0)
    chunk_0_1_2 = chunk_distributor.next_chunk_for(0, 1, 2)
    assert chunk_0_1_0 == 3
    assert chunk_0_1_2 == 5

    chunk_0_0_2 = chunk_distributor.next_chunk_for(0, 0, 2)
    assert chunk_0_0_2 == 5


def test_chunk_distribution_across_dps(chunk_distributor):
    chunk_0_0_0 = chunk_distributor.next_chunk_for(0, 0, 0)
    chunk_0_0_1 = chunk_distributor.next_chunk_for(0, 0, 1)
    chunk_0_0_2 = chunk_distributor.next_chunk_for(0, 0, 2)
    assert chunk_0_0_0 == 0
    assert chunk_0_0_1 == 1
    assert chunk_0_0_2 == 2

    chunk_1_0_0 = chunk_distributor.next_chunk_for(1, 0, 0)
    chunk_1_0_1 = chunk_distributor.next_chunk_for(1, 0, 1)
    chunk_1_0_2 = chunk_distributor.next_chunk_for(1, 0, 2)
    assert chunk_1_0_0 == 3
    assert chunk_1_0_1 == 4
    assert chunk_1_0_2 == 5

    chunk_0_1_0 = chunk_distributor.next_chunk_for(0, 1, 0)
    chunk_0_1_1 = chunk_distributor.next_chunk_for(0, 1, 1)
    chunk_0_1_2 = chunk_distributor.next_chunk_for(0, 1, 2)
    assert chunk_0_1_0 == 0
    assert chunk_0_1_1 == 1
    assert chunk_0_1_2 == 2

    chunk_1_1_0 = chunk_distributor.next_chunk_for(1, 1, 0)
    chunk_1_1_1 = chunk_distributor.next_chunk_for(1, 1, 1)
    chunk_1_1_2 = chunk_distributor.next_chunk_for(1, 1, 2)
    assert chunk_1_1_0 == 3
    assert chunk_1_1_1 == 4
    assert chunk_1_1_2 == 5


def test_cache_management(chunk_distributor):
    # Access the same chunk multiple times and check cache behavior
    _ = chunk_distributor.next_chunk_for(0, 0, 0)
    _ = chunk_distributor.next_chunk_for(0, 1, 0)
    # This should remove the chunk from cache as it has been accessed by all nodes
    assert 0 not in chunk_distributor._chunk_cache[0]


def test_chunk_cache_eviction_multiple_chunks(chunk_distributor):
    # Test chunk cache eviction with multiple chunks
    chunk_0_0_0 = chunk_distributor.next_chunk_for(0, 0, 0)
    chunk_0_1_0 = chunk_distributor.next_chunk_for(0, 1, 0)
    chunk_0_0_1 = chunk_distributor.next_chunk_for(0, 0, 1)
    chunk_0_1_1 = chunk_distributor.next_chunk_for(0, 1, 1)
    assert chunk_0_0_0 == 0
    assert chunk_0_1_0 == 0
    assert chunk_0_0_1 == 1
    assert chunk_0_1_1 == 1
    assert 0 not in chunk_distributor._chunk_cache[0]
    assert 1 not in chunk_distributor._chunk_cache[0]


def test_chunk_reuse_across_nodes(chunk_distributor):
    # Test chunk reuse across different nodes
    chunk_0_0_0 = chunk_distributor.next_chunk_for(0, 0, 0)
    chunk_0_1_0 = chunk_distributor.next_chunk_for(0, 1, 0)
    assert chunk_0_0_0 == 0
    assert chunk_0_1_0 == 0

    chunk_0_0_1 = chunk_distributor.next_chunk_for(0, 0, 1)
    chunk_0_1_1 = chunk_distributor.next_chunk_for(0, 1, 1)
    assert chunk_0_0_1 == 1
    assert chunk_0_1_1 == 1


def test_serialization(chunk_distributor, query_result):
    state = chunk_distributor.__getstate__()
    new_distributor = ChunkDistributor(2, 2, 3, query_result)
    new_distributor.__setstate__(state)
    assert new_distributor._dp_groups == 2


def test_end_of_generator(chunk_distributor):
    with pytest.raises(StopIteration):
        for _ in range(101):
            chunk_distributor.next_chunk_for(0, 0, 0)


def test_chunk_exhaustion(chunk_distributor):
    # Test behavior when chunks are exhausted
    for _ in range(33):
        chunk_distributor.next_chunk_for(0, 0, 0)
        chunk_distributor.next_chunk_for(0, 0, 1)
        chunk_distributor.next_chunk_for(0, 0, 2)

    chunk_distributor.next_chunk_for(0, 0, 0)

    with pytest.raises(StopIteration):
        chunk_distributor.next_chunk_for(0, 0, 1)
