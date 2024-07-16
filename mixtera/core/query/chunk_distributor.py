import multiprocessing as mp
import threading
from multiprocessing.managers import DictProxy, ValueProxy
from typing import Generator, Optional

from mixtera.core.datacollection.index import ChunkerIndex
from mixtera.core.query.query_result import QueryResult


class ChunkDistributor:
    def __init__(self, dp_groups: int, nodes_per_group: int, num_workers: int, query_result: QueryResult) -> None:
        if dp_groups < 1:
            raise ValueError(f"dp_groups = {dp_groups} < 1")
        self._dp_groups = dp_groups
        self._num_workers = num_workers if num_workers > 0 else 1  # num_workers 0 => interpreted as 1 worker
        self._nodes_per_group = nodes_per_group

        self._query_result = query_result
        self._manager = mp.Manager()

        # We need to pre-initialize all dicts because the manager does not get pickled.

        # dp group -> chunk id -> chunk
        self._chunk_cache: DictProxy[int, DictProxy[int, ChunkerIndex]] = self._manager.dict()
        # dp group -> chunk id -> number of nodes that have consumed it
        self._chunk_usage: DictProxy[int, DictProxy[int, int]] = self._manager.dict()
        # dp group -> node id -> next chunk
        self._next_chunk: DictProxy[int, DictProxy[int, DictProxy[int, ValueProxy[int]]]] = self._manager.dict()
        # dp group -> lock
        self._dp_locks: DictProxy[int, threading.Lock] = self._manager.dict()

        # Initialize nested dicts
        for dp_group in range(dp_groups):
            self._chunk_cache[dp_group] = self._manager.dict()
            self._chunk_usage[dp_group] = self._manager.dict()
            self._next_chunk[dp_group] = self._manager.dict()
            self._dp_locks[dp_group] = self._manager.Lock()
            for node in range(nodes_per_group):
                self._next_chunk[dp_group][node] = self._manager.dict()
                for worker_id in range(num_workers):
                    # Since the workers do not share chunks, we offset their start by 1 each
                    self._next_chunk[dp_group][node][worker_id] = self._manager.Value("i", worker_id)

    def next_chunk_for(self, dp_group: int, node_id: int, worker_id: int) -> Optional[ChunkerIndex]:
        assert dp_group < self._dp_groups
        assert node_id < self._nodes_per_group
        assert worker_id < self._num_workers

        with self._dp_locks[dp_group]:
            next_chunk_id = self._next_chunk[dp_group][node_id][worker_id].value

            if next_chunk_id not in self._chunk_cache[dp_group]:
                # Load new chunk if not in cache
                chunk = next(self._query_result)
                self._chunk_cache[dp_group][next_chunk_id] = chunk
                self._chunk_usage[dp_group][next_chunk_id] = 0

            # Increment usage count for this chunk
            self._chunk_usage[dp_group][next_chunk_id] += 1
            chunk_to_return = self._chunk_cache[dp_group][next_chunk_id]

            # Check if all nodes have seen this chunk
            if self._chunk_usage[dp_group][next_chunk_id] >= self._nodes_per_group:
                del self._chunk_cache[dp_group][next_chunk_id]
                del self._chunk_usage[dp_group][next_chunk_id]

            # We don't increment by 1 but instead by num_workers, because otherwise
            # we get an overlap between workers after the first chunk
            self._next_chunk[dp_group][node_id][worker_id].value += self._num_workers

            return chunk_to_return

    def _stream_chunks_for_worker(
        self, dp_group_id: int, node_id: int, worker_id: int
    ) -> Generator[ChunkerIndex, None, None]:
        while True:
            try:
                chunk = self.next_chunk_for(dp_group_id, node_id, worker_id)
                yield chunk
            except StopIteration:
                return

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        if "_manager" in state:
            del state["_manager"]

        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__ = state
