import os
import threading
from typing import Generator

import dill
from loguru import logger
from mixtera.core.datacollection.index import ChunkerIndex
from mixtera.core.query.query_result import QueryResult

SerializedChunkerIndex = bytes


class ChunkDistributor:
    def __init__(
        self,
        dp_groups: int,
        nodes_per_group: int,
        num_workers: int,
        query_result: QueryResult,
        job_id: str,
    ) -> None:
        if dp_groups < 1:
            raise ValueError(f"dp_groups = {dp_groups} < 1")

        logger.debug(f"[{os.getpid()}/{threading.get_native_id()}] Instantiating ChunkDistributor for job {job_id}")

        self._dp_groups = dp_groups
        self._num_workers = num_workers if num_workers > 0 else 1  # num_workers 0 => interpreted as 1 worker
        self._nodes_per_group = nodes_per_group

        self._query_result = query_result
        self._constructor_pid = os.getpid()

        self._chunk_cache: dict[int, dict[int, SerializedChunkerIndex | ChunkerIndex]] = {}
        self._chunk_usage: dict[int, dict[int, int]] = {}
        self._next_chunk: dict[int, dict[int, dict[int, int]]] = {}

        for dp_group in range(dp_groups):
            self._chunk_cache[dp_group] = {}
            self._chunk_usage[dp_group] = {}
            self._next_chunk[dp_group] = {}

            for node in range(nodes_per_group):
                self._next_chunk[dp_group][node] = {}
                for worker_id in range(self._num_workers):
                    # Note that we don't initialize to 0 but to worker_id since each worker
                    # should see a different chunk.
                    self._next_chunk[dp_group][node][worker_id] = worker_id

    def next_chunk_for(
        self, dp_group: int, node_id: int, worker_id: int, deserialize: bool
    ) -> ChunkerIndex | SerializedChunkerIndex:
        """
        Retrieves the next data chunk for a specified worker in a data parallel group and node.

        Manages the distribution of chunks by tracking their usage and caching them to minimize data fetching.
        This method ensures that each worker receives the appropriate chunk as needed for processing.

        This function is NOT threadsafe. This is not a problem. In the server case, asyncio coroutines
        will not be executed in parallel. In the local case, we fork anyways.

        Args:
            dp_group (int): Data parallel group ID
            node_id (int): Node ID within the group
            worker_id (int): Worker ID within the node
        Raises:
            AssertionError: If the provided IDs are out of range
            StopIteration: If there are no
        """

        assert dp_group < self._dp_groups
        assert node_id < self._nodes_per_group
        assert worker_id < self._num_workers
        chunk_to_return: ChunkerIndex | SerializedChunkerIndex

        if self._nodes_per_group == 1:
            # No need for caching logic.
            # We will hand out each chunk exactly once.
            # Just get the next chunk, which is currently cross-process safe: Even if we fork (local mode),
            # then each chunk will be handed out once, which is exactly what we need.
            chunk_to_return = next(self._query_result)
            if deserialize:
                return chunk_to_return

            return dill.dumps(chunk_to_return)

        # In this case, we're in server mode and need to use the caching logic.
        # If we fork in server mode, we won't have a shared cache, and that will go wrong.
        curr_pid = os.getpid()
        if curr_pid != self._constructor_pid:
            raise RuntimeError(
                f"We seem to have forked ({curr_pid} vs {self._constructor_pid}) but we're in server mode."
            )

        next_chunk_id = self._next_chunk[dp_group][node_id][worker_id]
        # The data parallel groups operate on different chunks, i.e., chunk 1 is different for dp 0 and 1
        if next_chunk_id not in self._chunk_cache[dp_group]:
            # Potentially useful debug log
            # logger.debug(f"Fetching chunk {next_chunk_id} for dp_group {dp_group} /
            # node {node_id} requested by worker {worker_id} from QueryResult.")

            # Fetch new chunk from query result and put into cache
            chunk_to_return = next(self._query_result)
            serialized_chunk = dill.dumps(chunk_to_return)
            self._chunk_cache[dp_group][next_chunk_id] = serialized_chunk
            self._chunk_usage[dp_group][next_chunk_id] = 0

            if not deserialize:
                chunk_to_return = serialized_chunk
        else:
            # Potentially useful debug log
            # logger.debug(f"Fetching chunk {next_chunk_id} for dp_group {dp_group} /
            # node {node_id} requested by worker {worker_id} from cache.")
            # Load from cache
            chunk_to_return = self._chunk_cache[dp_group][next_chunk_id]  # always serialized in cache
            if deserialize:
                chunk_to_return = dill.loads(chunk_to_return)

        # Increment usage count for this chunk
        self._chunk_usage[dp_group][next_chunk_id] += 1

        # Check if all nodes have seen this chunk
        if self._chunk_usage[dp_group][next_chunk_id] >= self._nodes_per_group:
            # Potentially useful debug log
            # logger.debug(
            #    f"[{os.getpid()}/{threading.get_native_id()}] Purging chunk {next_chunk_id}"
            #    + f"for dp_group {dp_group} from cache."
            # )
            del self._chunk_cache[dp_group][next_chunk_id]
            del self._chunk_usage[dp_group][next_chunk_id]

        # We don't increment by 1 but instead by num_workers, because otherwise
        # we get an overlap between workers after the first chunk
        self._next_chunk[dp_group][node_id][worker_id] += self._num_workers
        return chunk_to_return

    def _stream_chunks_for_worker(
        self, dp_group_id: int, node_id: int, worker_id: int
    ) -> Generator[ChunkerIndex | SerializedChunkerIndex, None, None]:
        """
        Generate a stream of chunks for a specific worker.
        Used for local training.

        Args:
            dp_group_id (int): Data parallel group ID
            node_id (int): Node ID within the group
            worker_id (int): Worker ID within the node

        Yields:
            ChunkerIndex | SerializedChunkerIndex: The next chunk for the worker

        """

        while True:
            try:
                yield self.next_chunk_for(dp_group_id, node_id, worker_id, True)
            except StopIteration:
                return
