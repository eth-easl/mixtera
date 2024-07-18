import multiprocessing as mp
import os
import threading
import warnings
from typing import Any, Generator, Optional

import dill
from loguru import logger
from mixtera.core.datacollection.index import ChunkerIndex
from mixtera.core.query.query_result import QueryResult
from mixtera.utils import hash_string, list_shared_memory, max_shm_len, remove_shm_from_resource_tracker, shm_usage

remove_shm_from_resource_tracker()

# fmt: off
# This import needs to be BELOW the function call
# pylint:disable-next=import-outside-toplevel,unused-import, wrong-import-position
from cengal.hardware.memory.shared_memory import SharedMemory, wait_my_turn  # isort:skip # noqa: E402
# fmt: on


CHUNK_DISTRIBUTOR_SM_NAME = "cd"
CHUNK_DISTRIBUTOR_SM_SIZE_MB = (
    100 if os.getenv("GITHUB_ACTIONS") == "true" else 500
)  # TODO(create issue): make this configurable
SerializedChunkerIndex = bytes


# Filter warnings from unused dependency module
warnings.filterwarnings(action="ignore", module="cengal.code_flow_control.python_bytecode_manipulator")


class ChunkDistributor:
    """
    A class for distributing chunks of data across multiple data parallel groups, nodes, and workers.

    This class manages the distribution of data chunks using shared memory for efficient inter-process
    communication. It handles caching, usage tracking, and cleanup of shared resources.

    Key features:
    - Uses shared memory for efficient data sharing across processes
    - Implements a caching mechanism to avoid redundant data fetching
    - Tracks chunk usage to manage memory efficiently
    - Handles cleanup and finalization of resources

    The class is designed to work with both fork and spawn multiprocessing start methods:
    - For fork: Shared memory is initialized in the constructor, and not copied when forking
    - For spawn: Shared memory is reinitialized in child processes using __setstate__

    IMPORTANT:
    As soon as you start obtaining chunks, you will NEED TO call finalize_worker() for each worker
    that finishes. If not, you will see memory leaks or an exception in __del__ (probably both).

    Args:
        dp_groups (int): Number of data parallel groups
        nodes_per_group (int): Number of nodes per data parallel group
        num_workers (int): Number of workers per node
        query_result (QueryResult): The source of data chunks
        job_id (str): Unique identifier for the job

    Raises:
        ValueError: If dp_groups is less than 1

    Note:
        - The class uses the cengal library for shared memory management
        - The cleanup process is designed to handle the last worker finalizing the shared resources
    """

    def __init__(
        self, dp_groups: int, nodes_per_group: int, num_workers: int, query_result: QueryResult, job_id: str
    ) -> None:
        """
        Initialize the ChunkDistributor with the given parameters.

        This method sets up shared memory, initializes data structures for chunk distribution,
        and prepares the instance for use across multiple processes.
        """

        remove_shm_from_resource_tracker()

        if dp_groups < 1:
            raise ValueError(f"dp_groups = {dp_groups} < 1")

        logger.debug(f"Instantiating ChunkDistributor for job {job_id}")

        # Check whether we have enough shared memory, log for debugging purposes
        total_mb, used_mb, free_mb = shm_usage()
        if free_mb > -1:
            logger.debug(f"-- SHM Usage (MB): Total: {total_mb} Used: {used_mb} Free: {free_mb}")
            if free_mb < CHUNK_DISTRIBUTOR_SM_SIZE_MB:
                logger.warning(
                    f"free_mb = {free_mb} < CHUNK_DISTRIBUTOR_SM_SIZE_MB = {CHUNK_DISTRIBUTOR_SM_SIZE_MB}"
                    + "We might crash due to little shared memory."
                )
                logger.warning(list_shared_memory())

        self._dp_groups = dp_groups
        self._num_workers = num_workers if num_workers > 0 else 1  # num_workers 0 => interpreted as 1 worker
        self._nodes_per_group = nodes_per_group

        self._query_result = query_result
        self._cleanedup = False  # Tracks whether _THIS PROCESS_ currently has an open SharedMemory object
        self._dp_locks: dict[int, Any] = {}  # maps data parallel group ID to mp.Locks, which cannot be used as a type
        self._finalized_workers = mp.Value("i", 0)  # number of workers which have finished
        self._global_cleanedup = mp.Value("B", False)  # tracks whether we are globally cleaned up.
        self._expected_finalizations = self._dp_groups * self._nodes_per_group * self._num_workers
        self._memory_id = f"{CHUNK_DISTRIBUTOR_SM_NAME}_{job_id}"
        self._pre_fork_pid = os.getpid()

        with self._global_cleanedup.get_lock():
            # In case _global_cleanedup is True, then we've already gone through the QueryResult
            # This can happen, for example, when the LocalStub holds multiple ChunkDistributors for multiple queries
            # When we fork/spawn new processes for the workers for the second query, we'll also copy again the instances
            # of the first query. However, we don't want to reinitialize the shared memory, hence we exit early.
            if self._global_cleanedup.get_obj().value:
                logger.debug(f"ChunkDistributor for job {job_id} is already done.")
                self._cleanedup = True
                return

        logger.debug(f"[{os.getpid()}/{threading.get_native_id()}] Initializing chunk with max len = {max_shm_len()}")

        if len(self._memory_id) > max_shm_len():
            # On different systems, the maximum length of shared memory segments differs
            # On my macOS installation, the max length is only 28
            # This code ensures our length stays below the limit to not throw an error
            new_mem_id = hash_string(self._memory_id, max_shm_len() - 1)
            logger.warning(f"shm id of {self._memory_id} is larger than {max_shm_len()}. Updating to {new_mem_id}.")
            self._memory_id = new_mem_id

        # Initialize shared memory
        self._shared_memory: SharedMemory | None = None
        _shared_memory = SharedMemory(self._memory_id, create=True, size=CHUNK_DISTRIBUTOR_SM_SIZE_MB * 1024 * 1024)
        # The _create flag is an internal attribute that decides whether the SM segment is
        # unlinked when close() is called.
        # We will clean up with the last worker that is done, hence not with the actual creator.
        _shared_memory._create = False

        with wait_my_turn(_shared_memory):
            # Initialize shared data structures
            self._chunk_cache, self._chunk_cache_offset = _shared_memory.put_message_2({})
            self._chunk_usage, self._chunk_usage_offset = _shared_memory.put_message_2({})
            self._next_chunk, self._next_chunk_offset = _shared_memory.put_message_2({})

            for dp_group in range(dp_groups):
                self._chunk_cache[dp_group] = _shared_memory.put_message({})
                self._chunk_usage[dp_group] = _shared_memory.put_message({})
                self._next_chunk[dp_group] = _shared_memory.put_message({})
                self._dp_locks[dp_group] = mp.Lock()

                for node in range(nodes_per_group):
                    self._next_chunk[dp_group][node] = _shared_memory.put_message({})
                    for worker_id in range(self._num_workers):
                        # Note that we don't initialize to 0 but to worker_id since each worker
                        # should see a different chunk.
                        self._next_chunk[dp_group][node][worker_id] = worker_id

        # We now delete all references to shared memory segments.
        # In case of 0 workers (no forking/spawning), we'll reinstantiate them in this process
        # in next_chunk for.
        # In case of 1+ workers and fork (Linux default), we'll also reinstantiate it in
        # next_chunk for.
        # In case of 1+ workers and spawn (macOS), we'll reinstantiate it in __setstate__ since
        # all objects are being pickled.
        del self._chunk_cache
        del self._chunk_usage
        del self._next_chunk
        _shared_memory.close()
        self._cleanedup = True  # Right now, we don't have any open shared_memory, so we're clean.
        # Setting this is important, otherwise if we have 1+ workers that terminate sucessfully, the
        # main process will not be marked as clean, throwing in __del__
        logger.debug(f"[{os.getpid()}/{threading.get_native_id()}] Constructor done")

    def next_chunk_for(
        self, dp_group: int, node_id: int, worker_id: int, deserialize: bool
    ) -> Optional[ChunkerIndex | SerializedChunkerIndex]:
        """
        Retrieves the next data chunk for a specified worker in a data parallel group and node.

        Manages the distribution of chunks by tracking their usage and caching them to minimize data fetching.
        This method ensures that each worker receives the appropriate chunk as needed for processing.

        Args:
            dp_group (int): Data parallel group ID
            node_id (int): Node ID within the group
            worker_id (int): Worker ID within the node
            deserialize (bool): Whether to deserialize the chunk before returning

        Returns:
            Optional[ChunkerIndex | SerializedChunkerIndex]: The next chunk or None if no more chunks are available

        Raises:
            AssertionError: If the provided IDs are out of range
        """

        assert dp_group < self._dp_groups
        assert node_id < self._nodes_per_group
        assert worker_id < self._num_workers

        if self._shared_memory is None:
            self._prep_shared_memory()

        with self._dp_locks[dp_group]:
            # The data parallel groups operate on different chunks, i.e., chunk 1 is different for dp 0 and 1
            with wait_my_turn(self._shared_memory):
                next_chunk_id = self._next_chunk[dp_group][node_id][worker_id]

                chunk_to_return: ChunkerIndex | SerializedChunkerIndex
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

                assert next_chunk_id in self._chunk_usage[dp_group], (
                    f"[{os.getpid()}/{threading.get_native_id()}] Chunk {next_chunk_id} is in cache"
                    + f", but not in usage!\n{self._chunk_usage[dp_group]}"
                    + f"\n{self._chunk_cache[dp_group].keys()}\n\n"
                )

                # Increment usage count for this chunk
                current_usage = self._chunk_usage[dp_group][next_chunk_id] + 1
                self._chunk_usage[dp_group][next_chunk_id] = current_usage

                # Check if all nodes have seen this chunk
                if current_usage >= self._nodes_per_group:
                    # Potentially useful debug log
                    # logger.debug(
                    #    f"[{os.getpid()}/{threading.get_native_id()}] Purging chunk {next_chunk_id}"
                    #    + f"for dp_group {dp_group} from cache."
                    # )
                    del self._chunk_cache[dp_group][next_chunk_id]
                    del self._chunk_usage[dp_group][next_chunk_id]
                # random
                # We don't increment by 1 but instead by num_workers, because otherwise
                # we get an overlap between workers after the first chunk
                self._next_chunk[dp_group][node_id][worker_id] = next_chunk_id + self._num_workers
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
            except Exception:
                logger.error("Unexpected error, cleaning up shared memory and reraising.")
                self.cleanup()
                raise

    def finalize_worker(self) -> None:
        """
        Finalize a worker and perform cleanup if it's the last worker.

        This method is called when a worker has finished processing. It increments
        the finalized workers counter and triggers the cleanup process if all
        workers have been finalized.
        """
        do_cleanup = True

        with self._finalized_workers.get_lock():
            self._finalized_workers.get_obj().value += 1
            logger.debug(
                f"[{os.getpid()}/{threading.get_native_id()}]"
                + f"Finalized workers: {self._finalized_workers.get_obj().value}/{self._expected_finalizations}"
            )
            if self._finalized_workers.get_obj().value == self._expected_finalizations:
                # Hack: the last worker to be finalized sets _create which
                # only then finally removes the global block
                assert self._shared_memory is not None
                self._shared_memory._create = True
                self._shared_memory._creator_destroy_timeout = (
                    0.0001  # we're the last worker, no need to wait for any consumer
                )
                with self._global_cleanedup.get_lock():
                    self._global_cleanedup.get_obj().value = True

                logger.debug(f"[{os.getpid()}/{threading.get_native_id()}] Last worker finalized!")
            else:
                assert self._shared_memory is not None
                self._shared_memory._create = False

                # If we're not the last worker AND we have not forked at all, we're using server mode
                # Local with 1+ workers would work
                # Local with 0 worker wouldn't fork but only finalizes once
                # In that case, do not clean up yet!
                if os.getpid() == self._pre_fork_pid:
                    do_cleanup = False

            if do_cleanup:
                self.cleanup()

    def cleanup(self) -> None:
        """
        Clean up shared resources associated with this ChunkDistributor instance.

        This method ensures that shared memory and other resources are properly
        released when they are no longer needed.
        """
        if not self._cleanedup:
            logger.debug(f"[{os.getpid()}/{threading.get_native_id()}] Cleaning up.")
            self._cleanedup = True

            if self._shared_memory is not None:
                self._shared_memory.proper_close()
                del self._shared_memory
                self._shared_memory = None

            if hasattr(self, "_chunk_cache"):
                del self._chunk_cache
            if hasattr(self, "_chunk_usage"):
                del self._chunk_usage
            if hasattr(self, "_next_chunk"):
                del self._next_chunk

    def __del__(self) -> None:
        # This indicates wrong usage and thus we should immediately fail to notice this
        assert (
            self._cleanedup
        ), f"[{os.getpid()}/{threading.get_native_id()}] You did not clean up the ChunkDistributor!"

    def _prep_shared_memory(self) -> None:
        """
        Prepare shared memory for use in a child process.

        This method is called to reinitialize shared memory access in a spawned (or forked)
        child process. It sets up the necessary shared memory connections and
        retrieves the shared objects.
        """

        self._cleanedup = False
        self._shared_memory = SharedMemory(self._memory_id)
        self._shared_memory.init_consumer()

        with wait_my_turn(self._shared_memory):
            self._chunk_cache = self._shared_memory.get_object(self._chunk_cache_offset)
            self._chunk_usage = self._shared_memory.get_object(self._chunk_usage_offset)
            self._next_chunk = self._shared_memory.get_object(self._next_chunk_offset)

    def __getstate__(self) -> dict:
        """
        Prepare the instance for pickling.

        This method is used when the instance needs to be serialized, typically
        for the 'spawn' start method in multiprocessing. It removes non-picklable
        attributes related to shared memory.

        Returns:
            dict: A picklable state of the instance
        """

        state = self.__dict__.copy()
        if "_shared_memory" in state:
            del state["_shared_memory"]
        if "_chunk_cache" in state:
            del state["_chunk_cache"]
        if "_chunk_usage" in state:
            del state["_chunk_usage"]
        if "_next_chunk" in state:
            del state["_next_chunk"]

        return state

    def __setstate__(self, state: dict) -> None:
        """
        Restore the instance state after unpickling.

        This method is used to reconstruct the instance in a child process,
        typically when using the 'spawn' start method. It reinitializes
        shared memory.

        Args:
            state (dict): The pickled state of the instance
        """
        self.__dict__ = state
        self._shared_memory = None

        with self._global_cleanedup.get_lock():
            if not self._global_cleanedup.get_obj().value:
                self._prep_shared_memory()
