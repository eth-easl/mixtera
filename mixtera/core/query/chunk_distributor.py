import hashlib
import multiprocessing as mp
import os
import threading
import warnings
from functools import cached_property
from typing import Any, Generator, Optional

import dill
from loguru import logger
from mixtera.core.datacollection.index import ChunkerIndex
from mixtera.core.query.query_result import QueryResult


# We need to fix the resource tracker of Python because otherwise it calls _shmunlink before we want it to.
def remove_shm_from_resource_tracker() -> None:
    """Monkey-patch multiprocessing.resource_tracker so SharedMemory won't be tracked

    More details at: https://bugs.python.org/issue38119
    """

    # We need this import, otherwise for some reason also mp.resource_tracker is undefined.
    from multiprocessing import resource_tracker  # noqa: F401 # pylint:disable=import-outside-toplevel,unused-import

    def fix_register(name: str, rtype: str) -> None:
        if rtype == "shared_memory":
            return
        mp.resource_tracker._resource_tracker.register(name, rtype)

    mp.resource_tracker.register = fix_register  # type: ignore

    def fix_unregister(name: str, rtype: str) -> None:
        if rtype == "shared_memory":
            return
        mp.resource_tracker._resource_tracker.unregister(name, rtype)

    mp.resource_tracker.unregister = fix_unregister  # type: ignore

    if "shared_memory" in mp.resource_tracker._CLEANUP_FUNCS:  # type: ignore
        del mp.resource_tracker._CLEANUP_FUNCS["shared_memory"]  # type: ignore


remove_shm_from_resource_tracker()

# fmt: off
# This import needs to be BELOW the function call
# pylint:disable-next=import-outside-toplevel,unused-import, wrong-import-position
from cengal.hardware.memory.shared_memory import SharedMemory, wait_my_turn  # isort:skip # noqa: E402
# fmt: on


CHUNK_DISTRIBUTOR_SM_NAME = "cd"
SerializedChunkerIndex = bytes


# Filter warnings from unused dependency module
warnings.filterwarnings(action="ignore", module="cengal.code_flow_control.python_bytecode_manipulator")


class ChunkDistributor:
    def __init__(
        self, dp_groups: int, nodes_per_group: int, num_workers: int, query_result: QueryResult, job_id: str
    ) -> None:
        remove_shm_from_resource_tracker()
        if dp_groups < 1:
            raise ValueError(f"dp_groups = {dp_groups} < 1")
        logger.debug(f"Instantiating ChunkDistributor for job {job_id}")
        self._dp_groups = dp_groups
        self._num_workers = num_workers if num_workers > 0 else 1  # num_workers 0 => interpreted as 1 worker
        self._nodes_per_group = nodes_per_group

        self._query_result = query_result
        self._cleanedup = False  # tracks whether _this process_ has cleaned up
        self._dp_locks: dict[int, Any] = {}
        self._finalized_workers = mp.Value("i", 0)
        self._global_cleanedup = mp.Value("B", False)  # tracks whether we are globally cleaned up
        self._expected_finalizations = self._dp_groups * self._nodes_per_group * self._num_workers
        self._memory_id = f"{CHUNK_DISTRIBUTOR_SM_NAME}_{job_id}"

        with self._global_cleanedup.get_lock():
            if self._global_cleanedup.get_obj().value:
                logger.debug(f"ChunkDistributor for job {job_id} is already done.")
                self._cleanedup = True
                return

        logger.info(f"[{os.getpid()}/{threading.get_native_id()}] Initializing chunk with max len = {self.max_shm_len}")
        if len(self._memory_id) > self.max_shm_len:
            new_mem_id = self.hash_string(self._memory_id, self.max_shm_len - 1)
            logger.warning(f"shm id of {self._memory_id} is larger than {self.max_shm_len}. Updating to {new_mem_id}.")
            self._memory_id = new_mem_id

        # Initialize shared memory
        self._shared_memory: SharedMemory | None = None
        _shared_memory = SharedMemory(self._memory_id, create=True, size=100 * 1024 * 1024)  # Adjust size as needed
        _shared_memory._create = False  # We will clean up with the last worker that is done

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
                        self._next_chunk[dp_group][node][worker_id] = worker_id

        _shared_memory.close_consumer()

    @cached_property
    def max_shm_len(self) -> int:
        base_name = "/a"
        for i in range(1, 500):
            try:
                name = base_name + "a" * i
                shm = mp.shared_memory.SharedMemory(name=name, create=True, size=10)
                shm.close()
                shm.unlink()
            except OSError:
                return i - 1
        return i

    def hash_string(self, input_string: str, length: int) -> str:
        if length > hashlib.sha256().digest_size:
            raise ValueError("Requested length exceeds the maximum allowed by SHA-256")
        hash_obj = hashlib.sha256()
        hash_obj.update(input_string.encode("utf-8"))
        hex_digest = hash_obj.hexdigest()
        return hex_digest[:length]

    def next_chunk_for(
        self, dp_group: int, node_id: int, worker_id: int, deserialize: bool
    ) -> Optional[ChunkerIndex | SerializedChunkerIndex]:
        assert dp_group < self._dp_groups
        assert node_id < self._nodes_per_group
        assert worker_id < self._num_workers

        if self._shared_memory is None:
            self._prep_shared_memory()

        # When this function is called directly (e.g., in tests), use self._shared_memory
        # shared_memory = shared_memory if shared_memory is not None else self._shared_memory
        # assert shared_memory is not None
        with self._dp_locks[dp_group]:
            with wait_my_turn(self._shared_memory):
                next_chunk_id = self._next_chunk[dp_group][node_id][worker_id]

                chunk_to_return: ChunkerIndex | SerializedChunkerIndex
                if next_chunk_id not in self._chunk_cache[dp_group]:
                    # Fetch new chunk from query result and put into cache
                    chunk_to_return = next(self._query_result)
                    serialized_chunk = dill.dumps(chunk_to_return)
                    self._chunk_cache[dp_group][next_chunk_id] = serialized_chunk
                    self._chunk_usage[dp_group][next_chunk_id] = 0

                    if not deserialize:
                        chunk_to_return = serialized_chunk
                else:
                    # Load from cache
                    chunk_to_return = self._chunk_cache[dp_group][next_chunk_id]  # always serialized in cache
                    if deserialize:
                        chunk_to_return = dill.loads(chunk_to_return)

                # Increment usage count for this chunk
                self._chunk_usage[dp_group][next_chunk_id] += 1

                # Check if all nodes have seen this chunk
                if self._chunk_usage[dp_group][next_chunk_id] >= self._nodes_per_group:
                    del self._chunk_cache[dp_group][next_chunk_id]
                    del self._chunk_usage[dp_group][next_chunk_id]

                # We don't increment by 1 but instead by num_workers, because otherwise
                # we get an overlap between workers after the first chunk
                self._next_chunk[dp_group][node_id][worker_id] += self._num_workers
                return chunk_to_return

    def _stream_chunks_for_worker(
        self, dp_group_id: int, node_id: int, worker_id: int
    ) -> Generator[ChunkerIndex | SerializedChunkerIndex, None, None]:
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

            self.cleanup()

    def cleanup(self) -> None:
        if not self._cleanedup:
            logger.debug(f"[{os.getpid()}/{threading.get_native_id()}] Cleaning up.")
            self._cleanedup = True

            if self._shared_memory is not None:
                self._shared_memory.proper_close()

            del self._shared_memory
            del self._chunk_cache
            del self._chunk_usage
            del self._next_chunk

    def __del__(self) -> None:
        # This indicates wrong usage and thus we should immediately fail to notice this
        assert self._cleanedup, "You did not clean up the ChunkDistributor!"

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        if not self._cleanedup:
            del state["_shared_memory"]
            del state["_chunk_cache"]
            del state["_chunk_usage"]
            del state["_next_chunk"]

        return state

    def _prep_shared_memory(self) -> None:
        self._shared_memory = SharedMemory(self._memory_id)
        self._shared_memory.init_consumer()

        with wait_my_turn(self._shared_memory):
            self._chunk_cache = self._shared_memory.get_object(self._chunk_cache_offset)
            self._chunk_usage = self._shared_memory.get_object(self._chunk_usage_offset)
            self._next_chunk = self._shared_memory.get_object(self._next_chunk_offset)

    def __setstate__(self, state: dict) -> None:
        self.__dict__ = state
        with self._global_cleanedup.get_lock():
            if not self._global_cleanedup.get_obj().value:
                self._prep_shared_memory()
