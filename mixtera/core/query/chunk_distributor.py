import multiprocessing as mp
import os
import threading
from copy import deepcopy
from pathlib import Path
from queue import Empty
from typing import Any, Generator

import dill
from loguru import logger
from mixtera.core.query.query_result import QueryResult
from mixtera.core.query.result_chunk import ResultChunk

SerializedResultChunk = bytes


class ChunkDistributor:
    """
    A class responsible for distributing data chunks across multiple data parallel groups, nodes, and workers.

    This class manages the distribution of data chunks, ensuring efficient caching and usage tracking
    to minimize data fetching and optimize performance in distributed computing environments. Caching is only
    used when required, i.e., when using more than one node per data parallel group. In this case, serialized
    chunks are cached to avoid serializing multiple times.
    """

    def __init__(
        self,
        dp_groups: int,
        nodes_per_group: int,
        num_workers: int,
        query_result: QueryResult,
        job_id: str,
    ) -> None:
        """
        Initialize the ChunkDistributor.

        Args:
            dp_groups (int): Number of data parallel groups.
            nodes_per_group (int): Number of nodes per data parallel group.
            num_workers (int): Number of workers per node.
            query_result (QueryResult): The source of data chunks.
            job_id (str): Unique identifier for the job.

        Raises:
            ValueError: If dp_groups is less than 1.
        """
        if dp_groups < 1:
            raise ValueError(f"dp_groups = {dp_groups} < 1")

        logger.debug(f"[{os.getpid()}/{threading.get_native_id()}] Instantiating ChunkDistributor for job {job_id}")

        self._dp_groups = dp_groups
        self._num_workers = num_workers if num_workers > 0 else 1  # num_workers 0 => interpreted as 1 worker
        self._og_num_workers = num_workers
        self._nodes_per_group = nodes_per_group

        self._query_result = query_result
        self._constructor_pid = os.getpid()

        self._chunk_cache: dict[int, dict[int, SerializedResultChunk | ResultChunk]] = {}
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

        # Global checkpointing data structures
        self._checkpoint_lock = mp.Lock()  # Global lock for checkpointing
        self._worker_statuses: dict[tuple[int, int], list[int]] = {}  # (dp_group_id, node_id) -> worker_status
        self._nodes_reported: set = (
            set()
        )  # Set of (dp_group_id, node_id) that have reported their status for checkpointing
        self._checkpoint_id_counter = mp.Value("i", 0)  # Counter to assign unique checkpoint IDs
        self._checkpoint_info: dict[str, dict[str, Any]] = {}  # Info for each checkpoint process on its status
        self._current_checkpoint_id: str | None = None
        self._worker_queues: None | list[mp.Queue] = None  # need this for local mode where we fork
        self.__local_mode = self._nodes_per_group == 1 and self._dp_groups == 1
        self._return_from_queue = {worker_id: False for worker_id in range(self._num_workers)}

    @property
    def _local_mode(self) -> bool:
        # Local mode is when nodes_per_group == 1, dp_groups == 1, and the current PID is different from constructor PID
        return self.__local_mode and (os.getpid() != self._constructor_pid or self._og_num_workers == 0)

    def _initialize_worker_queues(self) -> None:
        if not self._local_mode:
            raise RuntimeError("Unexpected behavior: Worker queues should not have been initialized.")

        if self._worker_queues is None:
            # Since we have forked, we need to initialize the queues
            self._worker_queues = [mp.Queue(maxsize=1) for _ in range(self._num_workers)]

    def next_chunk_for(
        self, dp_group: int, node_id: int, worker_id: int, deserialize: bool
    ) -> ResultChunk | SerializedResultChunk:
        """
        Retrieve the next data chunk for a specified worker in a data parallel group and node.

        This method manages the distribution of chunks by tracking their usage
        and caching them to minimize data fetching.
        It ensures that each worker receives the appropriate chunk as needed for processing.
        Chunks are cached serialized (when Mixtera is used as expected) to avoid the overhead
        of serializing multiple times.

        Note:
            This method is not thread-safe. In the server case, asyncio coroutines will not be executed in parallel.
            In the local case, the process is forked.

        Args:
            dp_group (int): Data parallel group ID.
            node_id (int): Node ID within the group.
            worker_id (int): Worker ID within the node.
            deserialize (bool): Whether to deserialize the chunk before returning.

        Returns:
            ResultChunk | SerializedResultChunk: The next chunk for the specified worker.

        Raises:
            AssertionError: If the provided IDs are out of range.
            StopIteration: If there are no more chunks available.
            RuntimeError: If a fork is detected in server mode.
        """

        assert dp_group < self._dp_groups
        assert node_id < self._nodes_per_group
        assert worker_id < self._num_workers
        chunk_to_return: ResultChunk | SerializedResultChunk

        if self._local_mode:  # No need for caching logic because each chunk will only be handed out exactly once
            # Furthermore, we cannot rely on the cache dicts because in local mode,
            # the worker processes have forked, i.e., the dicts are not shared memory.
            # We will hand out each chunk exactly once.
            # Just get the next chunk, which is currently cross-process safe: Even if we fork,
            # then each chunk will be handed out once, which is exactly what we need.
            self._initialize_worker_queues()
            assert self._worker_queues is not None
            queue = self._worker_queues[worker_id]
            if not self._return_from_queue[worker_id]:
                chunk_to_return = next(self._query_result)
            else:
                try:
                    chunk_to_return = queue.get(timeout=2)
                except Empty as e:
                    raise RuntimeError(f"self._return_from_queue = True but empty queue for worker {worker_id}") from e

                self._return_from_queue[worker_id] = False

            # Put last_chunk into the queue, ensuring the queue length remains 1
            try:
                queue.get_nowait()  # Remove existing item if any
            except Empty:
                pass
            # Put the new chunk into the queue
            if not deserialize:
                serialized_chunk = dill.dumps(chunk_to_return)
                queue.put(serialized_chunk)
                return serialized_chunk
            else:
                queue.put(chunk_to_return)
                return chunk_to_return

        # Server mode logic (or local with 0 workers)
        curr_pid = os.getpid()

        # In this case, we're in server mode and need to use the caching logic.
        # If we fork in server mode, we won't have a shared cache, and that will go wrong.
        if curr_pid != self._constructor_pid:
            raise RuntimeError(
                f"We seem to have forked ({curr_pid} vs {self._constructor_pid}) but we're in server mode."
            )

        if deserialize:
            logger.warning(
                "You are using Mixtera with caching, but do not serialize the chunks. This is unexpected behavior."
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

        # Check if all nodes have received this chunk
        if self._chunk_usage[dp_group][next_chunk_id] >= self._nodes_per_group:
            if (chunk_to_delete := next_chunk_id - self._num_workers) >= 0:
                # Delete the previous chunk as all nodes have now received the next chunk
                del self._chunk_cache[dp_group][chunk_to_delete]
                del self._chunk_usage[dp_group][chunk_to_delete]
                # Potentially useful debug log
                # logger.debug(
                #    f"[{os.getpid()}/{threading.get_native_id()}] Purging chunk {chunk_to_delete}"
                #    + f"for dp_group {dp_group} from cache."
                # )

        # We don't increment by 1 but instead by num_workers, because otherwise
        # we get an overlap between workers after the first chunk
        self._next_chunk[dp_group][node_id][worker_id] += self._num_workers
        return chunk_to_return

    def _stream_chunks_for_worker(
        self, dp_group_id: int, node_id: int, worker_id: int
    ) -> Generator[ResultChunk | SerializedResultChunk, None, None]:
        """
        Generate a stream of chunks for a specific worker.

        This method is used for local training, providing a continuous stream of data chunks
        for a given worker in a specific data parallel group and node.

        Args:
            dp_group_id (int): Data parallel group ID.
            node_id (int): Node ID within the group.
            worker_id (int): Worker ID within the node.

        Yields:
            ResultChunk | SerializedResultChunk: The next chunk for the worker.

        Note:
            The stream ends when there are no more chunks available (StopIteration is caught internally).
        """

        while True:
            try:
                yield self.next_chunk_for(dp_group_id, node_id, worker_id, True)
            except StopIteration:
                return

    def checkpoint(self, dp_group_id: int, node_id: int, worker_status: list[int], chkpnt_dir: Path) -> str:
        """
        Collect worker statuses from all nodes and initiate checkpointing once all have reported.

        Args:
            dp_group_id: Data parallel group ID.
            node_id: Node ID within the group.
            worker_status: Status of each worker on the node.
            chkpnt_dir: Directory where the checkpoint will be saved.

        Returns:
            str: The checkpoint ID assigned to this checkpoint.

        Raises:
            RuntimeError: If the node reports its status more than once.
        """
        if self._local_mode:
            with self._checkpoint_lock:
                self._checkpoint_id_counter.value += 1
                checkpoint_id = f"chkpnt_{self._checkpoint_id_counter.value}"
                last_chunks = self._collect_last_chunks_from_queues()
                self._checkpoint_info[checkpoint_id] = {}

            self._start_checkpointing_local_mode(checkpoint_id, chkpnt_dir, last_chunks, worker_status)
            return checkpoint_id
        else:
            with self._checkpoint_lock:
                key = (dp_group_id, node_id)
                if key in self._nodes_reported:
                    raise RuntimeError(f"Node {node_id} in dp_group {dp_group_id} has already reported status.")

                # Assign a checkpoint_id if it hasn't been assigned yet
                if self._current_checkpoint_id is None:
                    # First node reporting, assign new checkpoint_id
                    self._checkpoint_id_counter.value += 1
                    checkpoint_id = f"chkpnt_{self._checkpoint_id_counter.value}"
                    self._current_checkpoint_id = checkpoint_id
                    self._checkpoint_info[checkpoint_id] = {}
                else:
                    # Checkpoint already in progress
                    checkpoint_id = self._current_checkpoint_id

                self._worker_statuses[key] = worker_status
                self._nodes_reported.add(key)

                if len(self._nodes_reported) == self._dp_groups * self._nodes_per_group:
                    # All nodes have reported; proceed to validation and checkpointing
                    worker_sample_ids = self._validate_checkpoint_state()

                    # Start checkpointing process
                    # By having this in the lock,
                    # we ensure we finish the checkpoint in-memory copy first before handling the next request.
                    self._start_checkpointing(checkpoint_id, worker_sample_ids, chkpnt_dir)

                    # Reset for next checkpoint, afterwards release the lock, allowing potentially for the next request.
                    self._worker_statuses = {}
                    self._nodes_reported = set()
                    self._current_checkpoint_id = None

            return checkpoint_id

    def _collect_last_chunks_from_queues(self) -> dict[int, Any]:
        """
        Collects the last chunks from each worker's queue in local mode.

        Returns:
            dict[int, Any]: A dictionary mapping worker_id to last_chunk.
        """
        if self._worker_queues is None:
            raise RuntimeError("Initiating local checkpoint but self._worker_queues is None.")

        last_chunks = {}
        for worker_id, queue in enumerate(self._worker_queues):
            try:
                last_chunks[worker_id] = queue.get(timeout=1)
            except Empty:
                # It might be we checkpoint before we even requested a chunk for each worker.
                last_chunks[worker_id] = None
                logger.debug(f"No last_chunk received from worker {worker_id}")

        return last_chunks

    def _start_checkpointing_local_mode(
        self, checkpoint_id: str, chkpnt_dir: Path, last_chunks: dict[int, Any], worker_status: list[int]
    ) -> None:
        """
        Start the checkpointing process in local mode using last_chunks.

        Args:
            checkpoint_id: Identifier for the checkpoint.
            chkpnt_dir: Directory where the checkpoint will be saved.
            last_chunks: The last chunk from each worker.
        """
        # TODO(MaxiBoether): Can we refactor this to somehow use the some logic as the server checkpointing?

        checkpoint_path = chkpnt_dir / checkpoint_id
        checkpoint_path.mkdir(parents=True, exist_ok=False)
        logger.debug(f"Saving checkpoint to {checkpoint_path}")

        for worker_id, status in enumerate(worker_status):
            chunk: SerializedResultChunk | ResultChunk
            chunk = last_chunks[worker_id]
            if chunk is not None:
                is_serialized = isinstance(chunk, SerializedResultChunk)
                if is_serialized:
                    chunk = dill.loads(chunk)
                chunk._samples_to_skip = worker_status[worker_id]
                chunk = dill.dumps(chunk) if is_serialized else chunk
            last_chunks[worker_id] = chunk

        logger.debug("Writing last chunks to disk.")
        with open(checkpoint_path / "last_chunks.pkl", "wb") as f:
            dill.dump({"_checkpoint_id_counter": self._checkpoint_id_counter.value, "last_chunks": last_chunks}, f)

        logger.debug("Writing QueryResult to disk.")
        self._query_result.to_cache(checkpoint_path / "query_result")
        logger.debug("Wrote local checkpoint.")
        self._checkpoint_info[checkpoint_id]["local"] = True

    def _validate_checkpoint_state(self) -> dict[tuple[int, int], int]:
        """
        Validates the state in the system before a checkpoint.

        Raises:
            RuntimeError: If validation fails.

        Returns:
            Dictionary with potentially fixed inconsistent worker state:
                For each (dp_group, worker_id), tells at which sample to continue.
        """
        # First check whether within each dp_group, all workers are at the same chunk
        chunk_per_worker: dict[tuple[int, int], list[int]] = {}
        for dp_group, node_dict in self._next_chunk.items():
            for node_id, worker_dict in node_dict.items():
                for worker_id, next_chunk in worker_dict.items():
                    key = (dp_group, worker_id)
                    if key not in chunk_per_worker:
                        chunk_per_worker[key] = []
                    chunk_per_worker[key].append(next_chunk)

        for (dp_group_id, worker_id), next_chunks in chunk_per_worker.items():
            if not len(set(next_chunks)) == 1:
                raise RuntimeError(f"Invalid checkpoint state: dp = {dp_group_id} next chunks = {next_chunks}")

        # Now we know that all workers are at the same chunk.
        # Next we check if roughly all workers are at the same sample.

        worker_statuses_per_worker: dict[tuple[int, int], list[int]] = {}
        for (dp_group_id, node_id), worker_status_list in self._worker_statuses.items():
            for worker_id, sample_idx in enumerate(worker_status_list):
                key = (dp_group_id, worker_id)
                if key not in worker_statuses_per_worker:
                    worker_statuses_per_worker[key] = []
                worker_statuses_per_worker[key].append(sample_idx)

        result = {}
        # Validate: Within the same dp group, each worker should roughly be at the same sample.
        for (dp_group_id, worker_id), sample_indices in worker_statuses_per_worker.items():
            min_idx = min(sample_indices)
            max_idx = max(sample_indices)
            if max_idx - min_idx > 5:
                logger.warning(
                    f"Worker {worker_id} in dp_group {dp_group_id} has inconsistent"
                    + f"sample indice (drift is {max_idx - min_idx}).\nsample_indices = {sample_indices}"
                )
            result[(dp_group_id, worker_id)] = max_idx

        return result

    def _start_checkpointing(
        self, checkpoint_id: str, worker_sample_ids: dict[tuple[int, int], int], chkpnt_dir: Path
    ) -> None:
        """
        Start the checkpointing process in a separate process.

        The checkpointing process will create a deepcopy of the current state and persist it to disk.

        Args:
            checkpoint_id: Identifier for the checkpoint.
            chkpnt_dir: Directory where the checkpoint will be saved.
        """
        logger.debug("Copying the ChunkDistributor state.")

        state_to_save = {
            "chunk_cache": deepcopy(self._chunk_cache),
            "chunk_usage": deepcopy(self._chunk_usage),
            "next_chunk": deepcopy(self._next_chunk),
            "_dp_groups": self._dp_groups,
            "_num_workers": self._num_workers,
            "_nodes_per_group": self._nodes_per_group,
        }

        logger.debug("Copying the QueryResult.")

        _lock, _index = self._query_result._lock, self._query_result._index
        del self._query_result._lock
        del self._query_result._index
        query_result_copy = deepcopy(self._query_result)
        self._query_result._lock = _lock
        self._query_result._index = _index
        query_result_copy._lock = _lock
        query_result_copy._index = _index

        logger.debug("Spinning up the persisting process.")

        self._checkpoint_info[checkpoint_id]["process"] = None
        self._checkpoint_info[checkpoint_id]["status"] = "in_progress"

        p = mp.Process(
            target=self._persist_checkpoint_process,
            args=(
                checkpoint_id,
                chkpnt_dir,
                state_to_save,
                query_result_copy,
                worker_sample_ids,
            ),
        )
        self._checkpoint_info[checkpoint_id]["process"] = p
        p.start()

    @staticmethod
    def _persist_checkpoint_process(
        checkpoint_id: str,
        chkpnt_dir: Path,
        state_to_save: dict[str, Any],
        query_result_copy: QueryResult,
        worker_sample_ids: dict[tuple[int, int], int],
    ) -> None:
        """
        Runs in a separate process to persist the checkpoint.

        Args:
            checkpoint_id: Identifier for the checkpoint.
            chkpnt_dir: Directory where the checkpoint will be saved.
            state_to_save: The state to save.
            query_result_copy: The QueryResult to save.
        """
        try:
            checkpoint_path = chkpnt_dir / checkpoint_id
            checkpoint_path.mkdir(parents=True, exist_ok=False)

            logger.debug("Adjusting the state.")
            # Move next chunk 1 backwards (because on replay, we will need to hand out the current chunk again)
            # Also, integrate worker_sample_ids
            for dp_group in range(state_to_save["_dp_groups"]):
                for node in range(state_to_save["_nodes_per_group"]):
                    for worker_id in range(state_to_save["_num_workers"]):
                        chunk: ResultChunk | SerializedResultChunk

                        next_chunk = (
                            state_to_save["next_chunk"][dp_group][node][worker_id] - state_to_save["_num_workers"]
                        )
                        if next_chunk >= 0:
                            state_to_save["next_chunk"][dp_group][node][worker_id] = next_chunk
                            state_to_save["chunk_usage"][dp_group][next_chunk] = 0
                            chunk = state_to_save["chunk_cache"][dp_group][next_chunk]
                            is_serialized = isinstance(chunk, SerializedResultChunk)
                            if is_serialized:
                                chunk = dill.loads(chunk)
                            chunk._samples_to_skip = worker_sample_ids[(dp_group, worker_id)]
                            chunk = dill.dumps(chunk) if is_serialized else chunk
                            state_to_save["chunk_cache"][dp_group][next_chunk] = chunk
                        else:
                            logger.debug(
                                f"For dp={dp_group}, node={node}, w={worker_id}, next_chunk = {next_chunk}"
                                + "\nLikely, the checkpoint is created before all workers have requested a chunk" + "- otherwise this indicates an error..."
                            )

            logger.info("Checkpointing the state (without QueryResult).")
            with open(checkpoint_path / "chunk_distributor_state.pkl", "wb") as f:
                dill.dump(state_to_save, f, protocol=dill.HIGHEST_PROTOCOL)

            logger.info("Checkpointing the QueryResult.")
            qr_path = checkpoint_path / "query_result"
            qr_path.mkdir(exist_ok=False)
            query_result_copy.to_cache(qr_path)

        except Exception as e:
            logger.error(f"Error during checkpointing: {e}")
            raise e

    def checkpoint_completed(self, checkpoint_id: str, on_disk: bool) -> bool:
        """
        Check if the checkpoint has been completed.

        Args:
            checkpoint_id: Identifier for the checkpoint.
            on_disk: If True, returns True only if the checkpoint has been fully written to disk.

        Returns:
            bool: True if checkpoint is completed based on the `on_disk` parameter, False otherwise.

        Raises:
            RuntimeError: If the checkpoint process failed.
        """
        checkpoint_info = self._checkpoint_info.get(checkpoint_id, None)
        if checkpoint_info is None or not checkpoint_info:
            return False  # no checkpoint in progress

        if checkpoint_info.get("local", False):
            return True  # shortcut since local checkpoints are done after they are in the map

        if "process" not in checkpoint_info or checkpoint_info["process"] is None:
            return False  # No checkpoint in progress

        if not on_disk:
            # The in-memory copy has been made as the process has started
            return True

        process = checkpoint_info["process"]

        # Check if the process is alive
        if process.is_alive():
            return False  # Still in progress
        else:
            # Process has finished
            process.join()
            if process.exitcode != 0:
                raise RuntimeError(f"Checkpoint {checkpoint_id} failed with exit code {process.exitcode}.")
            return True

    @classmethod
    def from_checkpoint(
        cls,
        chkpnt_dir: Path,
        checkpoint_id: str,
        job_id: str,
    ) -> "ChunkDistributor":
        """
        Create a ChunkDistributor instance from a checkpoint directory.

        Args:
            chkpnt_dir: Directory where the checkpoint is stored.
            job_id: Unique identifier for the job.

        Returns:
            ChunkDistributor: A new ChunkDistributor instance with state restored from the checkpoint.

        Raises:
            FileNotFoundError: If necessary checkpoint files are not found.
            Exception: If there is an error during deserialization.
        """
        # Determine whether the checkpoint is from local mode or server mode
        checkpoint_dir = chkpnt_dir / checkpoint_id
        logger.debug(f"Loading checkpoint from {checkpoint_dir}")
        if (checkpoint_dir / "last_chunks.pkl").exists():
            return cls._from_local_checkpoint(checkpoint_dir, job_id)
        else:
            return cls._from_server_checkpoint(checkpoint_dir, job_id)

    @classmethod
    def _from_server_checkpoint(
        cls,
        chkpnt_dir: Path,
        job_id: str,
    ) -> "ChunkDistributor":
        """
        Restore ChunkDistributor from a server mode checkpoint.

        Args:
            chkpnt_dir: Directory where the checkpoint is stored.
            job_id: Unique identifier for the job.

        Returns:
            ChunkDistributor: The restored ChunkDistributor instance.

        Raises:
            FileNotFoundError: If necessary checkpoint files are not found.
        """
        # Load the chunk distributor state
        logger.debug("Loading ChunkDistributor state.")
        checkpoint_state_path = chkpnt_dir / "chunk_distributor_state.pkl"
        if not checkpoint_state_path.exists():
            raise FileNotFoundError(f"Checkpoint state file not found at {checkpoint_state_path}")

        with open(checkpoint_state_path, "rb") as f:
            state = dill.load(f)

        logger.debug("Loading QueryResult.")
        # Load the QueryResult
        query_result_path = chkpnt_dir / "query_result"
        if not query_result_path.exists():
            raise FileNotFoundError(f"QueryResult checkpoint not found at {query_result_path}")

        query_result = QueryResult.from_cache(query_result_path)

        logger.debug("Instantiating class.")

        chunk_distributor = cls(
            dp_groups=state["_dp_groups"],
            nodes_per_group=state["_nodes_per_group"],
            num_workers=state["_num_workers"],
            query_result=query_result,
            job_id=job_id,
        )

        # Restore the state
        chunk_distributor._chunk_cache = state["chunk_cache"]
        chunk_distributor._chunk_usage = state["chunk_usage"]
        chunk_distributor._next_chunk = state["next_chunk"]

        # Reset checkpoint-related attributes
        chunk_distributor._checkpoint_info = {}
        chunk_distributor._current_checkpoint_id = None
        chunk_distributor._worker_statuses = {}
        chunk_distributor._nodes_reported = set()
        chunk_distributor._checkpoint_id_counter.value = state["_checkpoint_id_counter"]

        return chunk_distributor

    @classmethod
    def _from_local_checkpoint(
        cls,
        chkpnt_dir: Path,
        job_id: str,
    ) -> "ChunkDistributor":
        """
        Restore ChunkDistributor from a local mode checkpoint.

        Args:
            chkpnt_dir: Directory where the checkpoint is stored.
            job_id: Unique identifier for the job.

        Returns:
            ChunkDistributor: The restored ChunkDistributor instance.

        Raises:
            FileNotFoundError: If necessary checkpoint files are not found.
        """
        logger.debug("Loading last chunks.")
        last_chunks_path = chkpnt_dir / "last_chunks.pkl"
        if not last_chunks_path.exists():
            raise FileNotFoundError(f"last_chunks.pkl not found at {last_chunks_path}")

        with open(last_chunks_path, "rb") as f:
            state = dill.load(f)
            last_chunks = state["last_chunks"]

        logger.debug("Loading QueryResult.")

        query_result_path = chkpnt_dir / "query_result"
        if not query_result_path.exists():
            raise FileNotFoundError(f"QueryResult checkpoint not found at {query_result_path}")

        query_result = QueryResult.from_cache(query_result_path)
        logger.debug("Instantiating class..")

        num_workers = len(last_chunks)
        chunk_distributor = cls(
            dp_groups=1,
            nodes_per_group=1,
            num_workers=num_workers,
            query_result=query_result,
            job_id=job_id,
        )

        # Initialize worker queues
        chunk_distributor._worker_queues = [mp.Queue(maxsize=1) for _ in range(num_workers)]
        chunk_distributor._checkpoint_id_counter.value = state["_checkpoint_id_counter"]

        # Put the last chunks into the queues
        for worker_id, chunk in last_chunks.items():
            queue = chunk_distributor._worker_queues[worker_id]
            if chunk is not None:
                queue.put(chunk)
                chunk_distributor._return_from_queue[worker_id] = True
            else:
                chunk_distributor._return_from_queue[worker_id] = False

        return chunk_distributor
