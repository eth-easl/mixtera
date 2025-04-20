import multiprocessing as mp
import os
import re
import shutil
import threading
from copy import deepcopy
from pathlib import Path
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
        cached_query: Path | None = None,
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
        self._query_result.stop_on_none = False
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
        self._cached_query = cached_query

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

        guaranteed_server = self._dp_groups > 1 or self._nodes_per_group > 1

        if os.getpid() != self._constructor_pid and guaranteed_server:
            raise RuntimeError(
                f"We seem to have forked ({os.getpid()} vs {self._constructor_pid})"
                + "but we're definitely in server mode."
            )

        if guaranteed_server and deserialize:
            logger.warning(
                "You are using Mixtera with caching, but do not serialize the chunks." + "This is unexpected behavior."
            )

        next_chunk_id = self._next_chunk[dp_group][node_id][worker_id]
        # The data parallel groups operate on different chunks, i.e., chunk 1 is different for dp 0 and 1
        if next_chunk_id not in self._chunk_cache[dp_group]:
            # Potentially useful debug log
            # logger.debug(
            #    f"Fetching chunk {next_chunk_id} for dp_group {dp_group} /
            #  # node {node_id} requested by worker {worker_id} from QueryResult.")

            # Fetch new chunk from query result and put into cache
            if (chunk_to_return := next(self._query_result)) is None:
                raise StopIteration

            serialized_chunk = dill.dumps(chunk_to_return)
            self._chunk_cache[dp_group][next_chunk_id] = serialized_chunk
            self._chunk_usage[dp_group][next_chunk_id] = 0

            if not deserialize:
                chunk_to_return = serialized_chunk
        else:
            # Potentially useful debug log
            # logger.debug(f"Fetching chunk {next_chunk_id} for dp_group {dp_group} /
            #  node {node_id} requested by worker {worker_id} from cache.")
            # Load from cache
            chunk_to_return = self._chunk_cache[dp_group][next_chunk_id]  # always serialized in cache
            if deserialize:
                chunk_to_return = dill.loads(chunk_to_return)

        # Increment usage count for this chunk
        self._chunk_usage[dp_group][next_chunk_id] += 1

        # Check if all nodes have received this chunk
        if self._chunk_usage[dp_group][next_chunk_id] >= self._nodes_per_group:
            if (chunk_to_delete := next_chunk_id - self._num_workers) >= 0:
                # Potentially useful debug log
                # logger.debug(f"[{os.getpid()}/{threading.get_native_id()}]
                # Purging chunk {chunk_to_delete} " + f"for dp_group {dp_group} from cache.")
                if chunk_to_delete in self._chunk_cache[dp_group]:
                    # Delete the previous chunk as all nodes have now received the next chunk
                    # In regular runs, this if is always True.
                    # However, when continuing from a checkpoint, the previous item was purged already.
                    del self._chunk_cache[dp_group][chunk_to_delete]
                    del self._chunk_usage[dp_group][chunk_to_delete]

        # We don't increment by 1 but instead by num_workers, because otherwise
        # we get an overlap between workers after the first chunk
        self._next_chunk[dp_group][node_id][worker_id] += self._num_workers
        # logger.debug(f"Next chunk for dp_group {dp_group} / node {node_id}
        # now is {self._next_chunk[dp_group][node_id][worker_id]}.")
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

    def checkpoint(
        self, dp_group_id: int, node_id: int, worker_status: list[int], chkpnt_dir: Path, server: bool
    ) -> None | str:
        """
        Initiate a checkpoint operation for the specified data parallel group and node.

        This method collects worker statuses from all nodes and initiates checkpointing once all have reported.
        Checkpointing involves saving the current state of the data loaders and the chunk distributor
        so that training can be resumed later from the same point.

        Args:
            dp_group_id (int): Data parallel group ID.
            node_id (int): Node ID within the group.
            worker_status (list[int]): List containing the current status (sample indices) of each data loader worker.
            chkpnt_dir (Path): Directory where the checkpoint will be saved.
            server (bool): Indicates whether the checkpoint is being called from a Mixtera server or in local mode

        Returns:
            str or None: The checkpoint ID assigned to this checkpoint if checkpointing has started,
                         None otherwise.

        Raises:
            RuntimeError: If the node reports its status more than once or if checkpointing is not supported
                          in the current configuration.
            NotImplementedError: If checkpointing is not supported for the current setup.
        """
        if self._dp_groups == 1 and self._nodes_per_group == 1 and self._og_num_workers > 0 and not server:
            raise NotImplementedError(
                "Checkpointing not supported for single-node, single GPU, multi dataloader worker non-server training"
            )
            # In this case, we fork on each worker, copying the ChunkDistributor.
            # The chunk_cache is not shared between the instances.
            # Hence, we do not have access to the chunks we handed out in the main process.
            # We need to share the nested dicts between the processes,
            # for which we need to implement a cache using shared_memory.
            # A multiprocessing.manager.dict() is very slow, so we should try one of the high performance alternatives.

        if os.getpid() != self._constructor_pid:
            # Should not happen, let's catch it anyways.
            raise RuntimeError("We forked - this could happen when calling this inside a dataloader worker.")

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

    def _validate_checkpoint_state(self) -> dict[tuple[int, int], int]:
        """
        Validate the system state before performing a checkpoint.

        This method checks whether within each data parallel group, all workers are at the same chunk
        and roughly at the same sample index. It returns a mapping indicating the sample index
        each worker should start from upon resuming.

        Raises:
            RuntimeError: If workers within the same data parallel group are at inconsistent states.

        Returns:
            dict[tuple[int, int], int]: A dictionary mapping (dp_group_id, worker_id) to the sample index
                                        at which to continue.
        """
        # First check whether within each dp_group, all workers are at the same chunk
        chunk_per_worker: dict[tuple[int, int], list[int]] = {}
        for dp_group, node_dict in self._next_chunk.items():
            for _, worker_dict in node_dict.items():
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
        for (dp_group_id, _), worker_status_list in self._worker_statuses.items():
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
            checkpoint_id (str): Identifier for the checkpoint.
            worker_sample_ids (dict[tuple[int, int], int]): Mapping of worker sample indices to use.
            chkpnt_dir (Path): Directory where the checkpoint will be saved.
        """

        logger.debug("Copying the ChunkDistributor state.")

        state_to_save = {
            "chunk_cache": deepcopy(self._chunk_cache),
            "chunk_usage": deepcopy(self._chunk_usage),
            "next_chunk": deepcopy(self._next_chunk),
            "_dp_groups": self._dp_groups,
            "_num_workers": self._num_workers,
            "_nodes_per_group": self._nodes_per_group,
            "_checkpoint_id_counter": self._checkpoint_id_counter.value,
        }

        # The fundamental issue is that with multiprocessing.Spawn,
        # the QueryResult will be pickled when we start another process.
        # This means we cannot store the result in another process - the expensive operation is the pickling.
        directory_to_copy = None
        if self._checkpoint_id_counter.value == 1:
            logger.debug("This is the first checkpoint.")
            if self._cached_query is not None:
                logger.debug("Copying cached QueryResult to setup initial checkpoint in subprocess.")
                directory_to_copy = self._cached_query
            else:
                logger.debug("Query has not been cached - pickling now. This may take some time.")
                self._query_result.to_cache(chkpnt_dir / "queryresult")
            logger.debug("Handled pickling of QueryResult.")

        if hasattr(self._query_result, "_mixture") and self._query_result._mixture is not None:
            self._query_result._mixture.write_logs()

        logger.debug("Preparing the QueryResult properties.")

        mixture_log = deepcopy(self._query_result._mixture_log)

        if (
            len(mixture_log) > 0
            and hasattr(mixture_log[-1][1], "_mixing_alg")
            and self._query_result._mixture is not None
            and hasattr(self._query_result._mixture, "_mixing_alg")
        ):
            # if we have a mixing alg, that might have been updated.
            # we need to checkpoint that state as well, so we have to update the log
            logger.debug("Updating the mixing alg. Previous alg:")
            logger.debug(str(mixture_log[-1][1]._mixing_alg))
            mixture_log[-1][1]._mixing_alg = deepcopy(self._query_result._mixture._mixing_alg)
            logger.debug("Updated alg:")
            logger.debug(str(mixture_log[-1][1]._mixing_alg))

        returns_gen = self._query_result._num_returns_gen

        logger.debug("Spinning up the persisting process.")

        self._checkpoint_info[checkpoint_id]["process"] = None
        self._checkpoint_info[checkpoint_id]["status"] = "in_progress"

        p = mp.Process(
            target=self._persist_checkpoint_process,
            args=(
                checkpoint_id,
                chkpnt_dir,
                state_to_save,
                mixture_log,
                returns_gen,
                worker_sample_ids,
                directory_to_copy,
            ),
        )
        self._checkpoint_info[checkpoint_id]["process"] = p
        p.start()

    @staticmethod
    def _persist_checkpoint_process(
        checkpoint_id: str,
        chkpnt_dir: Path,
        state_to_save: dict[str, Any],
        mixture_log: list[tuple[int, Any]],
        num_returns_gen: int,
        worker_sample_ids: dict[tuple[int, int], int],
        directory_to_copy: Path | None,
    ) -> None:
        """
        Persist the checkpoint to disk in a separate process.

        This method adjusts the state as needed, serializes it, and saves it to the specified directory.
        It also saves the QueryResult state for resuming.

        Args:
            checkpoint_id (str): Identifier for the checkpoint.
            chkpnt_dir (Path): Directory where the checkpoint will be saved.
            state_to_save (dict[str, Any]): The state dictionary to save.
            query_result_copy (QueryResult): The QueryResult to save.
            worker_sample_ids (dict[tuple[int, int], int]): Mapping of worker sample indices per worker.

        Raises:
            Exception: If an error occurs during checkpointing.
        """

        try:
            if directory_to_copy is not None:
                logger.debug("Starting to copy directory in process.")
                shutil.copytree(directory_to_copy, chkpnt_dir / "queryresult")
                logger.debug("Directory copied..")

            assert (
                chkpnt_dir / "queryresult"
            ).exists(), f"QueryResult directory {(chkpnt_dir / 'queryresult')} should exist now!"

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
                                + "\nLikely, the checkpoint is created before all workers have requested a chunk"
                                + "- otherwise this indicates an error..."
                            )

            logger.info("Checkpointing the state (without QueryResult).")
            with open(checkpoint_path / "chunk_distributor_state.pkl", "wb") as f:
                dill.dump(state_to_save, f, protocol=dill.HIGHEST_PROTOCOL)

            logger.info(f"Checkpointing the mixture log and num returns with num returns = {num_returns_gen}")
            with open(checkpoint_path / "query_result_state.pkl", "wb") as f:
                dill.dump(
                    {"num_returns_gen": num_returns_gen, "mixture_log": mixture_log}, f, protocol=dill.HIGHEST_PROTOCOL
                )

            logger.debug("Wrote checkpoint.")
        except Exception as e:
            logger.error(f"Error during checkpointing: {e}")
            raise e

    def checkpoint_completed(self, checkpoint_id: str, on_disk: bool) -> bool:
        """
        Check if the checkpoint operation has been completed.

        Depending on the `on_disk` parameter, this method checks whether the checkpointing process
        has started (if `on_disk` is False) or whether it has finished and the data is fully written to disk
        (if `on_disk` is True).

        Args:
            checkpoint_id (str): Identifier for the checkpoint.
            on_disk (bool): If True, returns True only if the checkpoint has been fully written to disk.
                            If False, returns True as soon as the checkpoint is stored in memory.

        Returns:
            bool: True if checkpoint is completed based on the `on_disk` parameter, False otherwise.

        Raises:
            RuntimeError: If the checkpoint process failed.
        """
        checkpoint_info = self._checkpoint_info.get(checkpoint_id, None)
        if checkpoint_info is None or not checkpoint_info:
            return False  # no checkpoint in progress

        if "process" not in checkpoint_info or checkpoint_info["process"] is None:
            return False  # No checkpoint in progress

        if not on_disk:
            # The in-memory copy has been made as the process has started
            return True

        process = checkpoint_info["process"]

        # Check if the process is alive
        if process.is_alive():
            return False  # Still in progress

        # Process has finished
        process.join()
        if process.exitcode != 0:
            raise RuntimeError(f"Checkpoint {checkpoint_id} failed with exit code {process.exitcode}.")
        return True

    @classmethod
    def from_checkpoint(
        cls, chkpnt_dir: Path, checkpoint_id: str, job_id: str, query_log_dir: Path | None
    ) -> "ChunkDistributor":
        """
        Create a ChunkDistributor instance from a saved checkpoint.

        This method restores the state of the ChunkDistributor and the associated QueryResult
        from the specified checkpoint directory, allowing for resuming operations from where they left off.

        Args:
            chkpnt_dir (Path): Directory where the checkpoint is stored.
            checkpoint_id (str): Identifier for the checkpoint.
            job_id (str): Unique identifier for the job.

        Returns:
            ChunkDistributor: A new ChunkDistributor instance with state restored from the checkpoint.

        Raises:
            FileNotFoundError: If necessary checkpoint files are not found.
            Exception: If there is an error during deserialization.
        """
        # Determine whether the checkpoint is from local mode or server mode
        chkpnt_dir = chkpnt_dir / checkpoint_id
        query_dir = chkpnt_dir.parent / "queryresult"

        logger.debug(f"Loading checkpoint from {chkpnt_dir}")
        logger.debug("Loading ChunkDistributor state.")
        checkpoint_state_path = chkpnt_dir / "chunk_distributor_state.pkl"
        query_result_state_path = chkpnt_dir / "query_result_state.pkl"

        if not checkpoint_state_path.exists():
            raise FileNotFoundError(f"Checkpoint state file not found at {checkpoint_state_path}")
        if not query_result_state_path.exists():
            raise FileNotFoundError(f"QueryResult state file not found at {query_result_state_path}")
        if not query_dir.exists():
            raise FileNotFoundError(f"QueryResult directory not found at {query_dir}")

        logger.debug("Loading states.")
        with open(checkpoint_state_path, "rb") as f:
            state = dill.load(f)
        with open(query_result_state_path, "rb") as f:
            query_result_state = dill.load(f)

        logger.debug("Loading QueryResult.")
        query_result = QueryResult.from_cache(query_dir, replay=False)
        query_result._mixture_log = query_result_state["mixture_log"]
        query_result._query_log_dir = query_log_dir
        if query_log_dir is not None:
            query_log_dir.mkdir(exist_ok=True)
        query_result.replay(query_result_state["num_returns_gen"])

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

        # Update the _checkpoint_id_counter.value to avoid overwriting existing checkpoints
        checkpoint_dirs = chkpnt_dir.parent.glob("chkpnt_*")
        checkpoint_numbers = [chunk_distributor._checkpoint_id_counter.value]
        pattern = re.compile(r"chkpnt_(\d+)")
        for checkpoint_dir in checkpoint_dirs:
            match = pattern.match(checkpoint_dir.name)
            if match:
                checkpoint_numbers.append(int(match.group(1)))
        if checkpoint_numbers:
            chunk_distributor._checkpoint_id_counter.value = max(checkpoint_numbers)

        logger.debug("Loaded Distributor from checkpoint.")

        return chunk_distributor
