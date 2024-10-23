import multiprocessing as mp
import os
import sys
import threading
from multiprocessing import shared_memory
from typing import Any, Generator

import numpy as np
from loguru import logger
from mixtera.core.client import MixteraClient
from mixtera.core.client.mixtera_client import QueryExecutionArgs, ResultStreamingArgs
from mixtera.core.query import Query
from mixtera.core.query.mixture import Mixture
from numpy.typing import NDArray
from torch.utils.data import IterableDataset, get_worker_info  # pylint: disable=import-error,no-name-in-module


class MixteraTorchDataset(IterableDataset):
    def __init__(
        self,
        client: MixteraClient,
        query: Query,
        query_execution_args: QueryExecutionArgs,
        result_streaming_args: ResultStreamingArgs,
        execute_query: bool = True,
    ):
        # TODO(#63): This needs to be passed information on transformation, e.g., tokenization functions etc.
        # Alternative: Let people inherit from this.
        self._client = client
        self._query = query
        self._res_str_args = result_streaming_args
        self._query_execution_args = query_execution_args
        self._status_shm: shared_memory.SharedMemory | None = None
        self._comp_shm: shared_memory.SharedMemory | None = None

        assert self._dp_group_id < query_execution_args.dp_groups
        assert self._node_id < query_execution_args.nodes_per_group

        self._init_status_shm()

        if self._node_id == 0 and self._dp_group_id == 0 and execute_query:
            logger.info(
                f"[{os.getpid()}/{threading.get_native_id()}] "
                + "Since this is node 0 in data parallel group 0, executing query!"
            )
            # Execute query on primary node pre-fork, to share the results among all forked workers
            self._client.execute_query(query, self._query_execution_args)

    def _init_status_shm(self) -> None:
        # This function intiailizes a shared memory segment
        # in which all workers write at which sample in the current chunk they are.
        # At the Mixtera server, we keep track of which chunk they are at.
        # This way, we can on checkpoint remember at which sample to continue at the worker.

        assert mp.current_process().name == "MainProcess", (
            "The checkpointing logic requires the dataset to be constructured in the main process\n"
            + f"This process is {mp.current_process().name}"
        )

        # Initialize status_array, indicating for each worker which sample they are currently handling in the chunk
        status_dtype = np.int64
        array_shape = (self.num_workers,)  # One index per worker
        status_array_size = self.num_workers * status_dtype(0).nbytes
        status_shm_name = f"mts_{self._query.job_id}"
        status_shm_name = status_shm_name[:30] if sys.platform == "darwin" else status_shm_name
        self._status_shm = shared_memory.SharedMemory(name=status_shm_name, create=True, size=status_array_size)
        status_array: NDArray[np.int64] = np.ndarray(array_shape, dtype=status_dtype, buffer=self._status_shm.buf)
        status_array[:] = 0  # Initialize indices to zero

        # Initialize competion_array, indicating for each worker whether they have finished training
        # Used to trigger cleanup at the end
        comp_dtype = np.int8
        comp_array_size = self.num_workers * comp_dtype(0).nbytes
        comp_shm_name = f"mtc_{self._query.job_id}"
        comp_shm_name = comp_shm_name[:30] if sys.platform == "darwin" else comp_shm_name
        self._comp_shm = shared_memory.SharedMemory(name=comp_shm_name, create=True, size=comp_array_size)
        completion_array: NDArray[np.int8] = np.ndarray(array_shape, dtype=comp_dtype, buffer=self._comp_shm.buf)
        completion_array[:] = 0  # Initialize indices to zero

        # We cannot share the arrays via self, then the arrays get un-synced after spawning/forking.

        self.completion_lock = mp.Lock()
        self.status_shm_name = status_shm_name
        self.comp_shm_name = comp_shm_name

        logger.debug(f"[Process {os.getpid()}] Created shared memory objects for {self.num_workers} workers")

    @property
    def _dp_group_id(self) -> int:
        return self._res_str_args.dp_group_id

    @property
    def _node_id(self) -> int:
        return self._res_str_args.node_id

    @property
    def _mixture(self) -> Mixture:
        return self._query_execution_args.mixture

    @property
    def num_workers(self) -> int:
        return max(self._query_execution_args.num_workers, 1)

    @property
    def worker_id(self) -> int:
        worker_info = get_worker_info()
        if worker_info is None:
            # Non-multithreaded data loading. We use worker_id 0.
            worker_id = 0
        else:
            worker_id = worker_info.id

        assert worker_id < self.num_workers, f"Number of workers was invalid: {worker_id} vs {self.num_workers}"

        return worker_id

    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError("This is just overwritten to satify pylint.")

    def _cleanup_shared_memory(self, unlink: bool) -> None:
        # Attempt to close and unlink shared memory segments
        try:
            any_close = False
            if self._status_shm is not None:
                any_close = True
                self._status_shm.close()
                if unlink:
                    self._status_shm.unlink()
                self._status_shm = None

            if self._comp_shm is not None:
                any_close = True
                self._comp_shm.close()
                if unlink:
                    self._comp_shm.unlink()
                self._comp_shm = None

            if any_close:
                logger.info(f"[Worker {self.worker_id}] Shared memory closed.")
                if unlink:
                    logger.info(f"[Worker {self.worker_id}] Shared memory unlinked.")

        except FileNotFoundError as e:
            logger.error(
                f"FileNotFoundError during shared memory cleanup: {e}\n"
                + "This indicates multiple workers cleaned up the shm, which should not happen."
            )
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error(f"Error during shared memory cleanup: {e}")

    @property
    def worker_status(self) -> list[int] | None:
        if self._status_shm is None:
            return None

        try:
            return np.ndarray((self.num_workers,), dtype=np.int64, buffer=self._status_shm.buf).tolist()
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error(f"Error while fetching worker status from shm: {e}")
            return None

    def __iter__(self) -> Generator[str, None, None]:
        assert self._comp_shm is not None and self._status_shm is not None, "SharedMemory objects are None."

        try:
            # Initialize data structures to share status
            # Cannot share those in self - that does not work across spawn/fork
            status_array: NDArray[np.int64] = np.ndarray(
                (self.num_workers,), dtype=np.int64, buffer=self._status_shm.buf
            )
            completion_array: NDArray[np.int8] = np.ndarray(
                (self.num_workers,), dtype=np.int8, buffer=self._comp_shm.buf
            )
            status_array[self.worker_id] = 0
            completion_array[self.worker_id] = 0
            self._res_str_args.worker_id = self.worker_id

            for sample_chnk_idx, sample in self._client.stream_results(self._res_str_args):
                status_array[self.worker_id] = sample_chnk_idx
                yield sample

            with self.completion_lock:
                completion_array[self.worker_id] = 1
                if np.all(completion_array == 1):
                    logger.debug(f"[Worker {self.worker_id}] All data loader workers are done - will unlink.")
                    logger.debug(self.worker_status)
                    self._cleanup_shared_memory(True)
                else:
                    logger.debug(
                        f"[Worker {self.worker_id}] Only {np.sum(completion_array)}/{self.num_workers} are done."
                        + " Just cleaning up this worker."
                    )
                    self._cleanup_shared_memory(False)
        finally:
            self._cleanup_shared_memory(True)
