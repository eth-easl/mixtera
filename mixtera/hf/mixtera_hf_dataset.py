from copy import deepcopy
from multiprocessing import shared_memory
from multiprocessing.synchronize import Lock as LockT
from typing import Any, Generator, Tuple

import datasets
import numpy as np
from loguru import logger
from mixtera.core.client import MixteraClient
from mixtera.core.client.mixtera_client import QueryExecutionArgs, ResultStreamingArgs
from mixtera.core.query import Query
from mixtera.torch import MixteraTorchDataset

datasets.logging.set_verbosity_debug()


class _MixteraHFIterable(MixteraTorchDataset, datasets.iterable_dataset._BaseExamplesIterable):
    def __init__(
        self,
        client: MixteraClient,
        query: Query,
        query_execution_args: QueryExecutionArgs,
        result_streaming_args: ResultStreamingArgs,
        _shard_call_count: int = 0,
        _status_shm: shared_memory.SharedMemory | None = None,
        _comp_shm: shared_memory.SharedMemory | None = None,
        completion_lock: LockT | None = None,
    ):
        MixteraTorchDataset.__init__(
            self,
            client,
            query,
            query_execution_args,
            result_streaming_args,
            execute_query=_shard_call_count == 0,
            _status_shm=_status_shm,
            _comp_shm=_comp_shm,
            completion_lock=completion_lock,
        )
        datasets.iterable_dataset._BaseExamplesIterable.__init__(self)
        self._shard_call_count = _shard_call_count

    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError("This is just overwritten to satify pylint.")

    def _init_state_dict(self) -> dict:
        logger.info("_init_state_dict called.")
        return {"key": "random item to make huggingface happy"}

    def shuffle_data_sources(self, generator: np.random.Generator) -> datasets.iterable_dataset._BaseExamplesIterable:
        del generator
        logger.info("shuffle_data_sources called.")
        return self

    @property
    def n_shards(self) -> int:
        # HF requires us to set some number.
        return max(self._query_execution_args.num_workers, 1) * max(self._query_execution_args.dp_groups, 1) * 8

    @property
    def worker_id(self) -> int:
        assert self._shard_call_count > 0, "shard_data_sources should have been called - something went wrong."
        return self._res_str_args.worker_id

    def shard_data_sources(self, worker_id: int, num_workers: int) -> "_MixteraHFIterable":
        logger.debug(f"shard_data_sources called with worker_id={worker_id} and num_workers={num_workers}")
        # This is called in two cases:
        # On each dp node with num_workers = number of dp nodes
        # On each dp nodes with num_workers = num data loading workers IF num_workers > 0

        assert (
            num_workers == max(self._query_execution_args.num_workers, 1)
            or num_workers == self._query_execution_args.dp_groups
        ), (
            f"num_workers = {num_workers} != query.num_workers ="
            + f"{max(self._query_execution_args.num_workers,1)} defined at query execution."
        )

        res_args = deepcopy(self._res_str_args)
        res_args.worker_id = worker_id
        return _MixteraHFIterable(
            self._client,
            self._query,
            self._query_execution_args,
            res_args,
            _shard_call_count=self._shard_call_count + 1,
            _status_shm=self._status_shm,
            _comp_shm=self._comp_shm,
            completion_lock=self.completion_lock,
        )

    def validate_state(self) -> None:
        assert self._shard_call_count > 0, (
            f"[{self._dp_group_id}-{self._node_id}-{self.worker_id}]"
            + "shard_data_sources should have been called - something went wrong."
            + f"torch worker id = {MixteraTorchDataset.worker_id.fget(self)}, "
            + f"self.worker_id = {self.worker_id}"
        )
        assert self._shard_call_count <= 2, f"self._shard_call_count = {self._shard_call_count} > 2"

        if self._shard_call_count == 1:
            # we are training either dp > 1 with 0 workers or dp=1 with > n workers
            if self._query_execution_args.num_workers == 0:
                # In this case, we need to fix our worker ID becaues it's equal to the dp ID instead of worker id
                assert (
                    self.worker_id == self._dp_group_id
                ), f"self.worker_id = {self.worker_id} should be self._dp_group_id = {self._dp_group_id}"
                self._res_str_args.worker_id = 0

            assert (
                MixteraTorchDataset.worker_id.fget(self) == self.worker_id
            ), f"torch worker id = {MixteraTorchDataset.worker_id.fget(self)} != self.worker_id = {self.worker_id}"

    def __iter__(self) -> Generator[Tuple[str, dict], None, None]:
        datasets.logging.set_verbosity_debug()
        self.validate_state()
        idx = -1
        for idx, sample in enumerate(MixteraTorchDataset.__iter__(self)):
            yield (f"{self._dp_group_id}-{self._node_id}-{self.worker_id}-{idx}", {"text": sample})

        logger.info(f"[{self._dp_group_id}-{self._node_id}-{self.worker_id}] Reached EOS after sample {idx}")


class MixteraHFDataset(datasets.IterableDataset):
    def __init__(
        self,
        client: MixteraClient,
        query: Query,
        query_execution_args: QueryExecutionArgs,
        result_streaming_args: ResultStreamingArgs,
    ):
        super().__init__(_MixteraHFIterable(client, query, query_execution_args, result_streaming_args))
        self.info.features = datasets.Features({"text": datasets.Value(dtype="string")})
        self._ex_iterable: _MixteraHFIterable

    def __iter__(self) -> Generator[Any | dict, Any, None]:
        # We wrap IterableDataset.__iter__ to do some state assertions
        assert isinstance(self._ex_iterable, _MixteraHFIterable)
        if self._distributed is not None:
            assert self._distributed.world_size == self._ex_iterable._query_execution_args.dp_groups, (
                f"self._distributed.world_size = {self._distributed.world_size} != Mixtera"
                + f"dp_groups = {self._ex_iterable._query_execution_args.dp_groups}"
            )
            assert self._distributed.rank == self._ex_iterable._dp_group_id, (
                f"self._distributed.rank = {self._distributed.rank} != Mixtera"
                + f"dp_group_id = {self._ex_iterable._dp_group_id}"
            )
        else:
            assert self._ex_iterable._query_execution_args.dp_groups == 1
            assert self._ex_iterable._dp_group_id == 0

        yield from super().__iter__()
        logger.info(
            f"[{self._ex_iterable._dp_group_id}-{self._ex_iterable._node_id}-{self._ex_iterable.worker_id}]"
            + " Finished yielding."
        )
