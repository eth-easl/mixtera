from copy import deepcopy
from typing import Any, Generator, Tuple

import datasets
import numpy as np
from loguru import logger
from mixtera.core.client import MixteraClient
from mixtera.core.client.mixtera_client import QueryExecutionArgs, ResultStreamingArgs
from mixtera.core.query import Query
from mixtera.torch import MixteraTorchDataset


class _MixteraHFIterable(MixteraTorchDataset, datasets.iterable_dataset._BaseExamplesIterable):
    def __init__(
        self,
        client: MixteraClient,
        query: Query,
        query_execution_args: QueryExecutionArgs,
        result_streaming_args: ResultStreamingArgs,
        _shard_called: bool = False,
    ):
        MixteraTorchDataset.__init__(self, client, query, query_execution_args, result_streaming_args)
        datasets.iterable_dataset._BaseExamplesIterable.__init__(self)
        self._shard_called = _shard_called

    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError("This is just overwritten to satify pylint.")

    def _init_state_dict(self) -> dict:
        raise NotImplementedError("Do we need this?")

    def shuffle_data_sources(self, generator: np.random.Generator) -> datasets.iterable_dataset._BaseExamplesIterable:
        raise NotImplementedError("Do we need this?")

    @property
    def n_shards(self) -> int:
        return self._query_execution_args.num_workers * 8  # HF requires us to set some number.

    @property
    def worker_id(self) -> int:
        assert self._shard_called, "shard_data_sources should have been called - something went wrong."
        return self._res_str_args.worker_id

    def shard_data_sources(self, worker_id: int, num_workers: int) -> "_MixteraHFIterable":
        # This gets called by the IterableDataset from huggingface and should return
        # the iterable for the specific worker. In our case, Mixtera handles this implicitly.
        # Each worker handles one chunk at a time.
        logger.debug(f"shard_data_sources called with {worker_id} and {num_workers}")
        assert num_workers == self._query_execution_args.num_workers, (
            f"num_workers = {num_workers} != query.num_workers ="
            + f"{self._query_execution_args.num_workers} defined at query execution."
        )

        if not self._shard_called:
            res_args = deepcopy(self._res_str_args)
            res_args.worker_id = worker_id
            return _MixteraHFIterable(
                self._client, self._query, self._query_execution_args, res_args, _shard_called=True
            )

        assert (
            self._res_str_args.worker_id == worker_id
        ), f"worker_id = {worker_id} != self.worker_id = {self._res_str_args.worker_id}"
        return self

    def validate_state(self) -> None:
        assert self._shard_called, "shard_data_sources should have been called - something went wrong."
        assert (
            MixteraTorchDataset.worker_id(self) == self.worker_id
        ), f"torch worker id = {MixteraTorchDataset.worker_id(self)} != self.worker_id = {self.worker_id}"

    def __iter__(self) -> Generator[Tuple[str, dict], None, None]:
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

    def __iter__(self) -> Generator[Any | dict, Any, None]:
        # We wrap IterableDataset.__iter__ to do some state assertions
        assert isinstance(self._ex_iterable, _MixteraHFIterable)
        self._ex_iterable: _MixteraHFIterable
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
