from typing import Any, Generator

from loguru import logger
from mixtera.core.client import MixteraClient
from mixtera.core.client.mixtera_client import QueryExecutionArgs, ResultStreamingArgs
from mixtera.core.query import Query
from mixtera.core.query.mixture import Mixture
from torch.utils.data import IterableDataset, get_worker_info  # pylint: disable=import-error,no-name-in-module


class MixteraTorchDataset(IterableDataset):
    def __init__(
        self,
        client: MixteraClient,
        query: Query,
        query_execution_args: QueryExecutionArgs,
        result_streaming_args: ResultStreamingArgs,
    ):
        # TODO(#63): This needs to be passed information on transformation, e.g., tokenization functions etc.
        # Alternative: Let people inherit from this.
        self._client = client
        self._query = query
        self._res_str_args = result_streaming_args
        self._query_execution_args = query_execution_args

        assert self._dp_group_id < query_execution_args.dp_groups
        assert self._node_id < query_execution_args.nodes_per_group

        if self._node_id == 0 and self._dp_group_id == 0:
            logger.info("Since this is node 0 in data parallel group 0, executing query!")
            # Execute query on primary node pre-fork, to share the results among all forked workers
            self._client.execute_query(query, self._query_execution_args)

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

    def __iter__(self) -> Generator[str, None, None]:
        self._res_str_args.worker_id = self.worker_id
        yield from self._client.stream_results(self._res_str_args)
