from typing import Any, Generator

from loguru import logger
from mixtera.core.client import MixteraClient
from mixtera.core.query import Mixture, Query
from torch.utils.data import IterableDataset, get_worker_info  # pylint: disable=import-error,no-name-in-module


class MixteraTorchDataset(IterableDataset):
    def __init__(
        self,
        client: MixteraClient,
        query: Query,
        job_id: str,
        mixture: Mixture,
        dp_groups: int = 1,
        dp_group_id: int = 0,
        nodes_per_group: int = 1,
        node_id: int = 0,
        num_workers: int = 1,
        tunnel_via_server: bool = False,
    ):
        # TODO(#63): This needs to be passed information on transformation, e.g., tokenization functions etc.
        # Alternative: Let people inherit from this.
        self._client = client
        self._query = query
        self._training_id = job_id
        self._dp_group_id = dp_group_id
        self._node_id = node_id
        self._mixture = mixture
        self._tunnel_via_server = tunnel_via_server
        self._num_workers = num_workers if num_workers > 0 else 1

        assert self._dp_group_id < dp_groups
        assert self._node_id < nodes_per_group

        if self._node_id == 0 and self._dp_group_id == 0:
            logger.info("Since this is node 0 in data parallel group 0, executing query!")
            # Execute query on primary node pre-fork, to share the results among all forked workers
            self._client.execute_query(query, self._mixture, dp_groups, nodes_per_group, num_workers)

    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError("This is just overwritten to satify pylint.")

    def __iter__(self) -> Generator[str, None, None]:
        worker_info = get_worker_info()
        if worker_info is None:
            # Non-multithreaded data loading. We use worker_id 0.
            worker_id = 0
        else:
            worker_id = worker_info.id

        assert worker_id < self._num_workers, f"Number of workers was invalid: {worker_id} vs {self._num_workers}"

        yield from self._client.stream_results(
            self._training_id, self._dp_group_id, self._node_id, worker_id, self._tunnel_via_server
        )
