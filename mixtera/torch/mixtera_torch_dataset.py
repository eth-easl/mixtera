from typing import Any, Generator

from loguru import logger
from mixtera.core.client import MixteraClient
from mixtera.core.query import Mixture, Query
from torch.utils.data import IterableDataset  # pylint: disable=import-error,no-name-in-module


class MixteraTorchDataset(IterableDataset):
    def __init__(
        self,
        client: MixteraClient,
        query: Query,
        job_id: str,
        mixture: Mixture,
        node_id: int = 0,
        tunnel_via_server: bool = False,
    ):
        # TODO(#63): This needs to be passed information on transformation, e.g., tokenization functions etc.
        # Alternative: Let people inherit from this.
        self._client = client
        self._query = query
        self._training_id = job_id
        self._node_id = node_id
        self._mixture = mixture
        self._tunnel_via_server = tunnel_via_server

        if self._node_id == 0:
            logger.info("Since this is node 0, executing query!")
            # Execute query on primary node pre-fork, to share the results among all forked workers
            self._client.execute_query(query, self._mixture)

    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError("This is just overwritten to satify pylint.")

    def __iter__(self) -> Generator[str, None, None]:
        yield from self._client.stream_results(self._training_id, self._tunnel_via_server)
