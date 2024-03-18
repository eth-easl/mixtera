from typing import Any, Generator

from loguru import logger
from mixtera.core.datacollection import MixteraClient
from mixtera.core.query import Query
from torch.utils.data import IterableDataset  # pylint: disable=import-error,no-name-in-module


class MixteraTorchDataset(IterableDataset):
    def __init__(
        self,
        mdc: MixteraClient,
        query: Query,
        training_id: str,
        chunk_size: int,
        node_id: int = 0,
        tunnel_via_server: bool = False,
    ):
        # TODO(create issue): This needs to be passed information on transformation, e.g., tokenization functions etc.
        # Alternative: Let people inherit from this.
        self._mdc = mdc
        self._query = query
        self._training_id = training_id
        self._node_id = node_id
        self._chunk_size = chunk_size
        self._tunnel_via_server = tunnel_via_server

        if self._node_id == 0:
            logger.info("Since this is node 0, executing query!")
            # Execute query on primary node pre-fork, to share the results among all forked workers
            self._query.execute(self._mdc, self._chunk_size)

    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError("This is just overwritten to satify pylint.")

    def __iter__(self) -> Generator[str, None, None]:
        query_result = self._mdc.get_query_result(self._training_id)
        yield from self._mdc.stream_query_results(query_result, self._tunnel_via_server)
