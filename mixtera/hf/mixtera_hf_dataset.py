import datasets

from typing import Tuple, Generator, Optional

from loguru import logger
from mixtera.core.client import MixteraClient
from mixtera.core.query import Query


class _MixteraHFIterable(datasets.iterable_dataset._BaseExamplesIterable):
    def __init__(
        self,
        client: MixteraClient,
        query: Query,
        job_id: str,
        chunk_size: int,
        node_id: int = 0,
        tunnel_via_server: bool = False,
        worker_id: Optional[int] = None
    ):
        super().__init__()
        self._client = client
        self._query = query
        self._training_id = job_id
        self._node_id = node_id
        self._chunk_size = chunk_size
        self._tunnel_via_server = tunnel_via_server
        self._worker_id = worker_id

        if self._node_id == 0 and self._worker_id is None:
            # We execute hte query on primary node pre-fork of the dataloader,
            # to share the results among all forked workers
            logger.info("Since this is node 0, executing query!")
            self._client.execute_query(query, self._chunk_size)

    @property
    def n_shards(self) -> int:
        return 64 # This is an arbirary limit of 64 dataloader workers per node. HF requires us to set some number.
    
    def shard_data_sources(self, worker_id: int, num_workers: int) -> "_MixteraHFIterable":
        # This gets called by the IterableDataset from huggingface and should return
        # the iterable for the specific worker. In our case, Mixtera handles this implicitly.
        # Each worker handles one chunk at a time.
        logger.debug(f"shard_data_sources called with {worker_id} and {num_workers}")
        if self._worker_id is None:
            return _MixteraHFIterable(self._client, self._query, self._training_id, self._chunk_size, self._node_id, self._tunnel_via_server, worker_id)

        return self

    def __iter__(self) -> Generator[Tuple[str, dict], None, None]:
        assert self._worker_id is not None, "shard_data_sources should have been called - something went wrong."
        for idx, sample in enumerate(self._client.stream_results(self._training_id, self._tunnel_via_server)):
            yield (f"{self._node_id}-{self._worker_id}-{idx}", { "text": sample })

        logger.info("reached end of stream")

class MixteraHFDataset(datasets.IterableDataset):
    def __init__(self,
        client: MixteraClient,
        query: Query,
        job_id: str,
        chunk_size: int,
        node_id: int = 0,
        tunnel_via_server: bool = False,
        worker_id: Optional[int] = None):
            super().__init__(_MixteraHFIterable(client, query, job_id, chunk_size, node_id = node_id, tunnel_via_server=tunnel_via_server, worker_id=worker_id))
            self.info.features = datasets.Features({ "text": datasets.Value(dtype="string") })