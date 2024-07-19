import multiprocessing as mp
from pathlib import Path
from typing import Callable, Generator, Type

from loguru import logger
from mixtera.core.client import MixteraClient
from mixtera.core.datacollection import MixteraDataCollection, PropertyType
from mixtera.core.datacollection.datasets import Dataset
from mixtera.core.datacollection.index.index import ChunkerIndex
from mixtera.core.datacollection.index.parser import MetadataParser
from mixtera.core.processing import ExecutionMode
from mixtera.core.query import Mixture, Query, QueryResult
from mixtera.core.query.chunk_distributor import ChunkDistributor
from mixtera.utils import wait_for_key_in_dict


class LocalStub(MixteraClient):
    def __init__(self, directory: Path | str) -> None:
        if isinstance(directory, str):
            self.directory = Path(directory)
        else:
            self.directory = directory

        if not self.directory.exists():
            raise RuntimeError(f"Directory {self.directory} does not exist.")

        self._mdc = MixteraDataCollection(self.directory)
        self._training_query_map_lock = mp.Lock()
        self._training_query_map: dict[str, tuple[ChunkDistributor, Query, Mixture]] = {}  # (query, mixture_object)

    def register_dataset(
        self,
        identifier: str,
        loc: str | Path,
        dtype: Type[Dataset],
        parsing_func: Callable[[str], str],
        metadata_parser_identifier: str,
    ) -> bool:
        if isinstance(loc, Path):
            loc = str(loc)

        return self._mdc.register_dataset(identifier, loc, dtype, parsing_func, metadata_parser_identifier)

    def register_metadata_parser(
        self,
        identifier: str,
        parser: Type[MetadataParser],
    ) -> bool:
        return self._mdc._metadata_factory.add_parser(identifier, parser)

    def check_dataset_exists(self, identifier: str) -> bool:
        return self._mdc.check_dataset_exists(identifier)

    def list_datasets(self) -> list[str]:
        return self._mdc.list_datasets()

    def remove_dataset(self, identifier: str) -> bool:
        return self._mdc.remove_dataset(identifier)

    def execute_query(
        self, query: Query, mixture: Mixture, dp_groups: int, nodes_per_group: int, num_workers: int
    ) -> bool:
        assert dp_groups > 0 and nodes_per_group > 0 and num_workers >= 0
        query.execute(self._mdc, mixture)
        return self._register_query(query, mixture, dp_groups, nodes_per_group, num_workers)

    def is_remote(self) -> bool:
        return False

    def add_property(
        self,
        property_name: str,
        setup_func: Callable,
        calc_func: Callable,
        execution_mode: ExecutionMode,
        property_type: "PropertyType",
        min_val: float = 0.0,
        max_val: float = 1.0,
        num_buckets: int = 10,
        batch_size: int = 1,
        degree_of_parallelism: int = 1,
        data_only_on_primary: bool = True,
    ) -> bool:
        return self._mdc.add_property(
            property_name,
            setup_func,
            calc_func,
            execution_mode,
            property_type,
            min_val=min_val,
            max_val=max_val,
            num_buckets=num_buckets,
            batch_size=batch_size,
            degree_of_parallelism=degree_of_parallelism,
            data_only_on_primary=data_only_on_primary,
        )

    def _stream_result_chunks(
        self,
        job_id: str,
        dp_group_id: int,
        node_id: int,
        worker_id: int,
    ) -> Generator[ChunkerIndex, None, None]:
        yield from self._get_query_chunk_distributor(job_id)._stream_chunks_for_worker(dp_group_id, node_id, worker_id)

    def _get_result_metadata(
        self, job_id: str
    ) -> tuple[dict[int, Type[Dataset]], dict[int, Callable[[str], str]], dict[int, str]]:
        query_result = self._get_query_result(job_id)
        return query_result.dataset_type, query_result.parsing_func, query_result.file_path

    def _register_query(
        self, query: "Query", mixture: Mixture, dp_groups: int, nodes_per_group: int, num_workers: int
    ) -> bool:
        if query.job_id in self._training_query_map:
            logger.warning(f"We already have a query for job {query.job_id}!")
            return False

        with self._training_query_map_lock:
            self._training_query_map[query.job_id] = (
                ChunkDistributor(dp_groups, nodes_per_group, num_workers, query.results, query.job_id),
                query,
                mixture,
            )

        logger.info(f"Registered query {str(query)} for job {query.job_id}, with mixture {mixture}")

        return True

    def _get_query_result(self, job_id: str) -> QueryResult:
        if not wait_for_key_in_dict(self._training_query_map, job_id, 60.0):
            raise RuntimeError(f"Unknown job {job_id}")
        return self._training_query_map[job_id][1].results

    def _get_query_chunk_distributor(self, job_id: str) -> ChunkDistributor:
        if not wait_for_key_in_dict(self._training_query_map, job_id, 60.0):
            raise RuntimeError(f"Unknown job {job_id}")
        return self._training_query_map[job_id][0]
