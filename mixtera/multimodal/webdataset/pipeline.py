from typing import Any, Iterable
import webdataset as wds

from mixtera.core.client.mixtera_client import (
    MixteraClient,
    QueryExecutionArgs,
    ResultStreamingArgs,
)
from mixtera.core.query.query import Query
from mixtera.torch import MixteraTorchDataset


class MixteraDataPipeline(wds.DataPipeline):
    """
    Supports building arbitrary webdataset pipelines with Mixtera's `MixteraTorchDataset` as the data source.
    """

    def __init__(
        self,
        client: MixteraClient,
        query: Query,
        query_execution_args: QueryExecutionArgs,
        result_streaming_args: ResultStreamingArgs,
        pipeline: Iterable[Any], 
    ):
        super().__init__(*pipeline)
        self.client = client
        self.query = query
        self.query_execution_args = query_execution_args
        self.result_streaming_args = result_streaming_args

        torch_dataset = MixteraTorchDataset(
            client=client,
            query=query,
            query_execution_args=query_execution_args,
            result_streaming_args=result_streaming_args,
        )

        self.pipeline.insert(0, torch_dataset)
