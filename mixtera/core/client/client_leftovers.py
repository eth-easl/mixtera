    # TODO(MaxiBoether): Remove the next 3 functions. Instead, iterate on the query result directly and get samples.
    @abstractmethod
    def stream_query_results(
        self, query_result: "QueryResult", tunnel_via_server: bool = False
    ) -> Generator[str, None, None]:
        """
        Given a query_results object, iterates over the samples of the query.
        Args:
            query_result (QueryResult): The QueryResult object.
            tunnel_via_server (bool): If True, the sample payloads
                will be streamed via the Mixtera server. Otherwise,
                the client will access the files directly. Needs to be False
                for LocalDataCollection, and defaults to False.
        Returns:
            A Generator of samples.
        """
        raise NotImplementedError()

    @staticmethod
    def _stream_query_results(
        query_result: "QueryResult", server_connection: Optional["ServerConnection"] = None
    ) -> Generator[str, None, None]:
        """
        Internal implementation for streaming query results.
        Called from the Local/RemoteDataCollection implementations of `stream_query_results`.
        Args:
            query_result (QueryResult): The QueryResult object.
            server_connection (Optional[ServerConnection]): If given,
                the sample payloads are streamed via this Mixtera server.
        Returns:
            A Generator of samples.
        """
        for result_chunk in query_result:
            yield from MixteraClient._read_result_chunk(
                result_chunk,
                query_result.dataset_type,
                query_result.parsing_func,
                query_result.file_path,
                server_connection=server_connection,
            )

    @staticmethod
    def _read_result_chunk(
        result_chunk: IndexType,
        dataset_type_dict: dict[int, Type[Dataset]],
        parsing_func_dict: dict[int, Callable[[str], str]],
        file_path_dict: dict[int, str],
        server_connection: Optional["ServerConnection"] = None,
    ) -> Generator[str, None, None]:
        """
        Given a result chunk, iterates over the samples.

        TODO(create issue): This currently iterates through it property by property, dataset by dataset etc.
        Instead, we want to sample from a result chunk uniform at random
        It is a bit unclear how to implementing sampling here, since we work with ranges.
        In the best case, we would sample line by line u.a.r.

        Args:
            result_chunk (IndexType): The result chunk object.
            dataset_type_dict (dict): A mapping from dataset ID to dataset type.
            parsing_func_dict (dict): A mapping from dataset ID to parsing function.
            file_path_dict (dict): A mapping from file ID to file path.
            server_connection (Optional[ServerConnection]): If given,
                the sample payloads are streamed via this Mixtera server.

        Returns:
            A Generator of samples.
        """
        # TODO(create issue): In case server_connection is used, this is super duper mega slow, since
        # we restream the entire file each time.
        for _, property_dict in result_chunk._index.items():
            for _, val_dict in property_dict.items():
                for did, file_dict in val_dict.items():
                    filename_dict = {file_path_dict[file_id]: file_ranges for file_id, file_ranges in file_dict.items()}
                    yield from dataset_type_dict[did].read_ranges_from_files(
                        filename_dict, parsing_func_dict[did], server_connection
                    )

