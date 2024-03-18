class QueryResult(ABC):
    """QueryResult is a class that represents the results of a query.
    The QueryResult object is iterable and yields the results in chunks of size `chunk_size`.

    The QueryResult object also has three meta properties: `dataset_type`,
    `file_path` and `parsing_func`, each of which is a dictionary that maps
    dataset/file ids to their respective types, paths and parsing functions.
    """

    @abstractmethod
    def __next__(self) -> IndexType:
        raise NotImplementedError()

    def __iter__(self) -> "QueryResult":
        return self

    @property
    @abstractmethod
    def dataset_type(self) -> dict[int, Type[Dataset]]:
        raise NotImplementedError()

    @property
    @abstractmethod
    def file_path(self) -> dict[int, str]:
        raise NotImplementedError()

    @property
    @abstractmethod
    def parsing_func(self) -> dict[int, Callable[[str], str]]:
        raise NotImplementedError()


class RemoteQueryResult(QueryResult):
    def __init__(self, server_connection: ServerConnection, query_id: int):
        self._server_connection = server_connection
        self._query_id = query_id
        self._meta: dict[str, Any] = {}
        self._result_generator: Optional[Generator[IndexType, None, None]] = None

    def _fetch_meta_if_empty(self) -> None:
        if not self._meta:
            if (meta := self._server_connection.get_query_result_meta(self._query_id)) is None:
                raise RuntimeError("Error while fetching meta results")

            self._meta = meta

    @property
    def dataset_type(self) -> dict[int, Type[Dataset]]:
        self._fetch_meta_if_empty()
        return self._meta["dataset_type"]

    @property
    def file_path(self) -> dict[int, str]:
        self._fetch_meta_if_empty()
        return self._meta["file_path"]

    @property
    def parsing_func(self) -> dict[int, Callable[[str], str]]:
        self._fetch_meta_if_empty()
        return self._meta["parsing_func"]

    def _fetch_results_if_none(self) -> None:
        if self._result_generator is None:
            self._result_generator = self._server_connection.get_query_results(self._query_id)

    def __iter__(self) -> "RemoteQueryResult":
        self._fetch_results_if_none()
        return self

    def __next__(self) -> IndexType:
        if self._result_generator is None:
            raise StopIteration
        # This is thread safe in the sense that the server
        # handles each incoming request in a thread safe way
        # While worker 1 could send its request before worker 2,
        # worker 2 reads None, closes its generator, and then worker
        # 1 gets its data back, this is not a problem since we need
        # to consume all workers when in a multi data loader worker setting.

        try:
            return next(self._result_generator)
        except StopIteration:
            self._result_generator = None
            raise
