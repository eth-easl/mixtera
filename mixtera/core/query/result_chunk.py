import ast
import inspect
import multiprocessing as mp
import os
import random
import textwrap
import typing
from queue import Empty
from typing import TYPE_CHECKING, Callable, Iterator, Optional, Type

import dill
from loguru import logger
from mixtera.core.datacollection.datasets import Dataset
from mixtera.core.datacollection.index import ChunkerIndex, IndexRowRangeType, infer_mixture_from_chunkerindex
from mixtera.core.query.mixture import MixtureKey, StaticMixture
from mixtera.network.connection import ServerConnection
from mixtera.utils import is_on_github_actions, seed_everything_from_list

if TYPE_CHECKING:
    from mixtera.core.client.mixtera_client import MixteraClient, ResultStreamingArgs

Workload = tuple[int, int, IndexRowRangeType]
Workloads = list[Workload]

MULTIPROCESSING_TIMEOUT = 90
END_OF_STREAM_OBJECT = "END_OF_STREAM"

original_start = mp.Process.start


@typing.no_type_check
def allow_daemon_spawn() -> None:
    # PyTorch's data loader spawns data loading workers as daemon processes
    # Each data loader worker then uses this class here, meaning that it spawns processes
    # By default, this is not allowed, since daemon processes may not have children
    # In our case, we need to allow this, since we don't want to change torch's dataloader
    # To this end, we allow starting a daemon process from a daemon process
    # Note: We need to define this function within this module to properly monkey-patch this instance of multiprocessing
    def patched_start(self, *args, **kwargs) -> None:
        if self.daemon:  # if the child is a daemon
            # Goal: Remove assertion that our parent is not a daemon
            # Load source code of original start method
            source = textwrap.dedent(inspect.getsource(original_start))

            # Create AST
            tree = ast.parse(source)

            # Remove assertion from AST
            for i, node in enumerate(tree.body[0].body):
                if isinstance(node, ast.Assert) and "daemon" in ast.unparse(node):
                    tree.body[0].body[i] = ast.Pass()
                    break

            # Generate a new function with correct context that we can use without the assertion
            new_func = ast.FunctionDef(
                name="modified_start", args=tree.body[0].args, body=tree.body[0].body, decorator_list=[]
            )
            module = ast.Module(body=[new_func], type_ignores=[])
            compiled = compile(ast.fix_missing_locations(module), "<string>", "exec")

            namespace = original_start.__globals__.copy()
            namespace.update(self.__dict__)
            namespace.update(self.__class__.__dict__)

            # Execute the compiled code in this namespace and call it
            exec(compiled, namespace)  # pylint: disable=exec-used
            namespace["modified_start"](self, *args, **kwargs)
        else:
            # For non-daemon processes, use the original start method
            original_start(self, *args, **kwargs)

    if mp.Process.start == original_start:
        # Do the monkey-patch
        mp.Process.start = patched_start


class ResultChunk:
    def __init__(
        self,
        result_index: ChunkerIndex,
        dataset_type_dict: dict[int, Type[Dataset]],
        file_path_dict: dict[int, str],
        parsing_func_dict: dict[int, Callable[[str], str]],
        chunk_size: int,
        mixture: Optional[dict[MixtureKey, int]] = None,
    ) -> None:
        allow_daemon_spawn()

        self._result_index = result_index
        self._dataset_type_dict = dataset_type_dict
        self._file_path_dict = file_path_dict
        self._parsing_func_dict = parsing_func_dict
        self._chunk_size = chunk_size
        self._mixture = mixture

        self._server_connection: Optional[ServerConnection] = None
        self._degree_of_parallelism: int = 1
        self._per_window_mixture: bool = False
        self._window_size: int = 128

        self._iterator: Optional[Iterator[str]] = None

    def configure_result_streaming(self, client: "MixteraClient", args: "ResultStreamingArgs") -> None:
        """
        Configure the result streaming for the ResultChunk. This function sets the degree of parallelism,
        the window size, and the mixture based on the arguments.

        Args:
            client: The MixteraClient instance
            args: The ResultStreamingArgs instance
        """
        allow_daemon_spawn()

        self._degree_of_parallelism = args.chunk_reading_degree_of_parallelism
        self._per_window_mixture = args.chunk_reading_per_window_mixture
        self._window_size = args.chunk_reading_window_size

        from mixtera.core.client.server import ServerStub  # pylint: disable=import-outside-toplevel

        if args.tunnel_via_server:
            if isinstance(client, ServerStub):
                self._server_connection = client.server_connection
            else:
                raise RuntimeError(
                    "Currently, tunneling samples via the server is only supported when using a ServerStub."
                )

        if self._degree_of_parallelism < 1:
            if not is_on_github_actions:
                logger.warning(
                    f"Degree of parallelism is set to {self._degree_of_parallelism} which is invalid. "
                    "Setting degree of parallelism to 1."
                )
            self._degree_of_parallelism = 1

        if self._per_window_mixture and self._window_size > self._chunk_size:
            if not is_on_github_actions:
                logger.warning(
                    f"Window size is set to {self._window_size} which is > the chunk size of {self._chunk_size}. "
                    "Setting window size to the chunk size."
                )
            self._window_size = self._chunk_size

        if self._per_window_mixture and self._window_size < 1:
            if not is_on_github_actions:
                logger.warning(
                    f"Window size is set to {self._window_size} which is invalid. " "Setting window size to 128."
                )
            self._window_size = 128

        # To determine the number of processes per property combination, we need the mixture
        # for parallel reading. If the mixture is not defined, we infer it from the result index.
        if (self._per_window_mixture or self._degree_of_parallelism > 1) and (
            self._mixture is None or len(self._mixture) == 0
        ):
            if not is_on_github_actions:
                logger.debug("Mixture is not defined or empty but required. Infer mixture from the result index.")
            self._mixture = self._infer_mixture()

        # If we have a mixture, ensure that the mixture supports the chunk
        if self._mixture is not None and (not self._mixture.keys() == self._result_index.keys()):
            raise RuntimeError(
                "The received chunk has keys that do not match the mixture. That should not happen.\n"
                + f"{self._result_index.keys()}"
                + f"\n{self._mixture.keys()}"
                + f"\n{self._mixture}"
            )

    def _infer_mixture(self) -> dict[MixtureKey, int]:
        return StaticMixture(*infer_mixture_from_chunkerindex(self._result_index)).mixture_in_rows()

    def _iterate_samples(self) -> Iterator[str]:
        """
        Iterate over the samples in the result index. This function yields the samples in the correct mixture
        and window size.

        Returns:
            An iterator over the samples
        """
        active_iterators: dict[MixtureKey, Iterator[str]] = self._init_active_iterators()
        if self._per_window_mixture:
            yield_source = self._iterate_window_mixture(active_iterators)
        else:
            yield_source = self._iterate_overall_mixture(active_iterators)
        yield from yield_source

    def _init_active_iterators(self) -> dict[MixtureKey, Iterator[str]]:
        """
        Get the active iterators for the result index. This function prepares the workloads and spins up the
        reader processes if required by the degree of parallelism.
        """
        workloads: dict[MixtureKey, Workloads] = self._prepare_workloads()

        active_iterators: dict[MixtureKey, Iterator[str]] = {}
        if self._degree_of_parallelism == 1:
            active_iterators = {
                property_name: self._get_iterator_for_workload_st(workload)
                for property_name, workload in workloads.items()
            }
        elif self._degree_of_parallelism > 1:
            process_counts = self._get_process_counts()

            processes: dict[MixtureKey, list[tuple[mp.Queue, mp.Process]]] = self._spin_up_readers(
                workloads, process_counts
            )

            active_iterators = {
                property_name: self._get_iterator_for_workload_mt(process)
                for property_name, process in processes.items()
            }

        return active_iterators

    def _get_process_counts(self) -> dict[str, int]:
        """
        Get the number of processes per property combination. This function determines the number of processes
        to use based on the degree of parallelism and the mixture.

        Each property combination is assigned a number of processes based on the mass of the property combination
        in the mixture.
        """
        assert isinstance(
            self._mixture, dict
        ), "Mixture must be defined for parallel reading when getting the process counts, this should not happen."

        #  Determine the number of processes to use
        reader_count = min(
            self._degree_of_parallelism if self._degree_of_parallelism is not None else mp.cpu_count(),
            mp.cpu_count(),
        )

        # Determine how many processes should be assigned per property combination
        process_counts = {key: int((val / self._chunk_size) * reader_count) for key, val in self._mixture.items()}

        process_counts[list(process_counts.keys())[0]] += reader_count - sum(process_counts.values())

        return process_counts

    def _get_iterator_for_workload_st(self, workloads: Workloads) -> Iterator[str]:
        """
        Get the iterator for the workload in single-threaded mode. This function reads the instances from the
        files in the workload and yields them.
        """
        for dataset_id, file_id, ranges in workloads:
            filename_dict = {self._file_path_dict[file_id]: ranges}
            yield from self._dataset_type_dict[dataset_id].read_ranges_from_files(
                filename_dict, self._parsing_func_dict[dataset_id], self._server_connection
            )

    def _get_iterator_for_workload_mt(self, processes: list[tuple[mp.Queue, mp.Process]]) -> Iterator[str]:
        """
        Get the iterator for the workload in multi-threaded mode. This function yields the instances from the
        queues of the processes.
        """
        while processes:
            processes_to_remove = []
            for queue, proc in processes:
                try:
                    instance = queue.get(timeout=MULTIPROCESSING_TIMEOUT)
                except Empty as exc:
                    if not proc.is_alive():
                        proc.join()
                        processes_to_remove.append((queue, proc))
                        continue
                    raise RuntimeError(
                        "Queue timeout reached but process is still alive. Something went wrong."
                    ) from exc
                if instance == END_OF_STREAM_OBJECT:
                    proc.join()
                    processes_to_remove.append((queue, proc))
                    continue
                yield instance

            for queue, proc in processes_to_remove:
                processes.remove((queue, proc))

    def _iterate_window_mixture(self, active_iterators: dict[MixtureKey, Iterator[str]]) -> Iterator[str]:
        """
        Iterate over the samples in the result index with a windowed mixture. This function yields the samples
        in the correct mixture withing a window.
        """
        element_counts = self._get_element_counts()

        # Shuffle the results to ensure that the order of the property combinations is (reproducibly) random
        seed_everything_from_list(element_counts)
        random.shuffle(element_counts)

        # This assertion here makes sure our chunk matches the mixture, which is the underlying assumption
        # of the implementation below. There cannot be more than one key in the chunk we use per mixture key
        assert set(key for (key, _) in element_counts) == active_iterators.keys(), (
            f"element_counts.keys = {element_counts} != " + f"active_iterators.keys() = {active_iterators.keys()}"
        )

        deleted_keys: set[MixtureKey] = set()

        # Continue until all iterators are deleted
        while len(active_iterators) > len(deleted_keys):
            items_yielded = 0
            processed_items = {property_key: 0 for property_key, _ in element_counts}
            # This inner while loop represents one window with the correct mixture
            # We continue until the window is full or we don't have enough active iterators (outer condition)
            while len(active_iterators) > len(deleted_keys) and items_yielded < self._window_size:
                nothing_yielded_window = True
                for property_key, property_count in element_counts:
                    # We iterate through all mixture keys and yield one item for the key per iteration
                    if property_key in deleted_keys or processed_items[property_key] >= property_count:
                        # However, if we cannot produce any items for this key anymore, (key has been deleted)
                        # OR if we're done for this window, we move to the next property
                        # We might also want to support stopping here if one property name
                        continue
                    try:
                        # Yield the next sample from the iterator
                        yield next(active_iterators[property_key])
                        nothing_yielded_window = False
                        processed_items[property_key] += 1
                        items_yielded += 1
                        # If the window is full, break the inner for loop, will also break the outer (window) while loop
                        # since the items_yielded >= self._window_size, and then start the next window
                        if items_yielded >= self._window_size:
                            break
                    except StopIteration:
                        # If no more workloads, this property is done
                        deleted_keys.add(property_key)

                if nothing_yielded_window:
                    break

    def _iterate_overall_mixture(self, active_iterators: dict[MixtureKey, Iterator[str]]) -> Iterator[str]:
        """
        Iterate over the samples in the result index with an overall mixture. This function yields the samples
        in the overall correct mixture.
        """
        # Shuffle the results to ensure that the order of the property combinations is (reproducibly) random
        property_names = list(active_iterators.keys())
        seed_everything_from_list(property_names)
        random.shuffle(property_names)

        deleted_keys: set[MixtureKey] = set()

        while len(active_iterators) > len(deleted_keys):
            for property_name in property_names:
                # If the property is done, skip
                if property_name in deleted_keys:
                    continue
                try:
                    yield next(active_iterators[property_name])
                except StopIteration:
                    deleted_keys.add(property_name)

    def _get_element_counts(self) -> list[tuple[MixtureKey, int]]:
        """
        Get the element counts for each property combination. This is used to determine how many instances
        of each property combination should be yielded in a window.

        Returns:
            A list of tuples with the property combination and the number of instances to yield
        """
        assert isinstance(self._mixture, dict), "Mixture must be defined for windowed reading, this should not happen."

        # Determine the per-property combination batch counts
        initial_counts = [
            (key, int(self._window_size * (value / self._chunk_size))) for key, value in self._mixture.items()
        ]
        total_counts = sum(count for _, count in initial_counts)
        remainder = self._window_size - total_counts

        #  Adjust the counts to ensure that the window size is met
        adjusted_counts = [
            (key, count + remainder if i == 0 else count) for i, (key, count) in enumerate(initial_counts)
        ]

        return adjusted_counts

    def _spin_up_readers(
        self,
        workloads: dict[str, list[tuple[int, int, list]]],
        process_counts: dict[str, int],
    ) -> dict[str, list[tuple[mp.Queue, mp.Process]]]:
        """
        Spin up the reader processes for the workloads. This function creates the processes and queues
        for the workloads and starts the processes.

        Args:
            workloads: a dictionary with the workloads per property combination
            process_counts: a dictionary with the number of processes per property combination
        """
        processes: dict[str, list[tuple[mp.Queue, mp.Process]]] = {}
        total_processes = 0
        pickled_func_dict = dill.dumps(self._parsing_func_dict)
        start_as_daemon = True if mp.current_process().daemon else None
        for key, process_count in process_counts.items():
            processes[key] = []

            if process_count < 1:
                # TODO(#85): This will currently lead to more processes than
                # intended if degree_of_parallelism < properties
                logger.warning(
                    f"Number of processes for property combination {key} is set to {process_count} which is invalid. "
                    "Setting number of processes to 1."
                )
                process_count = 1

            # Calculate per-process partition sizes
            partition_size = max(1, len(workloads[key]) // process_count)
            partition_ranges = list(range(0, len(workloads[key]), partition_size)) + [len(workloads[key])]

            # Create and start the processes
            for i in range(1, len(partition_ranges)):
                total_processes += 1
                queue: mp.Queue = mp.Queue()
                processes[key].append(
                    (
                        queue,
                        mp.Process(
                            target=self._reader_process,
                            daemon=start_as_daemon,
                            args=(
                                queue,
                                self._dataset_type_dict,
                                self._file_path_dict,
                                pickled_func_dict,
                                self._server_connection,
                                workloads[key][partition_ranges[i - 1] : partition_ranges[i]],
                            ),
                        ),
                    )
                )

                # Start the process
                processes[key][-1][1].start()

        logger.debug(f"Started {total_processes} processes for chunk processing (dop = {self._degree_of_parallelism})")
        return processes

    @staticmethod
    def _reader_process(
        queue: mp.Queue,
        dataset_type_dict: dict[int, Type[Dataset]],
        file_path_dict: dict[int, str],
        pickled_parsing_func_dict: bytes,
        server_connection: ServerConnection,
        workloads: Workloads,
    ) -> None:
        """
        The reader process reads the instances from the files in the workloads and puts them into the queue.

        Args:
            queue: the queue to put the instances into
            dataset_type_dict: a dictionary with the dataset types
            file_path_dict: a dictionary with the file paths
            pickled_parsing_func_dict: a pickled dictionary with the parsing functions
            server_connection: the server connection to use
            workloads: the workloads to process
        """
        # We might have been started as a daemon,
        # in which case we need to clean up ourselves in case for whatever reason our parent exits.
        start_ppid = os.getppid() if mp.current_process().daemon else None
        parsing_func_dict: dict[int, Callable[[str], str]] = dill.loads(pickled_parsing_func_dict)
        for dataset_id, file_id, ranges in workloads:
            if start_ppid is not None and start_ppid != os.getppid():
                logger.error(
                    "In daemonic ResultChunk reader the parent pid changed "
                    + f"from {start_ppid} to {os.getppid()}. Assuming parent crashed and exiting."
                )
                try:
                    queue.put(END_OF_STREAM_OBJECT)
                    queue.close()
                except Exception as ex:  # pylint: disable=broad-exception-caught
                    logger.error(f"Error while putting EOS object into queue:\n{ex}\n\nExiting anyways.")

                return

            filename_dict = {file_path_dict[file_id]: ranges}
            instance_iterator = dataset_type_dict[dataset_id].read_ranges_from_files(
                filename_dict, parsing_func_dict[dataset_id], server_connection
            )
            for instance in instance_iterator:
                queue.put(instance)

        queue.put(END_OF_STREAM_OBJECT)

        queue.close()

    def _prepare_workloads(self) -> dict[MixtureKey, Workloads]:
        """
        Prepare the workloads for the result index. This function creates a dictionary with the workloads
        per property combination.

        Returns:
            A dictionary with the workloads per property combination
        """
        workloads: dict[MixtureKey, Workloads] = {}
        for property_combination, dataset_entries in self._result_index.items():
            if property_combination not in workloads:
                workloads[property_combination] = []
            for dataset_id, file_entries in dataset_entries.items():
                for file_id, ranges in file_entries.items():
                    workloads[property_combination].append((dataset_id, file_id, ranges))

            #  Shuffle the workloads to ensure that the order of the files is (reproducibly) random
            seed_everything_from_list([property_combination])
            random.shuffle(workloads[property_combination])

        return workloads

    def __iter__(self) -> "ResultChunk":
        self._iterator = self._iterate_samples()
        return self

    def __next__(self) -> Iterator[str]:
        if self._iterator is None:
            raise StopIteration
        try:
            return next(self._iterator)  # type: ignore  # MyPy seems to expect SupportsNext[Iterator[str]] here
        except StopIteration:
            self._iterator = None
            raise
