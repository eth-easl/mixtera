from pathlib import Path
from time import sleep
from typing import Any

import torch
from loguru import logger
from mixtera.torch import MixteraTorchDataset


def _get_mixtera_hf_dataset_or_client_from_iterabledataset(dataset: Any) -> Any:
    """
    Recursively retrieves a `MixteraHFDataset` from a potentially nested `datasets.IterableDataset`.

    This function attempts to extract the original `MixteraHFDataset` from a dataset that might have
    undergone several transformations or wrappers, resulting in a nested structure of `IterableDataset` instances.
    It navigates through the `_ex_iterable` or `ex_iterable` attributes to find the underlying `MixteraHFDataset`.

    Args:
        dataset (Any): The dataset to search through. It can be any object, but typically a `datasets.IterableDataset`.

    Returns:
        Any: The found `MixteraHFDataset` instance if present; otherwise, `None`.

    Note:
        - This function performs an inline import of `MixteraHFDataset` to avoid requiring the `datasets` library
          for users who do not have it installed.
        - The search relies on the presence of `_ex_iterable` or `ex_iterable` attributes, which are common when
          datasets are wrapped with transformations or other dataset utilities.
    """
    # inline import for people who do not have datasets installed.
    from mixtera.hf.mixtera_hf_dataset import (  # pylint: disable=import-outside-toplevel
        MixteraHFDataset,
        _MixteraHFIterable,
    )

    visited = set()
    to_visit = [dataset]

    while to_visit:
        current = to_visit.pop()
        if id(current) in visited:
            continue  # Avoid infinite loops in circular references
        visited.add(id(current))

        if isinstance(current, (MixteraHFDataset, _MixteraHFIterable)):
            return current

        # Get both '_ex_iterable' and 'ex_iterable' attributes - it's a bit inconsistent when which is used.
        next_iterable = getattr(current, "_ex_iterable", None)
        if next_iterable is not None:
            to_visit.append(next_iterable)
        next_iterable = getattr(current, "ex_iterable", None)
        if next_iterable is not None:
            to_visit.append(next_iterable)

    return None


def _recover_mixtera_dataset(dataloader_or_dataset: Any) -> MixteraTorchDataset | None:
    """
    Attempts to recover a `MixteraTorchDataset` from a provided DataLoader or Dataset.

    This function handles cases where the dataset might be wrapped in a DataLoader or
    have undergone transformations that wrap it in an `IterableDataset`.
    It navigates through potential wrappers to find the underlying `MixteraTorchDataset` or `MixteraHFDataset`.

    Args:
        dataloader_or_dataset (Any): The DataLoader or Dataset instance to recover from.

    Returns:
        MixteraTorchDataset | None: The recovered `MixteraTorchDataset` if found; otherwise, `None`.

    Note:
        - If the input is a DataLoader, the function accesses its `.dataset` attribute.
        - The function first checks if the dataset is an instance of `MixteraTorchDataset`.
        - If not, it attempts to import the `datasets` library and checks if the dataset is an `IterableDataset`.
        - It then uses `_get_mixtera_hf_dataset_from_iterabledataset` to search for a `MixteraHFDataset`.
        - If a `MixteraHFDataset` is found, it returns it; otherwise, the function returns `None`.
    """
    logger.debug(f"Type of received object is {type(dataloader_or_dataset)}")
    if isinstance(dataloader_or_dataset, torch.utils.data.DataLoader):  # type: ignore
        dataset = dataloader_or_dataset.dataset
    elif isinstance(dataloader_or_dataset, torch.utils.data.Dataset):  # type: ignore
        dataset = dataloader_or_dataset
    else:
        # Perhaps a generator from sanity_check_dataloader in Nanotron.
        iterator = dataloader_or_dataset
        dataset = dataloader_or_dataset
        try:
            iterator_frame = getattr(iterator, "gi_frame", None)
            if iterator_frame is not None:
                f_locals = iterator_frame.f_locals
                if "dataloader" in f_locals:
                    dataloader = f_locals["dataloader"]
                    if isinstance(dataloader, torch.utils.data.DataLoader):  # type: ignore
                        logger.debug("Recovered DataLoader from generator frame!")
                        dataset = dataloader.dataset
                    else:
                        logger.debug("The 'dataloader' in generator locals is not a DataLoader.")
                else:
                    logger.debug("Could not find 'dataloader' in generator frame locals.")
            else:
                logger.debug("The generator does not have a 'gi_frame' attribute.")
        except AttributeError as e:
            logger.debug(f"Could not access generator frame: {e}")

    if not isinstance(dataset, MixteraTorchDataset):
        try:
            import datasets  # pylint: disable=import-outside-toplevel
        except ImportError:
            logger.debug("Cannot import datasets - and is not a `MixteraTorchDataset`. No Mixtera Checkpoint.")
            return None

        if not isinstance(dataset, datasets.IterableDataset):
            logger.debug(
                "Dataset is neither `MixteraTorchDataset` nor `datasets.IterableDataset`. No Mixtera Checkpoint."
            )
            return None

        # Now, it could still be any IterableDataset.
        # Since we can apply arbitrary transformations, we need to recover the mixtera dataset
        og_type = type(dataset)
        if (dataset := _get_mixtera_hf_dataset_or_client_from_iterabledataset(dataset)) is None:
            logger.debug(
                "Dataset is `datasets.IterableDataset`, but could not find `MixteraHFDataset`"
                + f" (type = {og_type}). No Mixtera Checkpoint."
            )
            return None

        from mixtera.hf.mixtera_hf_dataset import (  # pylint: disable=import-outside-toplevel
            MixteraHFDataset,
            _MixteraHFIterable,
        )

        if isinstance(dataset, MixteraHFDataset):
            dataset = dataset._ex_iterable

        if not isinstance(dataset, _MixteraHFIterable):
            logger.debug(f"Unexpected type: {type(dataset)}. No Mixtera Checkpoint.")
            return None

    return dataset if isinstance(dataset, MixteraTorchDataset) else dataset


def handle_mixtera_checkpoint(
    dataloader_or_dataset: Any, checkpoint_path: Path, dp_group_id: int, node_id: int, wait_for_disk: bool
) -> None:
    """
    Handles the checkpointing process for a Mixtera dataset during training.

    This function initiates a checkpoint operation by collecting the current worker statuses
    and communicating with the Mixtera client. It ensures that the checkpoint is properly saved
    and synchronized across different nodes in a distributed training setup.

    Args:
        dataloader_or_dataset (Any): The DataLoader or Dataset being used in training.
                                     Should be or contain a `MixteraTorchDataset`.
        checkpoint_path (Path): The directory path where the checkpoint should be saved.
        dp_group_id (int): The data parallel group ID (e.g., for distributed training).
        node_id (int): The node ID within the data parallel group.
        wait_for_disk (bool): If `True`, the function waits until the checkpoint is fully written to disk
                              before proceeding. If `False`, it proceeds once the checkpoint is stored in memory.
                              Recommended to set to False for speedy training.

    Returns:
        None

    Raises:
        AssertionError: If `checkpoint_path` is not a directory.
        RuntimeError: If there is an inconsistency in the checkpoint state across nodes.

    Note:
        - The function first recovers the `MixteraTorchDataset` from the provided input.
        - It collects the worker statuses and job ID from the dataset.
        - It communicates with the Mixtera client to initiate the checkpoint and waits for completion.
        - The checkpoint ID is written to a file at `checkpoint_path / "mixtera.id"` for synchronization.
        - Only the node with `node_id == 0` and `dp_group_id == 0` performs the file write and final logging.
    """
    assert checkpoint_path.is_dir()

    if (torch_dataset := _recover_mixtera_dataset(dataloader_or_dataset)) is None:
        return

    # Collect relevant infos
    worker_status = torch_dataset.worker_status
    job_id = torch_dataset._query.job_id

    # Send worker status for this dp_group to server
    # Receive back from server checkpoint id, store that in checkpoint_path / mixtera.id
    logger.debug(f"[DP Group {dp_group_id}][Node {node_id}] Reporting worker status {worker_status}")
    checkpoint_id = torch_dataset._client.checkpoint(job_id, dp_group_id, node_id, worker_status)

    if node_id == 0 and dp_group_id == 0:
        logger.debug(f"[DP Group {dp_group_id}][Node {node_id}] Checkpoint ID is {checkpoint_id}")
        with open(checkpoint_path / "mixtera.id", "w+", encoding="utf-8") as fp:
            fp.write(checkpoint_id)

    while not torch_dataset._client.checkpoint_completed(job_id, checkpoint_id, wait_for_disk):
        sleep(0.1)

    if node_id == 0 and dp_group_id == 0:
        logger.info("Finalized Mixtera Checkpoint.")
