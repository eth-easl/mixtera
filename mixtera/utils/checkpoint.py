from pathlib import Path
from time import sleep
from typing import Any

from loguru import logger
from mixtera.torch import MixteraTorchDataset
from torch.utils.data import DataLoader


def _get_mixtera_hf_dataset_from_iterabledataset(dataset: Any):
    from mixtera.hf.mixtera_hf_dataset import (
        MixteraHFDataset,  # inline import for people who do not have datasets installed.
    )

    if isinstance(dataset, MixteraHFDataset):
        return dataset

    ex_iterable = getattr(dataset, "_ex_iterable", None)
    while ex_iterable is not None:
        if isinstance(ex_iterable, MixteraHFDataset):
            return ex_iterable
        ex_iterable = getattr(ex_iterable, "ex_iterable", None)
    return None


def _recover_mixtera_dataset(dataloader_or_dataset: DataLoader | Any) -> MixteraTorchDataset | None:
    if isinstance(dataloader_or_dataset, DataLoader):
        dataset = dataloader_or_dataset.dataset
    else:
        dataset = dataloader_or_dataset

    if not isinstance(dataset, MixteraTorchDataset):
        try:
            import datasets
        except ImportError:
            logger.debug(f"Cannot import datasets - and is not a `MixteraTorchDataset`. No Mixtera Checkpoint.")
            return None

        if not isinstance(dataset, datasets.IterableDataset):
            logger.debug(
                f"Dataset is neither `MixteraTorchDataset` nor `datasets.IterableDataset`. No Mixtera Checkpoint."
            )
            return None

        # Now, it could still be any IterableDataset.
        # Since we can apply arbitrary transformations, we need to recover the mixtera dataset
        og_type = type(dataset)
        if (dataset := _get_mixtera_hf_dataset_from_iterabledataset(dataset)) is None:
            logger.debug(
                f"Dataset is `datasets.IterableDataset`, but could not find `MixteraHFDataset` (type = {og_type}). No Mixtera Checkpoint."
            )
            return None

    return dataset if isinstance(dataset, MixteraTorchDataset) else dataset._ex_iterable


def handle_mixtera_checkpoint(
    dataloader_or_dataset: DataLoader | Any, checkpoint_path: Path, dp_group_id: int, node_id: int
) -> None:
    assert checkpoint_path.is_dir()

    if (torch_dataset := _recover_mixtera_dataset(dataloader_or_dataset)) is None:
        return

    # Collect relevant infos
    worker_status = torch_dataset.worker_status
    job_id = torch_dataset._query.job_id

    # Send worker status for this dp_group to server
    # Receive back from server checkpoint id, store that in checkpoint_path / mixtera.id
    # TODO(create issue): Make separate process configurable
    checkpoint_id = torch_dataset._client.checkpoint(job_id, dp_group_id, node_id, worker_status, True)
    logger.debug(f"[DP Group {dp_group_id}][Node {node_id}] Checkpoint ID is {checkpoint_id}")

    if node_id == 0 and dp_group_id == 0:
        with open(checkpoint_path / "mixtera.id", "w+", encoding="utf-8") as fp:
            fp.write(checkpoint_id)

    # At server: validate consistency [dont crash if not], since we store chunk only once in case of inconsistency pick highest number of samples.

    checkpoint_completed = dataset._client.checkpoint_completed(checkpoint_id, False)
    while not checkpoint_written:  # Busy wait until server has finished.
        checkpoint_written = dataset._client.checkpoint_completed(checkpoint_id, False)
        sleep(0.1)

    # Sanity check:
    with open(checkpoint_path / "mixtera.id", "r", encoding="utf-8") as fp:
        written_chkpnt = fp.readlines()

    if written_chkpnt != checkpoint_id:
        raise RuntimeError(
            f"[DP Group {dp_group_id}][Node {node_id}] Inconsistent checkpoint state: {written_chkpnt} vs {checkpoint_id}"
        )

    if node_id == 0 and dp_group_id == 0:
        logger.info("Finalized Mixtera Checkpoint.")
