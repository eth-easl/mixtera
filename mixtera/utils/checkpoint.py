from pathlib import Path
from time import sleep
from typing import Any

from loguru import logger

from mixtera.utils.dataset_utils import _recover_mixtera_dataset


def handle_mixtera_checkpoint(
    dataloader_or_dataset: Any, checkpoint_path: Path | str, dp_group_id: int, node_id: int, wait_for_disk: bool
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
    checkpoint_path = checkpoint_path if isinstance(checkpoint_path, Path) else Path(checkpoint_path)
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
