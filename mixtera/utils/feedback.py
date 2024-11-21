from typing import Any

from loguru import logger
from mixtera.network.client.client_feedback import ClientFeedback
from mixtera.utils.dataset_utils import _recover_mixtera_dataset


def handle_mixture_schedule_update(
    dataloader_or_dataset: Any, training_steps: int, dp_group_id: int, node_id: int
) -> None:
    assert training_steps >= 0, "Invalid number of training steps are received."

    if (torch_dataset := _recover_mixtera_dataset(dataloader_or_dataset)) is None:
        return

    # Creating the feedback object and sending to the server.
    feedback = ClientFeedback(training_steps)
    job_id = torch_dataset._query.job_id

    # Updating the schedule according to the feedback of the first node.
    if dp_group_id == 0 and node_id == 0:
        logger.debug(f"[DP Group {dp_group_id}][Node {node_id}] Training step is {training_steps}")
        success = torch_dataset._client.process_feedback(job_id, feedback)
        if success:
            logger.info("The mixture schedule is updated.")
