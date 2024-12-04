from typing import Any

from loguru import logger
from mixtera.network.client.client_feedback import ClientFeedback
from mixtera.utils.dataset_utils import _recover_mixtera_dataset
from mixtera.utils.utils import to_numpy_array


def handle_mixtera_feedback(
    dataloader_or_dataset: Any, training_steps: int, losses: Any, counts: Any, dp_rank: int, tp_rank: int
) -> None:
    assert training_steps >= 0, "Invalid number of training steps are received."

    if dp_rank != 0 or tp_rank != 0:
        return

    # Every pipeline stage with dp=0 and tp=0 will send this,
    # however, only the output stage will send the current steps with losses (if any).
    # Sending the same step multiple times is not harmful, and this is the easiest solution.

    if (torch_dataset := _recover_mixtera_dataset(dataloader_or_dataset)) is None:
        return

    losses_np = None if losses is None else to_numpy_array(losses)
    counts_np = None if counts is None else to_numpy_array(losses)

    mixture_id = torch_dataset._client.current_mixture_id

    assert mixture_id is not None, "mixture_id is None!"

    feedback = ClientFeedback(training_steps=training_steps, losses=losses_np, counts=counts_np, mixture_id=mixture_id)
    job_id = torch_dataset._query.job_id

    success = torch_dataset._client.process_feedback(job_id, feedback)
    if not success:
        logger.error("Error while procesisng client feedback.")
