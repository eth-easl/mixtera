import os
from typing import Any

from loguru import logger
import psutil
from mixtera.network.client.client_feedback import ClientFeedback
from mixtera.utils.dataset_utils import _recover_mixtera_dataset
from mixtera.utils.utils import to_numpy_array


def log_open_file_descriptors():
    process = psutil.Process(os.getpid())
    num_fds = process.num_fds()
    logger.debug(f"Number of open file descriptors: {num_fds}")

def log_open_connections():
    process = psutil.Process(os.getpid())
    connections = process.connections()
    num_conns = len(connections)
    logger.debug(f"Number of open connections: {num_conns}")
    states = {}
    for conn in connections:
        state = conn.status
        states[state] = states.get(state, 0) + 1
    logger.debug(f"Connection states: {states}")

def log_time_wait_sockets():
    process = psutil.Process(os.getpid())
    connections = process.connections()
    time_wait_count = sum(1 for conn in connections if conn.status == psutil.CONN_TIME_WAIT)
    logger.debug(f"Number of sockets in TIME_WAIT: {time_wait_count}")


def handle_mixtera_feedback(
    dataloader_or_dataset: Any, training_steps: int, losses: Any, counts: Any, dp_rank: int, tp_rank: int
) -> None:
    assert training_steps >= 0, "Invalid number of training steps are received."
    log_open_file_descriptors()
    log_open_connections()
    log_time_wait_sockets()
    
    if dp_rank != 0 or tp_rank != 0:
        return

    # Every pipeline stage with dp=0 and tp=0 will send this,
    # however, only the output stage will send the current steps with losses (if any).
    # Sending the same step multiple times is not harmful, and this is the easiest solution.

    if (torch_dataset := _recover_mixtera_dataset(dataloader_or_dataset)) is None:
        return

    losses_np = None if losses is None else to_numpy_array(losses)
    counts_np = None if counts is None else to_numpy_array(counts)

    mixture_id = torch_dataset._client.current_mixture_id

    assert mixture_id is not None, "mixture_id is None!"

    feedback = ClientFeedback(training_steps=training_steps, losses=losses_np, counts=counts_np, mixture_id=mixture_id)
    job_id = torch_dataset._query.job_id

    success = torch_dataset._client.process_feedback(job_id, feedback)
    if not success:
        logger.error("Error while processing client feedback.")
