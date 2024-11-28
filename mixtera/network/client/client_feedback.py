from dataclasses import dataclass

import numpy as np


@dataclass
class ClientFeedback:
    training_steps: int = 0
    losses: np.ndarray | None = None
    counts: np.ndarray | None = None
