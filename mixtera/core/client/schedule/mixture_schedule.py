import math

from mixtera.core.query import Mixture, ArbitraryMixture
from loguru import logger

MixtureInterval = tuple[int, int, Mixture]

class MixtureSchedule:
    """
    A class to represent the mixture schedule for MixteraClient.
    """
    def __init__(self, chunk_size: int):
        self._chunk_size = chunk_size
        self._schedule: list[MixtureInterval] = [] # Start, length, mixture of the interval

    def create_schedule(self, mixture_list: list[MixtureInterval]):
        self._schedule = mixture_list[:]
        
    def __str__(self) -> str:
        """String representation of this mixture schedule."""
        result = "The client has the following schedule: \n"
        for item in self._schedule:
            result += f"Start: {item[0]}, End: {item[0] + item[1]}, Mixture: {item[2]} \n"
        return result

    def is_schedule_defined(self) -> bool:
        """
        Returns if there is an existing schedule.
        """
        return len(self._schedule) > 0
    
    def define_arbitrary_schedule(self, schedule_length: int = 1000, interval_length: int = 200):
        """
        Defines a schedule with arbitrary mixtures in every interval.

        Args:
            schedule_length: The number of training steps for the client.
            interval_length: The length for a interval in training. Number of updates will be schedule_length / interval_length.

        Returns:
            ChunkerIndex: A nested dictionary mapping mixture keys to dataset IDs, file IDs, and intervals.
        """
        interval_count = math.ceil(schedule_length / interval_length)
        start = 0

        for _ in range(interval_count):
            self._schedule.append(start, interval_length, ArbitraryMixture(self._chunk_size))
            start += self._interval_size
    
    def get_current_mixture(self, step: int) -> Mixture | None:
        """
        Finding the mixture that should be applied in the given training step for predefined schedules.

        Args:
            step: The step which the mixture is looked for.

        Returns:
            Mixture: The mixture belonging to the step.
        """
        assert self.is_schedule_defined(), "There is no defined schedule."
        
        for interval in self._schedule:
            if step >= interval[0] and step < interval[0] + interval[1]:
                logger.info(f"Returning the mixture for {step} in the schedule.")
                return interval[2]
        
        logger.error(f"There is no defined mixture for the step {step}.")
        return None
        
    def add_to_schedule(self, training_length: int, mixture: Mixture):
        """
        Adds a new mixture to the schedule.

        Args:
            training_length: The duration that the client should be trained on that mixture.
            mixture: Mixture to be trained on.
        """
        last_start, last_length, _ = self._schedule[len(self._schedule) - 1]
        start = last_length + last_start
        self._schedule.append((start, training_length, mixture))

    


