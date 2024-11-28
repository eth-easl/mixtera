from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger
from mixtera.core.query.mixture import Mixture, MixtureKey

if TYPE_CHECKING:
    from mixtera.core.datacollection.index import ChunkerIndex


@dataclass
class ScheduleEntry:
    start_step: int
    mixture: Mixture


class MixtureSchedule(Mixture):
    """
    Mixture that changes over time based on a schedule plan.
    """

    def __init__(self, chunk_size: int, schedule: list[ScheduleEntry]) -> None:
        """
        Initialize the MixtureSchedule with a schedule plan.

        Args:
            chunk_size: The size of a chunk in number of instances.
            schedule: A list of ScheduleEntry indicating the mixture to use at each training step.
        """
        super().__init__(chunk_size)
        if len(schedule) == 0:
            raise RuntimeError("An empty schedule is tried to be set.")

        for idx, entry in enumerate(schedule):
            if entry.mixture.chunk_size != self.chunk_size:
                logger.warning(
                    "The chunk size of mixture"
                    f"{idx}({entry.mixture.chunk_size}) does not match the schedule chunk size. Adjusting it."
                )
                entry.mixture.chunk_size = self.chunk_size

        self.schedule = sorted(schedule, key=lambda entry: entry.start_step)
        self.current_step = 0

    def mixture_in_rows(self) -> dict[MixtureKey, int]:
        """
        Returns the mixture dictionary for the current training step.

        Returns:
            The mixture dictionary for the current training step.
        """
        return self.current_mixture.mixture_in_rows()

    def process_index(self, chunker_index: "ChunkerIndex") -> None:
        """
        Inform the current mixture about the overall chunker index.

        Args:
            chunker_index: The chunker index.
        """
        self.current_mixture.process_index(chunker_index)

    @property
    def current_mixture(self) -> Mixture:
        """
        Get the mixture corresponding to the current training step.

        Returns:
            The mixture for the current training step.
        """
        # Retrieve the mixture based on the current step from the schedule
        for entry in reversed(self.schedule):
            if self.current_step >= entry.start_step:
                return entry.mixture
        # Default to the first mixture if current_step is before any start_step
        return self.schedule[0].mixture

    def __str__(self) -> str:
        """
        String representation of this mixture schedule.

        Returns:
            A string describing the mixture schedule.
        """
        schedule_str = ", ".join(
            [f"(start_step: {entry.start_step}, mixture: {entry.mixture})" for entry in self.schedule]
        )
        return f'{{"mixture": "mixture_schedule", "schedule": [{schedule_str}], "chunk_size": {self.chunk_size}}}'
