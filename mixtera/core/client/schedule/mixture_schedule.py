from dataclasses import dataclass
from typing import TYPE_CHECKING

from mixtera.core.query import ArbitraryMixture, Mixture, MixtureKey

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
        self.schedule = sorted(schedule, key=lambda entry: entry.start_step)
        self.current_step = 0

    def is_schedule_defined(self) -> bool:
        """
        Returns if there is an existing schedule.
        """
        return len(self._schedule) > 0

    def set_current_step(self, current_step: int) -> None:
        """
        Set the current training step.

        Args:
            step: The current training step.
        """
        self.current_step = current_step

    def mixture_in_rows(self) -> dict[MixtureKey, int]:
        """
        Returns the mixture dictionary for the current training step.

        Returns:
            The mixture dictionary for the current training step.
        """
        current_mixture = self._get_current_mixture()
        return current_mixture.mixture_in_rows()

    def inform(self, chunker_index: "ChunkerIndex") -> None:
        """
        Inform the current mixture about the overall chunker index.

        Args:
            chunker_index: The chunker index.
        """
        current_mixture = self.get_current_mixture()
        current_mixture.inform(chunker_index)

    def get_current_mixture(self) -> Mixture:
        """
        Get the mixture corresponding to the current training step.

        Returns:
            The mixture for the current training step.
        """
        # Retrieve the mixture based on the current step from the schedule
        for entry in reversed(self.schedule):
            assert entry.mixture.chunk_size == self.chunk_size, "The chunk size of the mixtures does not match."
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

    def define_arbitrary_schedule(self, mixture_count: int = 5, interval_length: int = 200) -> None:
        """
        Defines a schedule with arbitrary mixtures in every interval.

        Args:
            mixture_count: The length of the schedule.
            interval_length: The interval length for the mixture.
        """
        start = 0
        self.schedule = []

        for _ in range(mixture_count):
            self._schedule.append(ScheduleEntry(start, ArbitraryMixture(self.chunk_size)))
            start += interval_length

    def add_to_schedule(self, start_step: int, mixture: Mixture) -> None:
        """
        Adds a new mixture to the schedule.

        Args:
            start_step: The starting training step for that mixture in the schedule.
            mixture: Mixture to be added to the schedule.
        """
        assert mixture.chunk_size == self.chunk_size, "The chunk size of the mixtures does not match."
        for item in self.schedule:
            # If the given starting step already existing, we just update the mixture.
            if item.start_step == start_step:
                item.mixture = mixture
                return

        self.schedule.append(ScheduleEntry(start_step, mixture))
        self.schedule = sorted(self.schedule, key=lambda item: item.start_step)
