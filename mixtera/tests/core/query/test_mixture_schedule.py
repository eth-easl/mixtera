import unittest

from mixtera.core.query.mixture import MixtureKey, MixtureSchedule, ScheduleEntry, StaticMixture


class TestMixtureSchedule(unittest.TestCase):

    def test_mixture_schedule(self):
        # Creating a schedule list
        chunk_size = 5

        mixture1 = StaticMixture(chunk_size, {MixtureKey({"language": ["JavaScript"]}): 1.0})
        mixture2 = StaticMixture(chunk_size, {MixtureKey({"language": ["HTML"]}): 1.0})

        schedule_list = [ScheduleEntry(0, mixture1), ScheduleEntry(200, mixture2)]

        mixture_schedule = MixtureSchedule(chunk_size, schedule_list)

        # Testing that it gives the corresponding mixture for a training step.
        mixture_schedule.current_step = 0
        current_mixture = mixture_schedule.current_mixture()
        assert (
            current_mixture.mixture_in_rows()[MixtureKey({"language": ["JavaScript"]})] == 5
        ), "Wrong mixture is used from the schedule."

        mixture_schedule.current_step = 300
        assert (
            mixture_schedule.current_mixture.mixture_in_rows()[MixtureKey({"language": ["HTML"]})] == 5
        ), "Wrong mixture is used from the schedule."


if __name__ == "__main__":
    unittest.main()
