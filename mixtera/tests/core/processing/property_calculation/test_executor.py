import unittest
from unittest.mock import Mock

from mixtera.core.processing import ExecutionMode
from mixtera.core.processing.property_calculation import LocalPropertyCalculationExecutor, PropertyCalculationExecutor


class TestPropertyCalculationExecutor(unittest.TestCase):
    def test_from_mode_raises_error_with_degree_of_parallelism_less_than_1(self):
        with self.assertRaises(RuntimeError) as context:
            PropertyCalculationExecutor.from_mode(ExecutionMode.LOCAL, 0, 10, Mock(), Mock())
        self.assertEqual(str(context.exception), "Degree of parallelism = 0 < 1")

    def test_from_mode_raises_error_with_batch_size_less_than_1(self):
        with self.assertRaises(RuntimeError) as context:
            PropertyCalculationExecutor.from_mode(ExecutionMode.LOCAL, 1, 0, Mock(), Mock())
        self.assertEqual(str(context.exception), "Batch size = 0 < 1")

    def test_from_mode_returns_local_executor_instance(self):
        setup_func = Mock()
        calc_func = Mock()
        executor = PropertyCalculationExecutor.from_mode(ExecutionMode.LOCAL, 1, 10, setup_func, calc_func)
        self.assertIsInstance(executor, LocalPropertyCalculationExecutor)

    def test_from_mode_raises_not_implemented_for_unknown_mode(self):
        with self.assertRaises(NotImplementedError) as context:
            PropertyCalculationExecutor.from_mode("UNKNOWN_MODE", 1, 10, Mock(), Mock())
        self.assertEqual(str(context.exception), "Mode UNKNOWN_MODE not yet implemented.")
