import unittest

from mixtera.core.datacollection.index import infer_mixture_from_chunkerindex
from mixtera.core.query.mixture import MixtureKey, StaticMixture


class TestIndexUtils(unittest.TestCase):
    def test_infer_mixture(self):
        mock_result_index = {
            MixtureKey({"property1": []}): {0: {0: [(0, 10), (20, 30)]}},
            MixtureKey({"property2": []}): {0: {0: [(0, 5)]}, 1: {0: [(5, 15)]}},
        }
        expected_partition_masses = {MixtureKey({"property1": []}): 20, MixtureKey({"property2": []}): 15}

        total_size, prob_mixture = infer_mixture_from_chunkerindex(mock_result_index)
        self.assertTrue(isinstance(prob_mixture, dict), f"prob_mixture is {prob_mixture}")
        self.assertEqual(total_size, 35)

        # We run it through a static mixture to get a testable representation
        static_mixture = StaticMixture(total_size, prob_mixture)
        test_mixture = static_mixture.mixture_in_rows()
        self.assertEqual(test_mixture, expected_partition_masses)


if __name__ == "__main__":
    unittest.main()
