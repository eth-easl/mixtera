import unittest

from mixtera.core.query import Component, HierarchicalMixture, HierarchicalStaticMixture, MixtureKey


class TestMixture(unittest.TestCase):
    def test_hiearchical_mixture_correctness(self):
        # Submixtures for English and German
        english_submixture = HierarchicalMixture(
            property_name="topic",
            components=[Component(value="law", weight=0.5), Component(value="medicine", weight=0.5)],
        )
        german_submixture = HierarchicalMixture(
            property_name="license", components=[Component(value="CC", weight=0.5), Component(value="MIT", weight=0.5)]
        )

        # Top-level mixture for language
        language_mixture = HierarchicalMixture(
            property_name="language",
            components=[
                Component(value="English", weight=0.6, submixture=english_submixture),
                Component(value="German", weight=0.4, submixture=german_submixture),
            ],
        )

        mixture = HierarchicalStaticMixture(
            100,  # Test chunk size.
            language_mixture,
        )

        assert (
            mixture._mixture[MixtureKey({"language": ["English"], "topic": ["law"]})] == 30  # chunk_size * 0.3
        ), "The portions are not correct."
        assert (
            mixture._mixture[MixtureKey({"language": ["English"], "topic": ["medicine"]})] == 30  # chunk_size * 0.3
        ), "The portions are not correct."
        assert (
            mixture._mixture[MixtureKey({"language": ["German"], "license": ["CC"]})] == 20  # chunk_size * 0.2
        ), "The portions are not correct."
        assert (
            mixture._mixture[MixtureKey({"language": ["German"], "license": ["MIT"]})] == 20  # chunk_size * 0.2
        ), "The portions are not correct."


if __name__ == "__main__":
    unittest.main()
