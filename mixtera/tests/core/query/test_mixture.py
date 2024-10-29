import unittest

from mixtera.core.query import Component, HierarchicalStaticMixture, MixtureKey, MixtureNode


class TestMixture(unittest.TestCase):

    def test_hiearchical_mixture_correctness_multiple_values(self):
        # Submixture for Turkish
        turkish_submixture = MixtureNode(
            property_name="topic",
            components=[Component(value=["law", "medicine"], weight=0.5), Component(value=["physics"], weight=0.5)],
        )

        # Testing a single component with a submixture with list of values.
        language_mixture = MixtureNode(
            property_name="language", components=[Component(value=["Turkish"], weight=1, submixture=turkish_submixture)]
        )

        mixture = HierarchicalStaticMixture(
            100,  # Test chunk size.
            language_mixture,
        )

        assert (
            mixture._mixture[MixtureKey({"language": ["Turkish"], "topic": ["law", "medicine"]})] == 50
        ), "The portions are not correct for Turkish mixture."  # chunk_size * 0.5
        assert (
            mixture._mixture[MixtureKey({"language": ["Turkish"], "topic": ["physics"]})] == 50
        ), "The portions are not correct for Turkish mixture."  # chunk_size * 0.5

    def test_hiearchical_mixture_correctness(self):
        # Submixtures for English and German
        english_submixture = MixtureNode(
            property_name="topic",
            components=[Component(value=["law"], weight=0.5), Component(value=["medicine"], weight=0.5)],
        )
        german_submixture = MixtureNode(
            property_name="license",
            components=[Component(value=["CC"], weight=0.5), Component(value=["MIT"], weight=0.5)],
        )

        # Top-level mixture for language
        language_mixture = MixtureNode(
            property_name="language",
            components=[
                Component(value=["English"], weight=0.6, submixture=english_submixture),
                Component(value=["German"], weight=0.4, submixture=german_submixture),
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
