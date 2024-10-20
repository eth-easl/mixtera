import unittest

from mixtera.core.query import HierarchicalStaticMixture, MixtureKey


class TestMixture(unittest.TestCase):
    def test_hiearchical_mixture_correctness(self):
        # An example hiearchical mixture definition.
        hiearchical_mixture = [
            {
                "language": "JavaScript",
                "portion": 0.6,
                "submixture": [
                    {
                        "licence": "CC",
                        "portion": 0.5,
                        "submixture": [
                            {"topic": "law", "portion": 0.8, "submixture": []},
                            {"topic": "medicine", "portion": 0.2, "submixture": []},
                        ],
                    },
                    {"licence": "All rights reserved", "portion": 0.5, "submixture": []},
                ],
            },
            {"language": "HTML", "portion": 0.4, "submixture": []},
        ]

        mixture = HierarchicalStaticMixture(
            100,  # Test chunk size.
            hiearchical_mixture,
        )

        assert (
            mixture._mixture[MixtureKey({"language": ["JavaScript"], "licence": ["CC"], "topic": ["law"]})]
            == 24  # chunk_size * 0.24
        ), "The portions are not correct."
        assert (
            mixture._mixture[MixtureKey({"language": ["JavaScript"], "licence": ["CC"], "topic": ["medicine"]})]
            == 6  # chunk_size * 0.06
        ), "The portions are not correct."
        assert (
            mixture._mixture[MixtureKey({"language": ["JavaScript"], "licence": ["All rights reserved"]})]
            == 30  # chunk_size * 0.3
        ), "The portions are not correct."
        assert (
            mixture._mixture[MixtureKey({"language": ["HTML"]})] == 40  # chunk_size * 0.4
        ), "The portions are not correct."


if __name__ == "__main__":
    unittest.main()
