import tempfile
import unittest
from pathlib import Path
from typing import Any, List
from unittest.mock import MagicMock, patch

import polars as pl
from integrationtests.utils import TestMetadataParser as ExampleMetadataParser
from mixtera.core.client.mixtera_client import MixteraClient
from mixtera.core.datacollection.datasets.jsonl_dataset import JSONLDataset
from mixtera.core.query import Operator, Query, QueryPlan
from mixtera.core.query.mixture import ArbitraryMixture
from polars.testing import assert_frame_equal


def parsing_func(sample):
    import json  # pylint: disable=import-outside-toplevel

    return json.loads(sample)["text"]


class TestQuery(unittest.TestCase):

    def setUp(self):
        self.job_id = "test_job_id"
        self.query = Query(self.job_id)
        self.temp_dir = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        self.dir = Path(self.temp_dir.name)
        # Preparation: instantiate a db, prep some data
        # While it would be nice to mock all of this, we need some DB to test the queries on
        # Which would mean instantiating the DB in a mock - it's not clear this is better than
        # just using the client directly.
        with open(self.dir / "data.jsonl", "w", encoding="utf-8") as text_file:
            text_file.write('{ "text": "0", "meta": { "language": "JavaScript", "license": "CC"} }\n')
            text_file.write('{ "text": "1", "meta": { "language": "JavaScript", "license": "CC"} }\n')
            text_file.write('{ "text": "2", "meta": { "language": "HTML", "license": "CC"} }\n')
            text_file.write('{ "text": "3", "meta": { "language": "HTML", "license": "CC"} }\n')
            text_file.write('{ "text": "4", "meta": { "language": "JavaScript", "license": "CC"} }\n')
            text_file.write('{ "text": "5", "meta": { "language": "HTML", "license": "CC"} }\n')
            text_file.write('{ "text": "6", "meta": { "language": "HTML", "license": "MIT"} }\n')
            text_file.write('{ "text": "7", "meta": { "language": "JavaScript", "license": "CC"} }\n')
        self.client = MixteraClient.from_directory(self.dir)
        self.client.register_metadata_parser("TEST_PARSER", ExampleMetadataParser)
        self.client.register_dataset(
            "query_test_execute_dataset", self.dir / "data.jsonl", JSONLDataset, parsing_func, "TEST_PARSER"
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_init(self):
        self.assertIsInstance(self.query.query_plan, QueryPlan)
        self.assertIsNone(self.query.results)
        self.assertEqual(self.query.job_id, self.job_id)

    def test_is_empty(self):
        self.assertTrue(self.query.is_empty())

        mock_operator = MagicMock(spec=Operator)
        self.query.query_plan.add(mock_operator)

        self.assertFalse(self.query.is_empty())

    def test_register(self):
        class TestOperator(Operator):
            def __init__(self, arg1, arg2):
                super().__init__()
                self.arg1 = arg1
                self.arg2 = arg2

            def generate_sql(self, schema: dict) -> tuple[str, List[Any]]:
                return ("test_operator", [])

        Query.register(TestOperator)

        self.assertTrue(hasattr(Query, "testoperator"))

        query = Query(self.job_id)
        result = query.testoperator("value1", arg2="value2")

        self.assertIsInstance(result, Query)
        self.assertIsNotNone(query.query_plan.root)
        self.assertIsInstance(query.query_plan.root, TestOperator)
        self.assertEqual(query.query_plan.root.arg1, "value1")
        self.assertEqual(query.query_plan.root.arg2, "value2")

    def test_for_job(self):
        query = Query.for_job("new_job_id")
        self.assertIsInstance(query, Query)
        self.assertEqual(query.job_id, "new_job_id")

    def test_root(self):
        mock_operator = MagicMock(spec=Operator)
        self.query.query_plan.add(mock_operator)

        self.assertEqual(self.query.root, mock_operator)

    @patch("mixtera.core.query.query_plan.QueryPlan.display")
    def test_display(self, mock_display):
        self.query.display()
        mock_display.assert_called_once()

    def test_str(self):
        mock_operator = MagicMock(spec=Operator)
        mock_operator.string.return_value = "MockOperator"
        self.query.query_plan.add(mock_operator)

        self.assertEqual(str(self.query), "MockOperator")

    @patch("mixtera.core.query.query.QueryResult", autospec=True)
    def test_execute_nofilter(self, mock_query_result):

        expected_df = pl.DataFrame(
            {
                "dataset_id": [1, 1, 1, 1, 1, 1],
                "file_id": [1, 1, 1, 1, 1, 1],
                "interval_start": [2, 5, 6, 0, 4, 7],
                "interval_end": [4, 6, 7, 2, 5, 8],
                "language": ["HTML", "HTML", "HTML", "JavaScript", "JavaScript", "JavaScript"],
                "doublelanguage": [
                    ["HTML", "HTML"],
                    ["HTML", "HTML"],
                    ["HTML", "HTML"],
                    ["JavaScript", "JavaScript"],
                    ["JavaScript", "JavaScript"],
                    ["JavaScript", "JavaScript"],
                ],
                "license": ["CC", "CC", "MIT", "CC", "CC", "CC"],
            }
        )

        query: Query = Query.for_job(("job_id")).select(None)
        query.execute(self.client._mdc, ArbitraryMixture(1))
        self.assertTrue(mock_query_result.called, "QueryResult was not called")

        call_args, _ = mock_query_result.call_args
        result_df: pl.DataFrame = call_args[1]
        result_df.drop_in_place("group_id")
        assert_frame_equal(expected_df, result_df, check_column_order=False, check_dtypes=False)

    @patch("mixtera.core.query.query.QueryResult", autospec=True)
    def test_execute_single_condition(self, mock_query_result):
        query = Query.for_job("job_id").select(("language", "==", "JavaScript"))
        query.execute(self.client._mdc, ArbitraryMixture(1))

        expected_df = pl.DataFrame(
            {
                "dataset_id": [1] * 3,
                "file_id": [1] * 3,
                "interval_start": [0, 4, 7],
                "interval_end": [2, 5, 8],
                "language": ["JavaScript"] * 3,
                "doublelanguage": [["JavaScript", "JavaScript"]] * 3,
                "license": ["CC"] * 3,
            }
        )

        call_args, _ = mock_query_result.call_args
        result_df: pl.DataFrame = call_args[1]
        result_df.drop_in_place("group_id")
        assert_frame_equal(expected_df, result_df, check_column_order=False, check_dtypes=False)

    @patch("mixtera.core.query.query.QueryResult", autospec=True)
    def test_execute_multiple_conditions(self, mock_query_result):
        query = Query.for_job("job_id").select([("language", "==", "JavaScript"), ("license", "==", "CC")])
        query.execute(self.client._mdc, ArbitraryMixture(1))

        expected_df = pl.DataFrame(
            {
                "dataset_id": [1] * 3,
                "file_id": [1] * 3,
                "interval_start": [0, 4, 7],
                "interval_end": [2, 5, 8],
                "language": ["JavaScript"] * 3,
                "doublelanguage": [["JavaScript", "JavaScript"]] * 3,
                "license": ["CC"] * 3,
            }
        )

        call_args, _ = mock_query_result.call_args
        result_df: pl.DataFrame = call_args[1]
        result_df.drop_in_place("group_id")
        assert_frame_equal(expected_df, result_df, check_column_order=False, check_dtypes=False)

    @patch("mixtera.core.query.query.QueryResult", autospec=True)
    def test_execute_or_conditions(self, mock_query_result):
        query = Query.for_job("job_id").select(("language", "==", "JavaScript"))
        query.select(("language", "==", "HTML"))
        query.execute(self.client._mdc, ArbitraryMixture(1))

        expected_df = pl.DataFrame(
            {
                "dataset_id": [1] * 6,
                "file_id": [1] * 6,
                "interval_start": [2, 5, 6, 0, 4, 7],
                "interval_end": [4, 6, 7, 2, 5, 8],
                "language": ["HTML", "HTML", "HTML", "JavaScript", "JavaScript", "JavaScript"],
                "doublelanguage": [
                    ["HTML", "HTML"],
                    ["HTML", "HTML"],
                    ["HTML", "HTML"],
                    ["JavaScript", "JavaScript"],
                    ["JavaScript", "JavaScript"],
                    ["JavaScript", "JavaScript"],
                ],
                "license": ["CC", "CC", "MIT", "CC", "CC", "CC"],
            }
        )

        call_args, _ = mock_query_result.call_args
        result_df: pl.DataFrame = call_args[1]
        result_df.drop_in_place("group_id")
        assert_frame_equal(expected_df, result_df, check_column_order=False, check_dtypes=False)

    @patch("mixtera.core.query.query.QueryResult", autospec=True)
    def test_execute_not_condition(self, mock_query_result):
        query = Query.for_job("job_id").select(("language", "!=", "JavaScript"))
        query.execute(self.client._mdc, ArbitraryMixture(1))

        expected_df = pl.DataFrame(
            {
                "dataset_id": [1] * 3,
                "file_id": [1] * 3,
                "interval_start": [2, 5, 6],
                "interval_end": [4, 6, 7],
                "language": ["HTML"] * 3,
                "doublelanguage": [["HTML", "HTML"]] * 3,
                "license": ["CC", "CC", "MIT"],
            }
        )

        call_args, _ = mock_query_result.call_args
        result_df: pl.DataFrame = call_args[1]
        result_df.drop_in_place("group_id")
        assert_frame_equal(expected_df, result_df, check_column_order=False, check_dtypes=False)

    @patch("mixtera.core.query.query.QueryResult", autospec=True)
    def test_execute_list_containment(self, mock_query_result):
        query = Query.for_job("job_id").select(("language", "==", ["JavaScript", "HTML"]))
        query.execute(self.client._mdc, ArbitraryMixture(1))

        expected_df = pl.DataFrame(
            {
                "dataset_id": [1] * 6,
                "file_id": [1] * 6,
                "interval_start": [2, 5, 6, 0, 4, 7],
                "interval_end": [4, 6, 7, 2, 5, 8],
                "language": ["HTML", "HTML", "HTML", "JavaScript", "JavaScript", "JavaScript"],
                "doublelanguage": [
                    ["HTML", "HTML"],
                    ["HTML", "HTML"],
                    ["HTML", "HTML"],
                    ["JavaScript", "JavaScript"],
                    ["JavaScript", "JavaScript"],
                    ["JavaScript", "JavaScript"],
                ],
                "license": ["CC", "CC", "MIT", "CC", "CC", "CC"],
            }
        )

        call_args, _ = mock_query_result.call_args
        result_df: pl.DataFrame = call_args[1]
        result_df.drop_in_place("group_id")
        assert_frame_equal(expected_df, result_df, check_column_order=False, check_dtypes=False)


if __name__ == "__main__":
    unittest.main()
