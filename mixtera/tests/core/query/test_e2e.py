import inspect
import json
import tempfile
import unittest
from pathlib import Path

from mixtera.core.client import MixteraClient
from mixtera.core.client.mixtera_client import QueryExecutionArgs
from mixtera.core.datacollection.datasets.jsonl_dataset import JSONLDataset
from mixtera.core.query import ArbitraryMixture, Query


class TestQueryE2E(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        self.directory = Path(self.temp_dir.name)
        client = MixteraClient.from_directory(self.directory)

        jsonl_file_path1 = self.directory / "temp1.jsonl"
        with open(jsonl_file_path1, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "text": "",
                    "meta": {
                        "content_hash": "4765aae0af2406ea691fb001ea5a83df",
                        "language": [{"name": "Go", "bytes": "734307"}, {"name": "Makefile", "bytes": "183"}],
                    },
                },
                f,
            )
            f.write("\n")
            json.dump(
                {
                    "text": "",
                    "meta": {
                        "content_hash": "324efbc1ad28fdfe902cd1e51f7e095e",
                        "language": [{"name": "Go", "bytes": "366"}, {"name": "CSS", "bytes": "39144"}],
                    },
                },
                f,
            )

        jsonl_file_path2 = self.directory / "temp2.jsonl"
        with open(jsonl_file_path2, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "text": "",
                    "meta": {
                        "content_hash": "324efbc1ad28fdfe902cd1e51f7e095e",
                        "language": [{"name": "ApacheConf", "bytes": "366"}, {"name": "CSS", "bytes": "39144"}],
                    },
                },
                f,
            )

        def parsing_func(data):
            return f"prefix_{data}"

        self.parsing_func_source = inspect.getsource(parsing_func)
        client.register_dataset("test_dataset", str(self.directory), JSONLDataset, parsing_func, "RED_PAJAMA")
        files = client._mdc._get_all_files()
        self.file1_id = [file_id for file_id, _, _, path in files if "temp1.jsonl" in path][0]
        self.file2_id = [file_id for file_id, _, _, path in files if "temp2.jsonl" in path][0]
        self.client = client

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_query_select(self):
        mixture = ArbitraryMixture(1)
        query = Query.for_job("job_id").select(("language", "==", "Go"))
        args = QueryExecutionArgs(mixture=mixture)
        assert self.client.execute_query(query, args)
        res = list(iter(query.results))
        for x in res:
            self.assertEqual(x, {"language:Go": {1: {self.file1_id: [(0, 1)]}}})
            break

    def test_union(self):
        mixture = ArbitraryMixture(1)
        query_1 = Query.for_job("job_id").select(("language", "==", "Go"))
        query_2 = Query.for_job("job_id")
        query_2.select(("language", "==", "CSS"))
        query_2 = query_2.union(query_1)
        args = QueryExecutionArgs(mixture=mixture)
        assert self.client.execute_query(query_2, args)
        query_result = query_2.results
        res = list(iter(query_result))

        # TODO(#41): We should update the test case once we have the
        # deduplication operator and `deduplicate` parameter in Union

        self.assertCountEqual(
            res,
            [
                {"language:Go": {1: {self.file1_id: [(0, 1)]}}},
                {"language:Go": {1: {self.file1_id: [(1, 2)]}}},
                {"language:CSS": {1: {self.file2_id: [(0, 1)]}}},
            ],
        )
        # check metadata
        self.assertEqual(query_result.dataset_type, {1: JSONLDataset})
        self.assertEqual(
            query_result.file_path,
            {self.file1_id: f"{self.directory}/temp1.jsonl", self.file2_id: f"{self.directory}/temp2.jsonl"},
        )
        parsing_func = {k: inspect.getsource(v) for k, v in query_result.parsing_func.items()}
        self.assertEqual(
            parsing_func,
            {1: self.parsing_func_source},
        )


if __name__ == "__main__":
    unittest.main()
