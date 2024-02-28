import json
import tempfile
import unittest
from pathlib import Path

from mixtera.core.datacollection.datasets.jsonl_dataset import JSONLDataset
from mixtera.core.datacollection.local import LocalDataCollection
from mixtera.core.query import Query


class TestQueryE2E(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        self.directory = Path(self.temp_dir.name)
        ldc = LocalDataCollection(self.directory)

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

        ldc.register_dataset(
            "test_dataset", str(self.directory), JSONLDataset, lambda data: f"prefix_{data}", "RED_PAJAMA"
        )
        self.mdc = ldc

    def test_query_select(self):
        query = Query.from_datacollection(self.mdc)
        query.select(("language", "==", "Go"))
        res = query.execute(chunk_size=1)
        res = list(res)
        self.assertEqual(res, [[{1: {1: [(0, 2)]}}]])

    def test_union(self):
        query_1 = Query.from_datacollection(self.mdc).select(("language", "==", "Go"))
        query_2 = Query.from_datacollection(self.mdc)
        query_2.select(("language", "==", "CSS"))
        query_2 = query_2.union(query_1)
        query_result = query_2.execute(chunk_size=1)
        res = list(query_result)
        self.assertEqual(res, [[{1: {1: [(0, 2)]}}], [{1: {1: [(1, 2)], 2: [(0, 1)]}}]])
        # check metadata
        self.assertEqual(query_result.dataset_type, {1: JSONLDataset})
        self.assertEqual(
            query_result.file_path, {1: f"{self.directory}/temp1.jsonl", 2: f"{self.directory}/temp2.jsonl"}
        )


if __name__ == "__main__":
    unittest.main()
