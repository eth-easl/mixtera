import os
import json
import sqlite3
from loguru import logger
from typing import Callable
import linecache

class MixteraDataset:

    @classmethod
    def from_folder(cls, dataset_path, **kwargs):
        num_proc = kwargs.get("num_proc", 1)
        return cls(dataset_path, int(num_proc))

    def __init__(self, dataset_path, num_proc=1) -> None:
        self.dataset_path = dataset_path
        self.num_proc = num_proc

    def __repr__(self) -> str:
        return f"MixteraDataset({self.dataset_path})"

    def _build_inverted_index(self, process_fn: Callable = None):
        """
        the name is not final...
        inverted index points from metadata to file location

        > e.g., toxicity < 0.5 -> [filename_line_start-end, filename2_line_start-end, ...]
        > right now for simplicity, we just use a single string to represent the key
        > we're using sqlite as a key-value store basically
        """
        logger.info("Creating metadata database...")
        # todo: also check if the table exists
        conn = sqlite3.connect(os.path.join(self.dataset_path, "invert_index.db"))
        cur = conn.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS metadata (key text, value text);")
        conn.commit()
        # load all jsonl files
        results = {}
        for iter in process_fn(self.dataset_path):
            filename, line_id, key = iter
            if key not in results:
                results[key] = {}
            if filename not in results[key]:
                results[key][filename] = []
            results[key][filename].append(line_id)
        logger.info("Inserting metadata into database...")
        for key in results:
            for filename in results[key]:
                value = f"{filename}:{str(results[key][filename])[1:-1]}"
                cur.execute("INSERT INTO metadata VALUES (?, ?);", (key, value))
        conn.commit()

    def build_index(self, process_fn: Callable = None):
        self._build_inverted_index(process_fn=process_fn)

    def prepare(self, process_fn: Callable = None):
        """ """
        logger.info("Preparing dataset...")
        self.build_index(process_fn=process_fn)

    def read_keys(self):
        conn = sqlite3.connect(os.path.join(self.dataset_path, "invert_index.db"))
        cur = conn.cursor()
        cur.execute("SELECT key FROM metadata;")
        return [x[0] for x in cur.fetchall()]

    def find_by_key(self, key):
        conn = sqlite3.connect(os.path.join(self.dataset_path, "invert_index.db"))
        cur = conn.cursor()
        cur.execute("SELECT value FROM metadata WHERE key=?;", (key,))
        res = cur.fetchone()[0]
        # convert it back to a list
        filename = res.split(":")[0]
        line_ids = [int(x) for x in res.split(":")[1].split(",")]
        return [f"{filename}:{x}" for x in line_ids]

    async def stream_values(self, keys):
        for key in keys:
            filename, lineid = key.split(":")
            res = linecache.getline(os.path.join(self.dataset_path, filename), int(lineid)+1)
            yield res

    def read_values(self, keys):
        self.results = []
        for key in keys:
            filename, lineid = key.split(":")
            # TODO(xiaozhe): seems linecache starts at 1...?
            res = linecache.getline(os.path.join(self.dataset_path, filename), int(lineid)+1)
            self.results.append(json.loads(res))
        return self.results