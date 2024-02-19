import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from loguru import logger
from mixtera.core.datacollection import IndexType
from mixtera.core.datacollection.datasets import Dataset
from mixtera.utils import ranges


class JSONLDataset(Dataset):
    type_id = 1

    @staticmethod
    def iterate_files(loc: str) -> Iterable[Path]:
        path = Path(loc)

        if not path.exists():
            raise RuntimeError(f"Path {path} does not exist.")

        is_directory = path.is_dir()

        if not is_directory:
            if not JSONLDataset._is_valid_jsonl(path):
                raise RuntimeError(
                    f"Path {path} does not belong to a directory and does not refer to a valid jsonl file."
                )

            yield path

        yield from path.glob("*.jsonl")

    @staticmethod
    def build_file_index(loc: Path, dataset_id: int, file_id: int) -> IndexType:
        index: IndexType = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
        # For now, I just hardcode the SlimPajama example. We have to generalize this to UDFs
        # index is the local index which we merge into the global index
        index_fields = ["language", "publication_date"]
        with open(loc, encoding="utf-8") as fd:
            for line_id, line in enumerate(fd):
                json_obj = json.loads(line)
                if "meta" not in json_obj:
                    continue

                meta = json_obj["meta"]
                del json_obj

                for index_field in index_fields:  # pylint: disable=consider-using-dict-items
                    if index_field not in meta:
                        continue
                    value = meta[index_field]

                    if index_field == "language":
                        # This is especially critical to generalize
                        for lang in value:
                            lang_name = lang["name"]
                            index[index_field][lang_name][dataset_id][file_id].append(line_id)
                    else:
                        # TODO(#11): Support numerical buckets, not just categorical
                        # logger.info(f"for index {index_field} the value is {value}")
                        index[index_field][lang_name][dataset_id][file_id].append(line_id)

        # Rangi-fy the list for each index, i.e., we go from [1,2,3,5,6,7] to [(1,4), (5,8)] to compress
        for index_field, buckets in index.items():
            for _, bucket_vals in buckets.items():
                bucket_vals[dataset_id][file_id] = ranges(bucket_vals[dataset_id][file_id])

        return index

    @staticmethod
    def read_file(ranges_per_file: list[str]) -> Iterable[str]:
        raise NotImplementedError()

    @staticmethod
    def read_ranges_from_files(ranges_per_file: dict[str, list[tuple[int, int]]]) -> Iterable[str]:
        raise NotImplementedError()

    @staticmethod
    def _is_valid_jsonl(path: Path) -> bool:
        try:
            with open(path, "r", encoding="utf-8") as file:
                for line_number, line in enumerate(file, start=1):
                    try:
                        json.loads(line)
                    except json.JSONDecodeError as e:
                        logger.error(f"Invalid JSON at line {line_number}: {e}")
                        return False
            return True
        except IOError as e:
            logger.error(f"IO error: {e}")
            return False
