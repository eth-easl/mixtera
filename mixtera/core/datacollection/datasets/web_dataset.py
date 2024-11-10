import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable, Optional

import wids
from loguru import logger
from mixtera.core.datacollection.datasets import Dataset, DatasetType
from mixtera.core.datacollection.index.parser import MetadataParser
from mixtera.core.filesystem import FileSystem
from mixtera.network.connection import ServerConnection


class WebDataset(Dataset):
    type: DatasetType = DatasetType.WEB_DATASET

    @staticmethod
    def iterate_files(loc: str) -> Iterable[str]:
        if not FileSystem.is_dir(loc):
            if not WebDataset._is_valid_webdataset(loc):
                raise RuntimeError(
                    f"Path {loc} does not belong to a directory and does not refer to a valid WebDataset folder."
                )

            yield loc

        logger.info(f"Starting to iterate over samples in folder: {loc}")

        yield from FileSystem.get_all_files_with_ext(loc, "tar")

    @staticmethod
    def _get_wids_dataset_from_file(loc: Path | str) -> tuple[wids.ShardListDataset, int]:
        n_samples = WebDataset._count_unique_samples(str(loc))

        shard_info = {
            "url": str(loc),
            "nsamples": n_samples,
        }

        dataset = wids.ShardListDataset(
            shards=[shard_info],
            cache_size=int(1e9),  # TODO: Should be configurable
            lru_size=10,  # TODO: Should be configurable
        )

        return dataset, n_samples

    @staticmethod
    def inform_metadata_parser(loc: Path, metadata_parser: MetadataParser) -> None:
        # TODO: This should use our FileSystem class and properly support server connections as well
        # but it requires changes to the FileSystem abstraction

        dataset, n_samples = WebDataset._get_wids_dataset_from_file(loc)

        for idx in range(n_samples):
            metadata_parser.parse(line_number=idx, payload=dataset[idx])

    @staticmethod
    def read_ranges_from_files(
        ranges_per_file: dict[str, list[tuple[int, int]]],
        parsing_func: Callable[[str | dict], str],  # TODO: Will not necessarily take a string?
        server_connection: Optional[ServerConnection],
    ) -> Iterable[str | dict]:
        for file, range_list in ranges_per_file.items():
            yield from WebDataset._read_ranges_from_file(file, range_list, parsing_func, server_connection)

    @staticmethod
    def _read_ranges_from_file(  # pylint: disable=contextmanager-generator-missing-cleanup
        file: str,
        range_list: list[tuple[int, int]],
        parsing_func: Callable[[dict], str],
        server_connection: Optional[ServerConnection],  # pylint: disable=unused-argument
    ) -> Iterable[str]:
        # TODO: need to modify our FileSystem abstraction to use server_connection
        dataset, _ = WebDataset._get_wids_dataset_from_file(file)

        last_line_read = 0
        last_r_start = -1
        for r_start, r_end in range_list:
            if r_start < last_r_start:
                raise RuntimeError(f"Ranges not sorted by start ({last_r_start} vs {r_start})")

            if last_line_read > r_start:
                raise RuntimeError(f"Overlapping ranges: start at {r_start} but previous ended at {last_line_read}")

            last_r_start = r_start

            # Yielding samples in the current range
            yield from (parsing_func(dataset[line]) for line in range(r_start, r_end))

            last_line_read = r_end

    @staticmethod
    def _count_unique_samples(tar_path: str) -> int:
        """
        Count the number of unique samples in a .tar file based on prefixes.

        Parameters:
        ----------
        tar_path : str
            Path to the .tar file.

        Returns:
        -------
        int
            Number of unique samples.
        """
        try:
            with tarfile.open(tar_path, "r") as tar:
                file_names = [member.name for member in tar.getmembers() if member.isfile()]
        except tarfile.TarError as e:
            logger.error(f"Error opening tar file {tar_path}: {e}")
            return 0

        # Extract prefixes
        prefixes = [Path(name).stem for name in file_names]
        unique_prefixes = set(prefixes)
        return len(unique_prefixes)

    @staticmethod
    def _is_valid_webdataset_tar(tar_path: str) -> bool:
        """
        Validate a .tar file to ensure it follows WebDataset conventions by checking
        that files are consistently grouped by their prefixes.

        Parameters:
        ----------
        tar_path : str
            Path to the .tar file.

        Returns:
        -------
        bool
            True if the .tar file is valid, False otherwise.
        """
        try:
            with tarfile.open(tar_path, "r") as tar:
                file_members = [member for member in tar.getmembers() if member.isfile()]
                if not file_members:
                    logger.error(f"No files found in tar archive: {tar_path}")
                    return False

                # Group files by prefix
                sample_dict = defaultdict(set)
                for member in file_members:
                    filename = Path(member.name)
                    prefix = filename.stem
                    ext = filename.suffix.lower()
                    sample_dict[prefix].add(ext)

                if not sample_dict:
                    logger.error(f"No valid samples found in tar archive: {tar_path}")
                    return False

                # Optionally, you can add checks here for consistency within samples
                # For example, ensuring no sample has conflicting data
                # Since we're not enforcing specific extensions, we'll skip this

        except tarfile.TarError as e:
            logger.error(f"Error reading tar file {tar_path}: {e}")
            return False

        return True

    @staticmethod
    def _is_valid_webdataset(path: str) -> bool:
        """
        Validate all .tar files in a specified folder to ensure they follow WebDataset conventions.

        Parameters:
        ----------
        path : str
            Path to the folder containing .tar files.

        Returns:
        -------
        bool
            True if all .tar files are valid, False otherwise.
        """
        folder = Path(path)
        if not folder.is_dir():
            logger.error(f"The provided path is not a directory: {path}")
            return False

        tar_files = list(folder.glob("*.tar"))
        if not tar_files:
            logger.error(f"No .tar files found in the directory: {path}")
            return False

        all_valid = True
        for tar_file in tar_files:
            logger.info(f"Validating tar file: {tar_file}")
            if not WebDataset._is_valid_webdataset_tar(str(tar_file)):
                logger.error(f"Validation failed for tar file: {tar_file}")
                all_valid = False
            else:
                num_samples = WebDataset._count_unique_samples(str(tar_file))
                logger.info(f"Tar file '{tar_file}' is valid with {num_samples} samples.")

        if all_valid:
            logger.info("All .tar files are valid WebDataset files.")
        else:
            logger.warning("Some .tar files have invalid structures.")

        return all_valid
