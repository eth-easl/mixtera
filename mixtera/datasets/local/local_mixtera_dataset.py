from pathlib import Path

from mixtera.datasets import DatasetTypes, MixteraDataset


class LocalMixteraDataset(MixteraDataset):

    def __init__(self, directory: Path) -> None:
        if not directory.exists():
            raise RuntimeError(f"Directory {directory} does not exist.")

        self._directory = directory
        self._database_path = self._directory / "mixtera.sqlite"

        if not self._database_path.exists():
            self._init_database()
            assert self._database_path.exists()

    def _init_database(self) -> None:
        assert hasattr(self, "_database_path")
        assert not self._database_path.exists()
        raise NotImplementedError("Not yet implemented")

    def register_dataset(self, identifier: str, loc: str, dtype: DatasetTypes) -> bool:
        raise NotImplementedError("Not yet implemented")

    def check_dataset_exists(self, identifier: str) -> bool:
        raise NotImplementedError("Not yet implemented")
