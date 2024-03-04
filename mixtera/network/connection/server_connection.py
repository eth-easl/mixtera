from typing import Iterable


class ServerConnection:
    def __init__(self) -> None:
        raise NotImplementedError("To be implemented in a future PR.")

    def get_file_iterable(self, filesys_id: int, file_path: str) -> Iterable[str]:
        raise NotImplementedError("To be implemented in a future PR.")
