import gzip
import io
from functools import partial
from typing import Any, Generic, Iterator, Type, TypeVar


def create_mock_dataset() -> Type[Any]:
    """Create a mock Dataset class that supports generic subscripting"""
    T = TypeVar("T")

    class MockDataset(Generic[T]):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __len__(self) -> int:
            return 0

        def __getitem__(self, idx: Any) -> None:
            return None

    return MockDataset


# wids depends on torch in its imports, but we don't rely on torch-related functionality
try:
    from wids.wids import group_by_key, splitname
    from wids.wids_mmtar import MMIndexedTar
except ImportError as e:
    TORCH_AVAILABLE = True
    try:
        import torch  # noqa: F401 # pylint: disable=unused-import
    except ImportError:
        TORCH_AVAILABLE = False

    if not TORCH_AVAILABLE:
        # Mock torch and retry
        import sys
        from unittest.mock import MagicMock

        torch_mock = MagicMock()
        torch_distributed_mock = MagicMock()
        torch_utils_data_mock = MagicMock()
        torch_utils_data_mock.Dataset = create_mock_dataset()

        sys.modules["torch"] = torch_mock
        sys.modules["torch.distributed"] = torch_distributed_mock
        sys.modules["torch.utils.data"] = torch_utils_data_mock

        from wids.wids import group_by_key, splitname
        from wids.wids_mmtar import MMIndexedTar

        del sys.modules["torch"]
        del sys.modules["torch.distributed"]
        del sys.modules["torch.utils.data"]
    else:
        raise ImportError("Non-torch related error in wids") from e


def decode(sample: dict[str, Any], decode_image: bool = True) -> dict[str, Any]:
    """
    A utility function to decode the samples from the tar file for many common extensions.
    """
    sample = dict(sample)
    for key, stream in sample.items():
        extensions = key.split(".")
        if len(extensions) < 1:
            continue
        extension = extensions[-1]
        if extension in ["gz"]:
            decompressed = gzip.decompress(stream.read())
            stream = io.BytesIO(decompressed)
            if len(extensions) < 2:
                sample[key] = stream
                continue
            extension = extensions[-2]
        if key.startswith("__"):
            continue
        if extension in ["txt", "text"]:
            value = stream.read()
            sample[key] = value.decode("utf-8")
        elif extension in ["cls", "cls2"]:
            value = stream.read()
            sample[key] = int(value.decode("utf-8"))
        elif extension in ["jpg", "png", "ppm", "pgm", "pbm", "pnm"] and decode_image:
            import torchvision.transforms.functional as F  # pylint: disable=import-outside-toplevel
            from PIL import Image  # pylint: disable=import-outside-toplevel

            image = Image.open(stream)
            sample[key] = F.to_tensor(image)
        elif extension == "json":
            import json  # pylint: disable=import-outside-toplevel

            value = stream.read()
            sample[key] = json.loads(value)
        elif extension == "npy":
            import numpy as np  # pylint: disable=import-outside-toplevel

            sample[key] = np.load(stream)
        elif extension in ["pickle", "pkl"]:
            import pickle  # pylint: disable=import-outside-toplevel

            sample[key] = pickle.load(stream)
    return sample


class IndexedTarSamples:
    def __init__(self, path: str, decode_images: bool = True):
        """
        A class for efficient reading of tar files for web datasets.

        This class uses the `wids` library's `MMIndexedTar` to read tar files.
        It's a simplified version of the `wids` library's `IndexedTarSamples` without support for streams
        and with decoding integrated.
        """
        self.path = path
        self.decoder = partial(decode, decode_image=decode_images)
        self.stream = open(self.path, "rb")  # pylint: disable=consider-using-with
        self.reader = MMIndexedTar(self.stream)

        all_files = self.reader.names()
        self.samples = group_by_key(all_files)

    def __enter__(self) -> "IndexedTarSamples":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:  # type: ignore
        self.close()

    def close(self) -> None:
        if self.reader is not None:
            self.reader.close()
        if self.stream is not None and not self.stream.closed:
            self.stream.close()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self.samples is not None and self.reader is not None:
            indexes = self.samples[idx]
            sample = {}
            key = None
            for i in indexes:
                fname, data = self.reader.get_file(i)
                k, ext = splitname(fname)
                key = key or k
                assert key == k, "Inconsistent keys in the same sample"
                sample[ext] = data
            sample["__key__"] = key
            return self.decoder(sample)
        raise ValueError("Co")

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for idx in range(len(self)):
            yield self[idx]
