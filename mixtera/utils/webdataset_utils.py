import gzip
import io
from typing import Any, Iterator

from wids.wids import group_by_key, splitname
from wids.wids_mmtar import MMIndexedTar


def decode_sample(sample: dict[str, Any]) -> dict[str, Any]:
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
        elif extension in ["jpg", "png", "ppm", "pgm", "pbm", "pnm"]:
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


class MMIndexedTarRawBytes(MMIndexedTar):
    """
    A subclass of `MMIndexedTar` that returns the raw bytes instead of an IOBytes object.
    """

    def get_file(self, i):
        fname, data = self.get_at_index(i)
        return fname, data


class IndexedTarSamples:
    def __init__(self, path: str, decode: bool = False):
        """
        A class for efficient reading of tar files for web datasets.

        This class uses the `wids` library's `MMIndexedTar` to read tar files.
        It's a simplified version of the `wids` library's `IndexedTarSamples` without support for streams
        and with decoding integrated.
        """
        self.path = path
        self.decoder = decode_sample
        self.decode = decode
        self.reader = MMIndexedTarRawBytes(path)

        all_files = self.reader.names()
        self.samples = group_by_key(all_files)

    def __enter__(self) -> "IndexedTarSamples":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:  # type: ignore
        self.close()

    def close(self) -> None:
        if self.reader is not None:
            self.reader.close()

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
                sample[ext[1:]] = data
            sample["__key__"] = key
            return self.decoder(sample) if self.decode else sample
        raise ValueError("Error reading sample")

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for idx in range(len(self)):
            yield self[idx]
