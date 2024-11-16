import gzip
import io
from functools import partial
from typing import Dict, Any

from wids.wids import group_by_key, splitname
from wids.wids_mmtar import MMIndexedTar


def decode(sample: Dict[str, Any], decode_image: bool = True):
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
        elif extension in ["txt", "text"]:
            value = stream.read()
            sample[key] = value.decode("utf-8")
        elif extension in ["cls", "cls2"]:
            value = stream.read()
            sample[key] = int(value.decode("utf-8"))
        elif extension in ["jpg", "png", "ppm", "pgm", "pbm", "pnm"] and decode_image:
            import PIL.Image

            sample[key] = PIL.Image.open(stream)
        elif extension == "json":
            import json

            value = stream.read()
            sample[key] = json.loads(value)
        elif extension == "npy":
            import numpy as np

            sample[key] = np.load(stream)
        elif extension in ["pt", "pth"]:
            import torch

            sample[key] = torch.load(stream)
        elif extension in ["pickle", "pkl"]:
            import pickle

            sample[key] = pickle.load(stream)
    return sample


class IndexedTarSamples:
    def __init__(
        self,
        path: str,
        decode_images: bool = True,
    ):
        self.path = path
        self.stream = open(self.path, "rb")
        self.reader = MMIndexedTar(self.stream)

        self.decoder = partial(decode, decode_image=decode_images)

        all_files = self.reader.names()

        self.samples = group_by_key(all_files)

    def close(self):
        self.reader.close()
        if not self.stream.closed:
            self.stream.close()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
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

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]
