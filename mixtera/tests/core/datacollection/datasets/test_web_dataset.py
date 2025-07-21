import io
import tarfile
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from mixtera.core.datacollection.datasets import WebDataset  # Ensure this is the correct import path


class TestWebDataset(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        self.directory = Path(self.temp_dir.name)

        # Create multiple .tar files with fake data
        create_fake_webdataset_tar(self.directory / "dataset_part1.tar", num_samples=3, start_id=1)
        create_fake_webdataset_tar(self.directory / "dataset_part2.tar", num_samples=2, start_id=4)
        create_fake_webdataset_tar(self.directory / "dataset_part3.tar", num_samples=3, start_id=6)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_iterate_tar_files_directory(self):
        tar_files = sorted([str(self.directory / f"dataset_part{i}.tar") for i in range(1, 4)])
        iterated_files = sorted(list(WebDataset.iterate_files(str(self.directory))))
        self.assertListEqual(iterated_files, tar_files)

    def test_build_file_index(self):
        pass  # Implement when WebDataset's indexing is defined

    def test_read_ranges_from_tar_e2e(self):
        """
        End-to-end test for reading specific ranges from multiple .tar files.
        """
        # Define the ranges for each tar file
        ranges_per_file = {
            str(self.directory / "dataset_part1.tar"): [(0, 3)],  # Read all three samples
            str(self.directory / "dataset_part2.tar"): [(0, 2)],  # Read both samples
            str(self.directory / "dataset_part3.tar"): [(1, 3)],  # Read last two samples
        }

        expected = [
            {"__key__": "000001", ".cls": 1},
            {"__key__": "000002", ".cls": 0},
            {"__key__": "000003", ".cls": 1},
            {"__key__": "000004", ".cls": 0},
            {"__key__": "000005", ".cls": 1},
            {"__key__": "000007", ".cls": 1},
            {"__key__": "000008", ".cls": 0},
        ]

        result = list(WebDataset.read_ranges_from_files(ranges_per_file, lambda x: x, None))
        assert all(".png" in sample for sample in result)

        result = [{k: v for k, v in sample.items() if k in [".cls", "__key__"]} for sample in result]
        self.assertEqual(result, expected)


def create_dummy_image(image_size=(64, 64), color=(255, 0, 0)) -> bytes:
    """
    Creates a dummy image and returns its bytes.
    """
    img = Image.new("RGB", image_size, color)
    with io.BytesIO() as img_bytes:
        img.save(img_bytes, format="PNG")
        return img_bytes.getvalue()


def create_fake_webdataset_tar(tar_path: Path, num_samples: int, start_id: int = 0) -> None:
    with tarfile.open(tar_path, "w") as tar:
        for i in range(start_id, start_id + num_samples):
            sample_id = f"{i:06d}"

            image_data = create_dummy_image(color=(i % 256, (i * 2) % 256, (i * 3) % 256))
            image_name = f"{sample_id}.png"
            image_info = tarfile.TarInfo(name=image_name)
            image_info.size = len(image_data)
            tar.addfile(image_info, fileobj=io.BytesIO(image_data))

            class_label = "0" if i % 2 == 0 else "1"
            class_content = class_label.encode("utf-8")
            class_name = f"{sample_id}.cls"
            class_info = tarfile.TarInfo(name=class_name)
            class_info.size = len(class_content)
            tar.addfile(class_info, fileobj=io.BytesIO(class_content))
