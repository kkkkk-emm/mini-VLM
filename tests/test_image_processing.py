import unittest

import torch
from PIL import Image

from data.custom_transforms import GlobalAndSplitImages
from data.processors import get_image_processor


class ImageProcessingTests(unittest.TestCase):
    def test_global_patch_keeps_batch_dimension_when_local_patches_exist(self):
        chunks, grid = GlobalAndSplitImages(2)(torch.ones(3, 2, 4))

        self.assertEqual(grid, (1, 2))
        self.assertEqual(chunks.shape, (3, 3, 2, 2))

    def test_image_processor_normalizes_pixels_like_siglip(self):
        processor = get_image_processor(
            max_image_size=2,
            splitted_image_size=2,
            resize_to_max_side_len=True,
        )

        chunks, grid = processor(Image.new("RGB", (2, 2), color="black"))

        self.assertEqual(grid, (1, 1))
        self.assertTrue(torch.allclose(chunks, torch.full_like(chunks, -1.0)))


if __name__ == "__main__":
    unittest.main()
