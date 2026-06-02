import unittest

import torch

from data.custom_transforms import GlobalAndSplitImages


class ImageProcessingTests(unittest.TestCase):
    def test_global_patch_keeps_batch_dimension_when_local_patches_exist(self):
        chunks, grid = GlobalAndSplitImages(2)(torch.ones(3, 2, 4))

        self.assertEqual(grid, (1, 2))
        self.assertEqual(chunks.shape, (3, 3, 2, 2))


if __name__ == "__main__":
    unittest.main()
