import unittest

import numpy as np

from skm_tea.utils.visualizer import draw_reconstructions


class TestDrawReconstructions(unittest.TestCase):
    def test_hw(self):
        images = np.random.randn(3, 3)
        target = np.random.randn(3, 3)
        kspace = np.random.randn(3, 3).astype(np.complex64)

        outputs = draw_reconstructions(images, target, kspace, padding=0)
        assert all(x in outputs for x in ["images", "errors", "masks"])
        assert outputs["images"].shape == (3, 3, 6)
        assert outputs["errors"].shape == (3, 3, 3)
        assert outputs["masks"].shape == (3, 3, 3)

    def test_hw2(self):
        images = np.random.randn(3, 3).astype(np.complex64)
        target = np.random.randn(3, 3).astype(np.complex64)
        kspace = np.random.randn(3, 3).astype(np.complex64)
        outputs = draw_reconstructions(images, target, kspace)
        assert all(x in outputs for x in ["images", "errors", "phases", "masks"]), outputs.keys()
        assert outputs["images"].shape == (3, 3, 6)
        assert outputs["errors"].shape == (3, 3, 3)
        assert outputs["phases"].shape == (3, 3, 6)
        assert outputs["masks"].shape == (3, 3, 3)

    def test_nhw(self):
        images = np.random.randn(3, 3)
        target = np.random.randn(3, 3)
        kspace = np.random.randn(3, 3).astype(np.complex64)

        outputs = draw_reconstructions(images, target, kspace)
        assert all(x in outputs for x in ["images", "errors", "masks"])
        assert outputs["images"].shape == (3, 3, 6)
        assert outputs["errors"].shape == (3, 3, 3)
        assert outputs["masks"].shape == (3, 3, 3)

    def test_nhw2(self):
        images = np.random.randn(4, 3, 3).astype(np.complex64)
        target = np.random.randn(4, 3, 3).astype(np.complex64)
        kspace = np.random.randn(4, 3, 3).astype(np.complex64)

        outputs = draw_reconstructions(images, target, kspace, padding=0)
        assert all(x in outputs for x in ["images", "errors", "masks"])
        assert outputs["images"].shape == (3, 12, 6)


if __name__ == "__main__":
    unittest.main()
