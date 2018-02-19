"""
Tests for net.voc module
"""

import numpy as np

import net.voc


def test_get_void_mask():

    segmentation_image = np.zeros((10, 10, 3), dtype=np.int32)

    void_color = (50, 100, 150)

    segmentation_image[:4, :4, :] = (10, 20, 30)
    segmentation_image[8:, 7:, :] = void_color

    expected = np.zeros((10, 10), dtype=np.int32)
    expected[8:, 7:] = 1

    actual = net.voc.get_void_mask(segmentation_image, void_color)

    assert np.all(expected == actual)
