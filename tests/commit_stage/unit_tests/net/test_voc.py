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


def test_get_segmentation_cube():

    segmentation_image = np.zeros((10, 10, 3), dtype=np.int32)

    indices_to_colors_map = {
        0: (10, 20, 30),
        1: (20, 30, 40),
        2: (30, 40, 50),
        3: (40, 50, 60)
    }

    segmentation_image[1, 1] = indices_to_colors_map[0]
    segmentation_image[3:5, 7:] = indices_to_colors_map[1]
    segmentation_image[7, 1:4] = indices_to_colors_map[3]

    expected = np.zeros(shape=(10, 10, 4), dtype=np.int32)
    expected[1, 1, 0] = 1
    expected[3:5, 7:, 1] = 1
    expected[7, 1:4, 3] = 1

    actual = net.voc.get_segmentation_cube(segmentation_image, indices_to_colors_map)

    assert np.all(expected == actual)
