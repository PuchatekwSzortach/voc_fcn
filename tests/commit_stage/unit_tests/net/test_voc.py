"""
Tests for net.voc module
"""

import numpy as np

import net.data


def test_get_void_mask():
    """
    Test function computing void mask (pixels for which segmentation is undefined)
    """

    segmentation_image = np.zeros((10, 10, 3), dtype=np.int32)

    void_color = (50, 100, 150)

    segmentation_image[:4, :4, :] = (10, 20, 30)
    segmentation_image[8:, 7:, :] = void_color

    expected = np.zeros((10, 10), dtype=np.int32)
    expected[8:, 7:] = 1

    actual = net.data.get_void_mask(segmentation_image, void_color)

    assert np.all(expected == actual)


def test_get_segmentation_cube():
    """
    Test function splitting segmentations into 2D map for each segmentation - thus giving a 3D segmentation volume
    :return:
    """

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

    actual = net.data.get_segmentation_cube(segmentation_image, indices_to_colors_map)

    assert np.all(expected == actual)


def test_get_segmentation_image_binary_segmentation_cube_values():
    """
    Test function turning a segmentation cube into a single segmentation image.
    All segmentation cube values are 0s or 1s only
    """

    segmentation_cube = np.zeros(shape=(10, 10, 4), dtype=np.float32)

    segmentation_cube[0:2, 0:4, 0] = 1
    segmentation_cube[0:2, 4:6, 1] = 1
    segmentation_cube[4:8, 2:6, 2] = 1
    segmentation_cube[4:8, 8:10, 3] = 1

    indices_to_colors_map = {
        0: (50, 50, 50),
        1: (100, 100, 100),
        2: (150, 150, 150),
        3: (200, 200, 200),
    }

    void_color = (10, 10, 10)

    expected = np.zeros(shape=(10, 10, 3), dtype=np.uint8)
    expected[:, :] = void_color

    expected[0:2, 0:4] = indices_to_colors_map[0]
    expected[0:2, 4:6] = indices_to_colors_map[1]
    expected[4:8, 2:6] = indices_to_colors_map[2]
    expected[4:8, 8:10] = indices_to_colors_map[3]

    actual = net.data.get_segmentation_image(segmentation_cube, indices_to_colors_map, void_color)

    assert np.all(expected == actual)


def test_get_segmentation_image_float_segmentation_cube_values():
    """
    Test function turning a segmentation cube into a single segmentation image.
    Segmentation values are floats in <0, 1> range
    """

    segmentation_cube = np.zeros(shape=(10, 10, 4), dtype=np.float32)

    # Channel 0 has largest value
    segmentation_cube[1:3, 0:4, 0] = 0.6
    segmentation_cube[1:3, 0:4, 1] = 0.2
    segmentation_cube[1:3, 0:4, 2] = 0.1
    segmentation_cube[1:3, 0:4, 3] = 0.1

    # Channel 1 has largest value
    segmentation_cube[0:4, 5:9, 0] = 0.2
    segmentation_cube[0:4, 5:9, 1] = 0.8
    segmentation_cube[0:4, 5:9, 2] = 0.0
    segmentation_cube[0:4, 5:9, 3] = 0.0

    # Channel 2 has largest value
    segmentation_cube[5:7, 2:4, 0] = 0.1
    segmentation_cube[5:7, 2:4, 1] = 0.3
    segmentation_cube[5:7, 2:4, 2] = 0.5
    segmentation_cube[5:7, 2:4, 3] = 0.1

    # Channel 3 has largest value
    segmentation_cube[8:10, 7:9, 0] = 0.1
    segmentation_cube[8:10, 7:9, 1] = 0.0
    segmentation_cube[8:10, 7:9, 2] = 0.0
    segmentation_cube[8:10, 7:9, 3] = 0.9

    indices_to_colors_map = {
        0: (50, 50, 50),
        1: (100, 100, 100),
        2: (150, 150, 150),
        3: (200, 200, 200),
    }

    void_color = (10, 10, 10)

    expected = np.zeros(shape=(10, 10, 3), dtype=np.uint8)
    expected[:, :] = void_color

    expected[1:3, 0:4] = indices_to_colors_map[0]
    expected[0:4, 5:9] = indices_to_colors_map[1]
    expected[5:7, 2:4] = indices_to_colors_map[2]
    expected[8:10, 7:9] = indices_to_colors_map[3]

    actual = net.data.get_segmentation_image(segmentation_cube, indices_to_colors_map, void_color)

    assert np.all(expected == actual)
