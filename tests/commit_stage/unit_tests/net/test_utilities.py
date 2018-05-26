"""
Tests for utilities module
"""

import numpy as np

import net.utilities


def test_get_target_size():
    """
    Test get_target_image_size for a sample input
    """

    image_size = 100, 200
    size_factor = 32

    expected = 96, 192
    actual = net.utilities.get_target_image_size(image_size, size_factor)

    assert expected == actual


def test_get_categories_segmentations_maps():
    """
    Test that correct categories name to segmentation maps dictionary is computed
    """

    segmentation_cube = np.zeros(shape=(10, 10, 4))
    segmentation_cube[0, 0, 0] = 1
    segmentation_cube[3, 3, 3] = 1

    ids_to_categories_map = {
        0: "zero",
        1: "one",
        2: "two",
        3: "three"
    }

    expected = {
        "zero": segmentation_cube[:, :, 0],
        "three": segmentation_cube[:, :, 3]
    }

    actual = net.utilities.get_categories_segmentations_maps(segmentation_cube, ids_to_categories_map)

    assert expected.keys() == actual.keys()

    for expected_key, actual_key in zip(sorted(expected.keys()), sorted(actual.keys())):

        assert np.all(expected[expected_key] == actual[actual_key])
