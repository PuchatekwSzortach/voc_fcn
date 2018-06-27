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


def test_get_intersection_over_union_both_inputs_identical():
    """
    Test iou is 1 when both inputs are the same
    """

    image = np.zeros(shape=(10, 10))
    image[:5, :5] = 1

    expected = 1
    actual = net.utilities.get_intersection_over_union(image, image)

    assert expected == actual


def test_get_intersection_over_union_no_overlap():
    """
    Test iou is 0 if there is no overlap between segmentations
    """

    first_image = np.zeros(shape=(10, 10))
    first_image[:5, :5] = 1

    second_image = np.zeros(shape=(10, 10))
    second_image[5:, 5:] = 1

    expected = 0
    actual = net.utilities.get_intersection_over_union(first_image, second_image)

    assert expected == actual


def test_get_intersection_over_union_partial_overlap():
    """
    Test a non-zero iou
    """

    first_image = np.zeros(shape=(10, 10))
    first_image[:7, :7] = 1

    second_image = np.zeros(shape=(10, 10))
    second_image[3:, 3:] = 1

    expected = 16 / 82
    actual = net.utilities.get_intersection_over_union(first_image, second_image)

    assert np.isclose(expected, actual)


def test_get_segmentation_labels_image():
    """
    Test for get_segmentation_labels_image that changes segmentation image into a 2D array with segmentation categories
    set for corresponding pixels values
    """

    void_color = [100, 100, 100]

    indices_to_colors_map = {
        1: [10, 20, 30],
        2: [20, 30, 40],
        3: [30, 40, 50],
    }

    segmentation_image = np.ones(shape=(10, 10, 3)) * void_color
    segmentation_image[:3, :3] = indices_to_colors_map[1]
    segmentation_image[:3, 7:] = indices_to_colors_map[2]
    segmentation_image[8:, 6:] = indices_to_colors_map[3]

    expected = np.zeros(shape=(10, 10))
    expected[:3, :3] = 1
    expected[:3, 7:] = 2
    expected[8:, 6:] = 3

    actual = net.utilities.get_segmentation_labels_image(segmentation_image, indices_to_colors_map)

    assert np.all(expected == actual)
