"""
Tests for utilities module
"""

import net.utilities


def test_get_target_size():

    image_size = 100, 200
    size_factor = 32

    expected = 96, 192
    actual = net.utilities.get_target_image_size(image_size, size_factor)

    assert expected == actual
