"""
Module with various utilities
"""

import os
import logging

import numpy as np


def get_logger(path):
    """
    Returns a logger that writes to an html page
    :param path: path to log.html page
    :return: logger instance
    """

    os.makedirs(os.path.dirname(path), exist_ok=True)

    logger = logging.getLogger("voc_fcn")
    file_handler = logging.FileHandler(path, mode="w")

    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    return logger


def get_colors_info(categories_count):
    """
    Returns two dictionaries and a 3-element tuple.
     First dictionary maps VOC categories indices to colors in VOC segmentation images
     Second dictionary maps segmentation colors to categories
     3-element tuple represents color of void - that is ambiguous regions
    Code adapted from https://gist.github.com/wllhf/a4533e0adebe57e3ed06d4b50c8419ae
    :param categories_count: number of categories - includes background, but doesn't include void
    :return: map, map, tuple
    """

    colors_count = 256

    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    colors_matrix = np.zeros(shape=(colors_count, 3), dtype=np.int)

    for color_index in range(colors_count):

        r = g = b = 0
        c = color_index

        for j in range(8):

            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        colors_matrix[color_index] = r, g, b

    indices_to_colors_map = {color_index: tuple(colors_matrix[color_index]) for color_index in range(categories_count)}
    colors_to_indices_map = {color: index for index, color in indices_to_colors_map.items()}

    return indices_to_colors_map, colors_to_indices_map, tuple(colors_matrix[-1])


def get_image_colors(image):
    """
    Given an image, returns a list of colors found in the image
    :param image: 3D numpy array
    :return: list of 3 element tuples
    """

    colors_set = set()

    for y in range(image.shape[0]):

        for x in range(image.shape[1]):

            color = image[y][x]
            colors_set.add(tuple(color))

    return list(colors_set)
