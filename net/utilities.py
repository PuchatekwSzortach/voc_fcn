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


def get_image_colors(image):
    """
    Given an image, returns a list of colors found in the image
    :param image: 3D numpy array
    :return: list of 3 element tuples
    """

    image_colors = image.reshape(-1, 3)
    unique_colors_array = unique_colors_array = np.unique(image_colors, axis=0)
    return set(tuple(color) for color in unique_colors_array)


def get_ids_to_values_map(values):

    return {id: category for id, category in enumerate(values)}


def get_target_image_size(image_size, size_factor):
    """
    Given an image_size tuple and size_factor, return a new image_size tuple that is a multiple of size_factor and
    as close to original image_size as possible
    :param image_size: tuple of two integers
    :param size_factor: integer
    :return: tuple of two integers
    """

    target_sizes = []

    for size in image_size:

        target_size = size_factor * (size // size_factor)
        target_sizes.append(target_size)

    return tuple(target_sizes)
