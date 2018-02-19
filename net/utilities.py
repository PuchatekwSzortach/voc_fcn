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
