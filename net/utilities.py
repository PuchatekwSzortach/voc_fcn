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


def get_colors_map():
    """
    Colormap used to denote VOC segmentations.
    Code adapted from https://gist.github.com/wllhf/a4533e0adebe57e3ed06d4b50c8419ae
    :return: colors array
    """

    colors_count = 256

    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    colors_map = np.zeros((colors_count, 3), dtype=np.uint8)

    for color_index in range(colors_count):

        r = g = b = 0
        c = color_index

        for j in range(8):

            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        colors_map[color_index] = np.array([r, g, b])

    return colors_map
