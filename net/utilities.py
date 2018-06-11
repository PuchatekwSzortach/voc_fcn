"""
Module with various utilities
"""

import os
import logging
import random

import numpy as np
import cv2


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
    """
    Turns a list of values into a dictionary {value index: value}
    :param values: list
    :return: dictionary
    """

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


def get_bilinear_kernel(height, width, channels):
    """
    GEt a bilinear kernel for FCN upscaling/deconvolution.
    It has a peak at center and drops off towards borders. Filters are the same across channels.
    :param height: filter height
    :param width: filter width
    :param channels: number of channels
    :return: 3D numpy array
    """

    if height % 2 != 0 or width % 2 != 0:
        raise ValueError("Odd height and width are not supported")

    height_array = np.zeros(height, dtype=np.float32)

    half_height_range = range(1, (height // 2) + 1)
    height_array[:height // 2] = half_height_range
    height_array[height // 2:] = list(reversed(half_height_range))

    width_array = np.zeros(height, dtype=np.float32)

    half_width_range = range(1, (width // 2) + 1)
    width_array[:width // 2] = half_width_range
    width_array[width // 2:] = list(reversed(half_width_range))

    unscaled_filter = np.dot(height_array.reshape(-1, 1), width_array.reshape(1, -1))
    scaled_filter = unscaled_filter / np.sum(unscaled_filter)

    return np.repeat(scaled_filter.reshape(height, width, 1), repeats=channels, axis=2)


def bilinear_initializer(shape, dtype, partition_info):
    """
    Bilinear initializer for deconvolution filters
    """

    kernel = get_bilinear_kernel(shape[0], shape[1], shape[2])

    broadcasted_kernel = np.repeat(kernel.reshape(shape[0], shape[1], shape[2], -1), repeats=shape[3], axis=3)
    return broadcasted_kernel


def get_categories_segmentations_maps(segmentation_cube, ids_to_categories_map):
    """
    Get categories names to their segmentation maps dictionary for all categories wit non-empty segmentations
    in segmentation cube
    :param segmentation_cube: 3D numpy array of segmentation masks
    :param ids_to_categories_map: map of ids to categories name
    :return: map of categories to segmentation maps
    """

    categories_segmentations_map = {}

    for index, category in ids_to_categories_map.items():

        segmentation = segmentation_cube[:, :, index]

        if np.any(segmentation):
            categories_segmentations_map[category] = segmentation

    return categories_segmentations_map


class DataAugmenter:
    """
    Simple class for data augmentation
    """

    @staticmethod
    def augment_samples(image, segmentation):
        """
        Performs random augmentations on copies of inputs and returns them
        :param image: numpy array
        :param segmentation: numpy array
        :return: tuple (augmented image, augmented segmentation)
        """

        # Random flip around horizontal axis
        if random.randint(0, 1) == 1:
            image = cv2.flip(image, flipCode=0)
            segmentation = cv2.flip(segmentation, flipCode=0)

        # Random flip around vertical axis
        if random.randint(0, 1) == 1:
            image = cv2.flip(image, flipCode=1)
            segmentation = cv2.flip(segmentation, flipCode=1)

        return image, segmentation

