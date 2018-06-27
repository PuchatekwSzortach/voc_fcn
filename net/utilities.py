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


def get_uint8_images(images):
    """
    Converts a list of float images to uint8, making sure to scale up their brighness.
    :param images: list of numpy arrays
    :return: list of numpy arrays
    """

    brightness_adjusted_images = [255 * image for image in images]
    return [image.astype(np.uint8) for image in brightness_adjusted_images]


class DataAugmenter:
    """
    Simple class for data augmentation
    """

    @staticmethod
    def augment_samples(image, segmentation, void_color):
        """
        Performs random augmentations on copies of inputs and returns them
        :param image: numpy array
        :param segmentation: numpy array
        :param void_color: three elements tuple, specifies color of pixels without a category
        :return: tuple (augmented image, augmented segmentation)
        """

        image, segmentation = DataAugmenter.flip_samples(image, segmentation)
        image = DataAugmenter.change_brightness(image)
        image, segmentation = DataAugmenter.rotate_samples(image, segmentation, void_color)

        return image, segmentation

    @staticmethod
    def flip_samples(image, segmentation):
        """
        Randomly flips samples around vertical axis
        :param image: numpy array
        :param segmentation: numpy array
        :return: tuple (augmented image, augmented segmentation)
        """

        # Random flip around vertical axis
        if random.randint(0, 1) == 1:

            image = cv2.flip(image, flipCode=1)
            segmentation = cv2.flip(segmentation, flipCode=1)

        return image, segmentation

    @staticmethod
    def rotate_samples(image, segmentation, void_color):
        """
        Performs random rotation around a random point near image center
        :param image: numpy array
        :param segmentation: numpy array
        :param void_color: 3-elements tuple of integers, color with which blank areas of rotated segmentation map
        should be filled in
        :return: tuple (rotated image, rotated segmentation)
        """

        width = image.shape[1]
        height = image.shape[0]

        x = random.randint(int(0.25 * width), int(0.75 * width))
        y = random.randint(int(0.25 * height), int(0.75 * height))

        # angle = random.randint(-15, 15)
        angle = random.randint(-20, 20)

        rotation_matrix = cv2.getRotationMatrix2D((x, y), angle, scale=1)
        augmented_image = cv2.warpAffine(image, rotation_matrix, (width, height), flags=cv2.INTER_NEAREST)

        # We need to convert channel values from np.int64 to int to make OpenCV happy
        void_color_tuple = [int(channel_value) for channel_value in void_color]

        augmented_segmentation = cv2.warpAffine(
            segmentation.astype(np.uint8), rotation_matrix, (width, height),
            flags=cv2.INTER_NEAREST, borderValue=void_color_tuple)

        return augmented_image, augmented_segmentation

    @staticmethod
    def change_brightness(image):
        """
        Randomly changes image
        :param image: numpy array
        :return: numpy array
        """

        augmented_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)

        augmented_image = augmented_image.astype(np.float32)

        hue_shift = np.random.randint(-10, 10)
        saturation_shift = np.random.randint(-30, 30)
        value_shift = np.random.randint(-30, 30)

        random_shift = [hue_shift, saturation_shift, value_shift]

        augmented_image = augmented_image + random_shift

        augmented_image = np.clip(augmented_image, 0, 255).astype(np.uint8)
        return cv2.cvtColor(augmented_image, cv2.COLOR_HSV2BGR_FULL)


def get_intersection_over_union(first_segmentation, second_segmentation):
    """
    Computes intersection over union (IoU) between two segmentation maps.
    Maps are binary - that is their values are 0s and 1s only.
    IoU is computed between non-zero elements of both segmentation maps.
    :param first_segmentation: 2D numpy array
    :param second_segmentation: 2D numpy array
    :return: float
    """

    union = np.logical_or(first_segmentation, second_segmentation)
    intersection = np.logical_and(first_segmentation, second_segmentation)

    return np.sum(intersection) / np.sum(union)


def get_segmentation_overlaid_image(image, segmentation, colors_to_ignore):
    """
    Return an image with segmentation mask overlaid over it.
    :param image: numpy array
    :param segmentation: numpy array
    :param colors_to_ignore: list of 3-elements tuples - segmentation colors that should be ignored when computing
    overlaid image
    :return: numpy array
    """

    segmentation_colors = get_image_colors(segmentation)

    overlay_colors = segmentation_colors.difference(colors_to_ignore)

    overlay_image = image.copy()
    blended_image = cv2.addWeighted(image, 0.5, segmentation, 0.5, 0)

    for color in overlay_colors:

        mask = np.all(segmentation == color, axis=2)
        overlay_image[mask] = blended_image[mask]

    return overlay_image


def get_segmentation_labels_image(segmentation_image, indices_to_colors_map):
    """
    Creates a segmentation labels image that translates segmentation color to index value.
    For each pixel without a reference color provided in indices_to_colors_map value 0 is used.
    :param segmentation_image: 3D array of segmentations
    :param indices_to_colors_map: dictionary mapping segmentation categories to colors
    :return: 2D numpy array with pixel values set to corresponding indices of segmentations categories.
    Pixels with no category assigned have value 0.
    """

    segmentation_labels_image = np.zeros(segmentation_image.shape[:2])

    for index, color in indices_to_colors_map.items():

        color_pixels = np.all(segmentation_image == color, axis=2)
        segmentation_labels_image[color_pixels] = index

    return segmentation_labels_image
