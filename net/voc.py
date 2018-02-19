"""
Code with VOC-specific functionality
"""

import glob
import os
import random

import cv2
import sklearn.utils
import numpy as np


class BatchesGeneratorFactory:

    def __init__(self, data_directory):

        self.data_directory = data_directory

    def get_generator(self, batch_size):

        data_map = self._get_data_map(self.data_directory)
        keys = list(data_map.keys())

        images = []
        segmentations = []

        while True:

            random.shuffle(keys)

            for key in keys:

                image_path, segmentation_path = data_map[key]

                image = cv2.imread(image_path)
                segmentation = cv2.imread(segmentation_path)

                images.append(image)
                segmentations.append(segmentation)

                if len(images) == batch_size:

                    yield sklearn.utils.shuffle(np.array(images), np.array(segmentations))

                    images.clear()
                    segmentations.clear()

    def _get_data_map(self, data_directory):

        images_paths = glob.glob(os.path.join(data_directory, "JPEGImages/**.jpg"))
        segmentation_paths = glob.glob(os.path.join(data_directory, "SegmentationClass/**.png"))

        file_name_to_image_path_map = {}

        # Get a dictionary mapping file names to full images pathes
        for image_path in images_paths:

            file_name_with_extension = os.path.basename(image_path)
            file_name = os.path.splitext(file_name_with_extension)[0]
            file_name_to_image_path_map[file_name] = image_path

        data_map = {}

        # Now prepare a dictionary mapping segmentation paths to corresponding images paths
        for segmentation_path in segmentation_paths:

            file_name_with_extension = os.path.basename(segmentation_path)
            file_name = os.path.splitext(file_name_with_extension)[0]
            image_path = file_name_to_image_path_map[file_name]

            data_map[file_name] = (image_path, segmentation_path)

        return data_map


def get_colors_info(categories_count):
    """
    Returns two dictionaries and a 3-element tuple.
     First dictionary maps VOC categories indices to colors in VOC segmentation images
     Second dictionary maps segmentation colors to categories
     3-element tuple represents color of void - that is ambiguous regions.
     All colors are returned in BGR order.
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

        # Writing colors in BGR order, since our image reading and logging routines use it
        colors_matrix[color_index] = b, g, r

    indices_to_colors_map = {color_index: tuple(colors_matrix[color_index]) for color_index in range(categories_count)}
    colors_to_indices_map = {color: index for index, color in indices_to_colors_map.items()}

    return indices_to_colors_map, colors_to_indices_map, tuple(colors_matrix[-1])


def get_void_mask(segmentation_image, void_color):
    """
    Compute a 2D void segmentation given segmentation image and void_color
    :param segmentation_image: numpy array, 3-channel segmentation image
    :param void_color: 3-element tuple representing void color
    :return: 2D binary array with 1 at void color location and 0 elsewhere
    """

    return np.all(segmentation_image == void_color, axis=-1).astype(np.int32)


# def get_segmentation_cube(segmentation_image, colors_to_indices_map):
#     """
#     Turns 2D 3-channel segmentation image with into a batch of 2D binary maps - one for each
#     segmentation category
#     :param segmentation_image: 2D 3-channnel segmentation image
#     :param colors_to_indices_map: map mapping image colors to categories indices
#     :return: 3D array with a binary 2D map for each category at a corresponding index
#     """
#
#     return None