"""
Code with VOC-specific functionality
"""

import glob
import os
import random
import copy

import numpy as np
import cv2

import net.utilities


def get_images_paths_and_segmentations_paths_tuples(data_directory):
    """
    Returns a list of tuples, each tuple is an image path, segmentation path pair for a single image
    :param data_directory: VOC data directory
    :return: list of tuples
    """

    images_paths = glob.glob(os.path.join(data_directory, "JPEGImages/**.jpg"))
    segmentation_paths = glob.glob(os.path.join(data_directory, "SegmentationClass/**.png"))

    file_name_to_image_path_map = {}

    # Get a dictionary mapping file names to full images paths
    for image_path in images_paths:

        file_name_with_extension = os.path.basename(image_path)
        file_name = os.path.splitext(file_name_with_extension)[0]
        file_name_to_image_path_map[file_name] = image_path

    images_paths_and_segmentation_paths_tuples = []

    # Now prepare a dictionary mapping segmentation paths to corresponding images paths
    for segmentation_path in segmentation_paths:

        file_name_with_extension = os.path.basename(segmentation_path)
        file_name = os.path.splitext(file_name_with_extension)[0]
        image_path = file_name_to_image_path_map[file_name]

        images_paths_and_segmentation_paths_tuples.append((image_path, segmentation_path))

    return images_paths_and_segmentation_paths_tuples


class VOCSamplesGeneratorFactory:
    """
    Factory class creating data batches generators that yield (image, segmentation image) pairs
    """

    def __init__(self, data_directory):
        """
        Constructor
        :param data_directory: directory with VOC data
        """

        self.images_paths_and_segmentations_paths_tuples = \
            get_images_paths_and_segmentations_paths_tuples(data_directory)

    def get_generator(self, size_factor):
        """
        Returns generator that yields (image_path, segmentation_image) pair on each yield
        :param size_factor: int, value by which height and with of outputs must be divisible
        :return: generator
        """

        local_images_paths_and_segmentations_paths_tuples = \
            copy.deepcopy(self.images_paths_and_segmentations_paths_tuples)

        while True:

            random.shuffle(local_images_paths_and_segmentations_paths_tuples)

            for image_path, segmentation_path in local_images_paths_and_segmentations_paths_tuples:

                image = cv2.imread(image_path)
                segmentation = cv2.imread(segmentation_path)

                target_size = net.utilities.get_target_image_size(image.shape[:2], size_factor)
                target_size = target_size[1], target_size[0]

                image = cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC)
                segmentation = cv2.resize(segmentation, target_size, interpolation=cv2.INTER_NEAREST)

                yield image, segmentation

    def get_size(self):
        """
        Gets size of dataset served by the generator
        :return: int
        """
        return len(self.images_paths_and_segmentations_paths_tuples)


class VOCOneHotEncodedSamplesGeneratorFactory:
    """
    Factory class creating data batches generators that yield (image, segmentation cube) pairs
    """

    def __init__(self, data_directory):
        """
        Constructor
        :param data_directory: directory with VOC data
        """

        self.voc_samples_generator_factory = VOCSamplesGeneratorFactory(data_directory)

    def get_generator(self, size_factor, indices_to_colors_map):
        """
        Returns generator that yields (image_path, segmentation_image) pair on each yield
        :param size_factor: int, value by which height and with of outputs must be divisible
        :return: generator
        """

        voc_samples_generator = self.voc_samples_generator_factory.get_generator(size_factor)

        while True:

            image, segmentation = next(voc_samples_generator)

            segmentation_cube = get_segmentation_cube(segmentation, indices_to_colors_map)
            yield image, segmentation_cube

    def get_size(self):
        """
        Gets size of dataset served by the generator
        :return: int
        """

        return self.voc_samples_generator_factory.get_size()


def get_colors_info(categories_count):
    """
    Get ids to colors dictionary and void color.
    Ids to colors dictionary maps gives colors used in VOC dataset for a given category id.
    Void color represents ambiguous regions in segmentations.
    All colors are returned in BGR order.
    Code adapted from https://gist.github.com/wllhf/a4533e0adebe57e3ed06d4b50c8419ae
    :param categories_count: number of categories - includes background, but doesn't include void
    :return: map, tuple
    """

    colors_count = 256

    def bitget(byte_value, idx):
        """
        Check if bit at given byte index is set
        :param byte_value: byte
        :param idx: index
        :return: bool
        """
        return (byte_value & (1 << idx)) != 0

    colors_matrix = np.zeros(shape=(colors_count, 3), dtype=np.int)

    for color_index in range(colors_count):

        red = green = blue = 0
        color = color_index

        for j in range(8):

            red = red | (bitget(color, 0) << 7 - j)
            green = green | (bitget(color, 1) << 7 - j)
            blue = blue | (bitget(color, 2) << 7 - j)
            color = color >> 3

        # Writing colors in BGR order, since our image reading and logging routines use it
        colors_matrix[color_index] = blue, green, red

    indices_to_colors_map = {color_index: tuple(colors_matrix[color_index]) for color_index in range(categories_count)}
    return indices_to_colors_map, tuple(colors_matrix[-1])


def get_void_mask(segmentation_image, void_color):
    """
    Compute a 2D void segmentation given segmentation image and void_color
    :param segmentation_image: numpy array, 3-channel segmentation image
    :param void_color: 3-element tuple representing void color
    :return: 2D binary array with 1 at void color location and 0 elsewhere
    """

    return np.all(segmentation_image == void_color, axis=-1).astype(np.int32)


def get_segmentation_cube(segmentation_image, indices_to_colors_map):
    """
    Turns 2D 3-channel segmentation image with into a batch of 2D binary maps - one for each
    segmentation category
    :param segmentation_image: 2D 3-channel segmentation image
    :param indices_to_colors_map: dictionary mapping categories indices to image colors
    :return: 3D array with a binary 2D map for each category at a corresponding index
    """

    categories_count = len(indices_to_colors_map.keys())

    shape = segmentation_image.shape[:2] + (categories_count,)
    segmentation_cube = np.zeros(shape, dtype=np.int32)

    for index, color in indices_to_colors_map.items():

        segmentation_mask = np.all(segmentation_image == color, axis=-1)
        segmentation_cube[:, :, index] = segmentation_mask

    return segmentation_cube


def get_segmentation_image(segmentation_cube, indices_to_colors_map, void_color):
    """
    Turns segmentation cube into a segmentation image.
    :param segmentation_cube: 3D array of segmentation maps, each map for a single category
    :param indices_to_colors_map: dictionary mapping categories indices to colors
    :param void_color: color to be used for areas with no segmentation category specified
    :return: 3D array
    """

    image_shape = segmentation_cube.shape[:2] + (3, )
    image = np.zeros(image_shape, dtype=np.uint8)
    image[:, :] = void_color

    max_segmentation_indices_array = np.argmax(segmentation_cube, axis=2)

    for y in range(image_shape[0]):
        for x in range(image_shape[1]):

            max_segmentation_index = max_segmentation_indices_array[y, x]

            if segmentation_cube[y, x, max_segmentation_index] != 0:
                image[y, x] = indices_to_colors_map[max_segmentation_index]

    return image
