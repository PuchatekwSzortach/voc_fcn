"""
Data generators and other data-related code
"""

import os
import random
import copy
import collections
import threading
import queue

import numpy as np
import scipy.io
import cv2

import net.utilities


def get_dataset_filenames(data_directory, data_set_path):
    """
    Get a list of filenames for the dataset
    :param data_directory: path to data directory
    :param data_set_path: path to file containing dataset filenames. This path is relative to data_directory
    :return: list of strings, filenames of images used in dataset
    """

    with open(os.path.join(data_directory, data_set_path)) as file:

        return [line.strip() for line in file.readlines()]


def get_images_paths_and_segmentations_paths_tuples(data_directory, data_set_path):
    """
    Returns a list of tuples, each tuple is an image path, segmentation path pair for a single image
    :param data_directory: VOC data directory
    :param data_set_path: path to list of filenames to be read from from data directory
    :return: list of tuples
    """

    file_stems = get_dataset_filenames(data_directory, data_set_path)

    images_paths_and_segmentation_paths_tuples = []

    for file_stem in file_stems:

        image_path = os.path.join(data_directory, "JPEGImages/{}.jpg".format(file_stem))
        segmentation_path = os.path.join(data_directory, "SegmentationClass/{}.png".format(file_stem))

        images_paths_and_segmentation_paths_tuples.append((image_path, segmentation_path))

    return images_paths_and_segmentation_paths_tuples


class VOCSamplesGeneratorFactory:
    """
    Factory class creating data batches generators that yield (image, segmentation image) pairs
    """

    def __init__(self, data_directory, data_set_path, size_factor):
        """
        Constructor
        :param data_directory: directory with VOC data
        :param data_set_path: path to list of filenames to be read from from data directory
        :param size_factor: int, value by which height and with of outputs must be divisible
        """

        self.images_paths_and_segmentations_paths_tuples = \
            get_images_paths_and_segmentations_paths_tuples(data_directory, data_set_path)

        self.size_factor = size_factor

    def get_generator(self):
        """
        Returns generator that yields (image_path, segmentation_image) pair on each yield

        :return: generator
        """

        local_images_paths_and_segmentations_paths_tuples = \
            copy.deepcopy(self.images_paths_and_segmentations_paths_tuples)

        while True:

            random.shuffle(local_images_paths_and_segmentations_paths_tuples)

            for image_path, segmentation_path in local_images_paths_and_segmentations_paths_tuples:

                image = cv2.imread(image_path)
                segmentation = cv2.imread(segmentation_path)

                target_size = net.utilities.get_target_image_size(image.shape[:2], self.size_factor)
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


class VOCSegmentationsLabelsSamplesGeneratorFactory:
    """
    Factory class creating data batches generators that yield (images batch, segmentations labels batch) pairs.
    Uses a queue internally to perform data loading and processing in a separate thread.
    This speeds up overall training, but requires you to explicitly close the generator thread once you're done
    using it.
    """

    def __init__(self, voc_samples_generator_factory, indices_to_colors_map, void_color, batch_size, use_augmentation):
        """
        Constructor
        :param voc_samples_generator_factory: factory that creates generator yielding (image, segmentation) tuples
        from VOC dataset
        :param indices_to_colors_map: dictionary mapping categories indices to colors
        :param void_color: 3-elements tuple of ints, specifies color of pixels without a category
        :param batch_size: int, specifies size of batches yielded by generator
        :param use_augmentation: bool, triggers data augmentation
        """

        self.voc_samples_generator_factory = voc_samples_generator_factory

        self.batch_generations_args_map = {
            "indices_to_colors_map": indices_to_colors_map,
            "void_color": void_color,
            "use_augmentation": use_augmentation,
            "batch_size": batch_size
        }

        self._batches_queue = queue.Queue(maxsize=100)
        self._batch_generation_thread = None
        self._continue_serving_batches = None

    def get_generator(self):
        """
        Returns generator that yields (image_path, segmentation_image) pair on each yield

        :return: generator
        """

        samples_generator = self.voc_samples_generator_factory.get_generator()

        self._continue_serving_batches = True

        self._batch_generation_thread = threading.Thread(
            target=self._batch_generation_task,
            args=(self.batch_generations_args_map, samples_generator, self._batches_queue))

        self._batch_generation_thread.start()

        while True:

            batch = self._batches_queue.get()
            self._batches_queue.task_done()
            yield batch

    def get_size(self):
        """
        Gets size of dataset served by the generator
        :return: int
        """

        return self.voc_samples_generator_factory.get_size()

    def stop_generator(self):
        """
        Signal data loading thread to finish working and purge the data queue.
        """

        self._continue_serving_batches = False

        while not self._batches_queue.empty():
            self._batches_queue.get()
            self._batches_queue.task_done()

        self._batches_queue.join()
        self._batch_generation_thread.join()

    def _batch_generation_task(self, batch_generations_args_map, samples_generator, batches_queue):

        images_batches_map = collections.defaultdict(list)
        segmentations_labels_batches_map = collections.defaultdict(list)
        masks_batches_map = collections.defaultdict(list)

        while self._continue_serving_batches is True:

            image, segmentation = next(samples_generator)

            if batch_generations_args_map["use_augmentation"]:
                image, segmentation = net.utilities.DataAugmenter.augment_samples(
                    image, segmentation, batch_generations_args_map["void_color"])

            segmentations_labels_image = net.utilities.get_segmentation_labels_image(
                segmentation, batch_generations_args_map["indices_to_colors_map"])

            # Mask that lets us differentiate pixels with and without categories
            mask = np.all(segmentation != batch_generations_args_map["void_color"], axis=2).astype(np.int32)

            images_batches_map[image.shape].append(image)
            segmentations_labels_batches_map[image.shape].append(segmentations_labels_image)
            masks_batches_map[image.shape].append(mask)

            if len(images_batches_map[image.shape]) == batch_generations_args_map["batch_size"]:

                batch = np.array(images_batches_map[image.shape]), \
                        np.array(segmentations_labels_batches_map[image.shape]), \
                        np.array(masks_batches_map[image.shape])

                batches_queue.put(batch)

                images_batches_map[image.shape].clear()
                segmentations_labels_batches_map[image.shape].clear()
                masks_batches_map[image.shape].clear()


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

    max_segmentation_indices_matrix = np.argmax(segmentation_cube, axis=2)
    max_segmentation_values = np.max(segmentation_cube, axis=2)

    for index, color in indices_to_colors_map.items():

        pixels_to_draw = (max_segmentation_indices_matrix == index) & (max_segmentation_values != 0)
        image[pixels_to_draw] = color

    return image


class CombinedPASCALDatasetsGeneratorFactory:
    """
    Factory class that merges VOC 2012 and Hariharan's PASCAL datasets.
    Builds a generator for that returns (image, segmentation) tuples.
    """

    def __init__(self, voc_config, hariharan_config, size_factor, categories_count):
        """
        Constructor
        :param voc_config: dictionary with voc dataset paths
        :param hariharan_config: dictionary with Hariharan's PASCAL dataset paths
        :param size_factor: int, value by which height and with of outputs must be divisible
        :param categories_count: int, number of categories in the datasets
        """

        self.voc_config = voc_config
        self.hariharan_config = hariharan_config
        self.size_factor = size_factor

        self.indices_to_colors_map, self.void_color = get_colors_info(categories_count)
        self.combined_datasets_filenames = self._get_combined_datasets_filenames()

    def get_generator(self):
        """
        Returns generator that yields (image, segmentation) tuple on each yield.
        :return: generator
        """

        local_combined_datasets_filenames = copy.deepcopy(self.combined_datasets_filenames)

        sample_getters_map = {
            "voc": self._get_voc_sample,
            "hariharan": self._get_hariharan_sample
        }

        while True:

            random.shuffle(local_combined_datasets_filenames)

            for dataset, filename in local_combined_datasets_filenames:

                image, segmentation = sample_getters_map[dataset](filename)

                target_size = net.utilities.get_target_image_size(image.shape[:2], self.size_factor)
                target_size = target_size[1], target_size[0]

                image = cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC)
                segmentation = cv2.resize(segmentation, target_size, interpolation=cv2.INTER_NEAREST)

                yield image, segmentation

    def get_size(self):
        """
        Gets size of dataset served by the generator
        :return: int
        """
        return len(self.combined_datasets_filenames)

    def _get_combined_datasets_filenames(self):

        # Remove from hariharan images that appear in voc
        voc_filenames_list = get_dataset_filenames(
            self.voc_config["data_directory"], self.voc_config["data_set_path"])

        hariharan_filenames_list = get_dataset_filenames(
            self.hariharan_config["data_directory"], self.hariharan_config["data_set_path"])

        # hariharan's files that don't appear in voc
        unique_hariharan_filenames_list = list(set(hariharan_filenames_list).difference(voc_filenames_list))

        combined_datasets_filenames = \
            [("voc", filename) for filename in voc_filenames_list] + \
            [("hariharan", filename) for filename in unique_hariharan_filenames_list]

        return combined_datasets_filenames

    def _get_voc_sample(self, filename):

        image_path = os.path.join(self.voc_config["data_directory"], "JPEGImages/{}.jpg".format(filename))
        segmentation_path = os.path.join(self.voc_config["data_directory"], "SegmentationClass/{}.png".format(filename))

        return cv2.imread(image_path), cv2.imread(segmentation_path)

    def _get_hariharan_sample(self, filename):

        image_path = os.path.join(self.hariharan_config["data_directory"], "dataset/img", filename + ".jpg")
        image = cv2.imread(image_path)

        segmentation_path = os.path.join(self.hariharan_config["data_directory"], "dataset/cls", filename + ".mat")
        segmentation_data = scipy.io.loadmat(segmentation_path)
        segmentation_matrix = segmentation_data["GTcls"][0][0][1]

        segmentation = self.void_color * np.ones(shape=image.shape, dtype=np.uint8)

        for category_index in set(segmentation_matrix.reshape(-1)):

            segmentation[segmentation_matrix == category_index] = self.indices_to_colors_map[category_index]

        return image, segmentation
