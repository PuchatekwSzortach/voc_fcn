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

        data_map = {}

        for segmentation_path in segmentation_paths:
            file_name_with_extension = os.path.basename(segmentation_path)
            file_name = os.path.splitext(file_name_with_extension)[0]

            image_path = [path for path in images_paths if file_name in path][0]

            data_map[file_name] = (image_path, segmentation_path)

        return data_map
