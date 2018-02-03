"""
Module for visualization of data generators outputs, model prediction, etc
"""

import glob
import os
import random
import collections
import pprint

import vlogging
import cv2
import xmltodict
import tqdm
import numpy as np

import net.config
import net.utilities
import net.voc


def get_categories(xml_path):

    categories_present = set()

    with open(xml_path) as file:

        xml_data = xmltodict.parse(file.read())

        objects = xml_data["annotation"]["object"]

        if type(objects) is list:

            for single_object in objects:
                categories_present.add(single_object["name"])

        elif type(objects) is collections.OrderedDict:

            categories_present.add(objects["name"])

        else:

            raise ValueError("Unknown objects type: {}".format(type(objects)))

    return categories_present


def main():

    logger = net.utilities.get_logger("/tmp/voc_fcn.html")

    generator = net.voc.BatchesGeneratorFactory(net.config.data_directory).get_generator(4)

    for _ in tqdm.tqdm(range(10)):

        images, segmentations = next(generator)

        for image, segmentation in zip(images, segmentations):

            logger.info(vlogging.VisualRecord("Data", [image, segmentation]))

    # images_paths = glob.glob(os.path.join(net.config.data_directory, "JPEGImages/**.jpg"))
    # segmentation_paths = glob.glob(os.path.join(net.config.data_directory, "SegmentationClass/**.png"))
    # xml_paths = glob.glob(os.path.join(net.config.data_directory, "Annotations/**.xml"))
    #
    # random.shuffle(segmentation_paths)
    #
    # colors_map = net.utilities.get_colors_map()
    #
    # for index in tqdm.tqdm(range(40)):
    #
    #     segmentation_path = segmentation_paths[index]
    #
    #     file_name_with_extension = os.path.basename(segmentation_path)
    #     file_name = os.path.splitext(file_name_with_extension)[0]
    #
    #     image_path = [path for path in images_paths if file_name in path][0]
    #     xml_path = [path for path in xml_paths if file_name in path][0]
    #
    #     categories_present = list(get_categories(xml_path))
    #
    #     image = cv2.imread(image_path)
    #     segmentation = cv2.imread(segmentation_path)
    #
    #     segmentation_colors_set = set()
    #
    #     for y in range(segmentation.shape[0]):
    #
    #         for x in range(segmentation.shape[1]):
    #
    #             color = segmentation[y][x]
    #             segmentation_colors_set.add(tuple(color))
    #
    #     categories_ids = [net.config.categories.index(category) for category in categories_present]
    #     categories_colors = [colors_map[id] for id in categories_ids]
    #
    #     message = "Segmentation colors: {}".format(segmentation_colors_set)
    #     message += "\ncategory - id - color"
    #
    #     categories_colors_images = []
    #
    #     for category, id, color in zip(categories_present, categories_ids, categories_colors):
    #
    #         inverted_color = color[2], color[1], color[0]
    #
    #         message += "\n{} - {} - {}".format(category, id, inverted_color)
    #         category_color_image = np.zeros((200, 200, 3))
    #         category_color_image[:, :, :] = inverted_color
    #
    #         categories_colors_images.append(category_color_image)
    #
    #     logger.info(vlogging.VisualRecord("Data", [image, segmentation] + categories_colors_images, message))


if __name__ == "__main__":

    main()
