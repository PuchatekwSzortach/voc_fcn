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


def main():

    logger = net.utilities.get_logger("/tmp/voc_fcn.html")

    generator = net.voc.BatchesGeneratorFactory(net.config.data_directory).get_generator(batch_size=1)

    indices_to_colors_map, colors_to_indices_map, void_color = net.utilities.get_colors_info(
        len(net.config.categories))

    print("colors_to_indices_map")
    pprint.pprint(indices_to_colors_map)

    print("Void color: {}".format(void_color))

    for index in tqdm.tqdm(range(10)):

        message = "Image index: {}".format(index)
        print(message)

        images, segmentations = next(generator)

        for image, segmentation in zip(images, segmentations):

            segmentation_colors = net.utilities.get_image_colors(segmentation)

            categories = []

            for color in segmentation_colors:

                inverted_color = color[::-1]

                print("Searching for color: {}".format(np.array(inverted_color)))

                if np.all(inverted_color == void_color):

                    categories.append("void")

                else:

                    index = colors_to_indices_map[tuple(np.array(inverted_color))]
                    categories.append(net.config.categories[index])

            logger.info(vlogging.VisualRecord("Data", [image, segmentation], footnotes=categories))


if __name__ == "__main__":

    main()
