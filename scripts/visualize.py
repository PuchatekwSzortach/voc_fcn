"""
Module for visualization of data generators outputs, model prediction, etc
"""

import vlogging
import cv2
import tqdm
import numpy as np

import net.config
import net.utilities
import net.voc


def main():

    logger = net.utilities.get_logger("/tmp/voc_fcn.html")

    generator = net.voc.BatchesGeneratorFactory(net.config.data_directory).get_generator(batch_size=1)

    indices_to_colors_map, colors_to_indices_map, void_color = net.voc.get_colors_info(
        len(net.config.categories))

    for index in tqdm.tqdm(range(10)):

        images, segmentations = next(generator)

        for image, segmentation in zip(images, segmentations):

            segmentation_colors = net.utilities.get_image_colors(segmentation)

            categories = []

            for color in segmentation_colors:

                inverted_color = color[::-1]

                if inverted_color == void_color:

                    categories.append("void")

                else:

                    index = colors_to_indices_map[inverted_color]
                    categories.append(net.config.categories[index])

            logger.info(vlogging.VisualRecord("Data", [image, segmentation], footnotes=categories))


if __name__ == "__main__":

    main()
