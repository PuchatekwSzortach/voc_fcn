"""
Module for visualization of data generators outputs, model prediction, etc
"""

import vlogging
import cv2
import tqdm
import numpy as np
import pprint

import net.config
import net.utilities
import net.voc


def main():

    logger = net.utilities.get_logger("/tmp/voc_fcn.html")

    generator = net.voc.BatchesGeneratorFactory(net.config.data_directory).get_generator(batch_size=1)

    indices_to_colors_map, colors_to_indices_map, void_color = net.voc.get_colors_info(
        len(net.config.categories))

    for _ in tqdm.tqdm(range(10)):

        images, segmentations = next(generator)

        for image, segmentation_image in zip(images, segmentations):

            void_mask = net.voc.get_void_mask(segmentation_image, void_color)
            segmentation_cube = net.voc.get_segmentation_cube(segmentation_image, indices_to_colors_map)

            categories = ["void"]
            segmentation_layers = []

            for index in range(segmentation_cube.shape[-1]):

                segmentation_layer = segmentation_cube[:, :, index]

                if np.any(segmentation_layer):

                    categories.append(net.config.categories[index])
                    segmentation_layers.append(255 * segmentation_layer)

            images_to_display = [image, segmentation_image, 255 * void_mask] + segmentation_layers
            logger.info(vlogging.VisualRecord("Data", images_to_display, footnotes=categories))


if __name__ == "__main__":

    main()
