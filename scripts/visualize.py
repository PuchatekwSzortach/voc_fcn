"""
Module for visualization of data generators outputs, model prediction, etc
"""

import vlogging
import tqdm
import numpy as np

import net.config
import net.utilities
import net.voc


def log_batch(logger, images, segmentations, indices_to_colors_map, void_color):
    """
    Log single batch of images and segmentation masks
    """

    for image, segmentation_image in zip(images, segmentations):

        void_mask = net.voc.get_void_mask(segmentation_image, void_color)
        segmentation_cube = net.voc.get_segmentation_cube(segmentation_image, indices_to_colors_map)

        categories = ["void"]
        segmentation_layers = []

        for index in range(segmentation_cube.shape[-1]):

            segmentation_layer = segmentation_cube[:, :, index]

            if np.any(segmentation_layer):
                categories.append(net.config.CATEGORIES[index])
                segmentation_layers.append(255 * segmentation_layer)

        images_to_display = [image, segmentation_image, 255 * void_mask] + segmentation_layers
        logger.info(vlogging.VisualRecord("Data", images_to_display, footnotes=categories))


def main():
    """
    Main runner
    """

    logger = net.utilities.get_logger("/tmp/voc_fcn.html")

    generator = net.voc.BatchesGeneratorFactory(net.config.DATA_DIRECTORY).get_generator(batch_size=1)

    indices_to_colors_map, _, void_color = net.voc.get_colors_info(
        len(net.config.CATEGORIES))

    for _ in tqdm.tqdm(range(10)):

        images, segmentations = next(generator)
        log_batch(logger, images, segmentations, indices_to_colors_map, void_color)


if __name__ == "__main__":

    main()
