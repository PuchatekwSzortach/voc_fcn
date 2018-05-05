"""
Module for visualization of data generators outputs, model prediction, etc
"""

import argparse
import sys

import yaml
import vlogging
import tqdm
import numpy as np

import net.utilities
import net.voc


def log_batch(logger, images, segmentations, categories, indices_to_colors_map, void_color):
    """
    Log single batch of images and segmentation masks
    """

    ids_to_categories_map = net.utilities.get_ids_to_values_map(categories)

    for image, segmentation_image in zip(images, segmentations):

        void_mask = net.voc.get_void_mask(segmentation_image, void_color)
        segmentation_cube = net.voc.get_segmentation_cube(segmentation_image, indices_to_colors_map)

        batch_categories = ["void"]
        segmentation_layers = []

        for index in range(segmentation_cube.shape[-1]):

            segmentation_layer = segmentation_cube[:, :, index]

            if np.any(segmentation_layer):

                batch_categories.append(ids_to_categories_map[index])
                segmentation_layers.append(255 * segmentation_layer)

        shapes = image.shape, segmentation_cube.shape
        footnotes = "{}\n{}".format(batch_categories, shapes)

        images_to_display = [image, segmentation_image, 255 * void_mask] + segmentation_layers
        logger.info(vlogging.VisualRecord("Data", images_to_display, footnotes=footnotes))


def main():
    """
    Main runner
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('--config', action="store", required=True)
    arguments = parser.parse_args(sys.argv[1:])

    with open(arguments.config) as file:
        config = yaml.safe_load(file)

    logger = net.utilities.get_logger(config["log_path"])
    categories = config["categories"]

    indices_to_colors_map, _, void_color = net.voc.get_colors_info(len(categories))

    generator_factory = net.voc.BatchesGeneratorFactory(config["data_directory"])
    generator = generator_factory.get_generator(config["size_factor"], config["batch_size"])

    indices_to_colors_map, _, void_color = net.voc.get_colors_info(len(categories))

    for _ in tqdm.tqdm(range(10)):

        images, segmentations = next(generator)
        log_batch(logger, images, segmentations, categories, indices_to_colors_map, void_color)


if __name__ == "__main__":

    main()
