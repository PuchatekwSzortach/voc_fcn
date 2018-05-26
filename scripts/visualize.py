"""
Module for visualization of data generators outputs, model prediction, etc
"""

import argparse
import sys

import yaml
import vlogging
import tqdm

import net.utilities
import net.voc
import net.ml


def log_generator_output(logger, generator, categories, indices_to_colors_map, void_color):
    """
    Logs generator output
    """

    ids_to_categories_map = net.utilities.get_ids_to_values_map(categories)

    for _ in tqdm.tqdm(range(10)):

        image, segmentation = next(generator)

        void_mask = net.voc.get_void_mask(segmentation, void_color)
        segmentation_cube = net.voc.get_segmentation_cube(segmentation, indices_to_colors_map)

        categories_segementations_map = net.utilities.get_categories_segmentations_maps(
            segmentation_cube, ids_to_categories_map)

        categories, segmentation_layers = zip(*categories_segementations_map.items())

        images_to_display = [image, segmentation, 255 * void_mask] + [255 * image for image in segmentation_layers]
        logger.info(vlogging.VisualRecord("Data", images_to_display, footnotes=str(["void"] + list(categories))))


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

    generator_factory = net.voc.DataGeneratorFactory(config["data_directory"])
    generator = generator_factory.get_generator(config["size_factor"])

    indices_to_colors_map, _, void_color = net.voc.get_colors_info(len(categories))
    log_generator_output(logger, generator, categories, indices_to_colors_map, void_color)


if __name__ == "__main__":

    main()
