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


def log_voc_samples_generator_output(logger, configuration):
    """
    Logs voc samples generator output
    """

    generator = net.voc.VOCSamplesGeneratorFactory(
        configuration["data_directory"]).get_generator(configuration["size_factor"])

    ids_to_colors_map, void_color = net.voc.get_colors_info(len(configuration["categories"]))
    ids_to_categories_map = net.utilities.get_ids_to_values_map(configuration["categories"])

    for _ in tqdm.tqdm(range(10)):

        image, segmentation = next(generator)

        void_mask = net.voc.get_void_mask(segmentation, void_color)
        segmentation_cube = net.voc.get_segmentation_cube(segmentation, ids_to_colors_map)

        categories_segementations_map = net.utilities.get_categories_segmentations_maps(
            segmentation_cube, ids_to_categories_map)

        categories, segmentation_layers = zip(*categories_segementations_map.items())

        images_to_display = [image, segmentation, 255 * void_mask] + [255 * image for image in segmentation_layers]
        logger.info(vlogging.VisualRecord("Data", images_to_display, footnotes=str(["void"] + list(categories))))


def log_one_hot_encoded_voc_samples_generator_output(logger, configuration):
    """
    Logs one hot encoded voc samples generator output
    """

    indices_to_colors_map, void_color = net.voc.get_colors_info(len(configuration["categories"]))

    generator_factory = net.voc.VOCOneHotEncodedSamplesGeneratorFactory(configuration["data_directory"])
    generator = generator_factory.get_generator(configuration["size_factor"], indices_to_colors_map)

    for _ in tqdm.tqdm(range(10)):

        image, segmentation_cube = next(generator)
        segmentation_image = net.voc.get_segmentation_image(segmentation_cube, indices_to_colors_map, void_color)

        logger.info(vlogging.VisualRecord("Data", [image, segmentation_image]))


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

    # log_voc_samples_generator_output(logger, config)
    log_one_hot_encoded_voc_samples_generator_output(logger, config)


if __name__ == "__main__":

    main()
