"""
Module for visualization of data generators outputs, model prediction, etc
"""

import argparse
import sys

import yaml
import vlogging
import tqdm
import numpy as np
import tensorflow as tf

import net.utilities
import net.voc
import net.ml


def log_data(logger, generator, categories, indices_to_colors_map, void_color):
    """
    Log single batch of images and segmentation masks
    """

    ids_to_categories_map = net.utilities.get_ids_to_values_map(categories)

    for _ in tqdm.tqdm(range(10)):

        image, segmentation = next(generator)

        void_mask = net.voc.get_void_mask(segmentation, void_color)
        segmentation_cube = net.voc.get_segmentation_cube(segmentation, indices_to_colors_map)

        batch_categories = ["void"]
        segmentation_layers = []

        for index in range(segmentation_cube.shape[-1]):

            segmentation_layer = segmentation_cube[:, :, index]

            if np.any(segmentation_layer):

                batch_categories.append(ids_to_categories_map[index])
                segmentation_layers.append(255 * segmentation_layer)

        shapes = image.shape, segmentation_cube.shape
        footnotes = "{}\n{}".format(batch_categories, shapes)

        images_to_display = [image, segmentation, 255 * void_mask] + segmentation_layers
        logger.info(vlogging.VisualRecord("Data", images_to_display, footnotes=footnotes))


def get_simple_statistics(data):

    statistics_template = "min: {}, max: {}, mean: {}, std: {}"
    return statistics_template.format(np.min(data), np.max(data), np.mean(data), np.std(data))


def compare_predictions(generator, categories, indices_to_colors_map):

    print("Uh, much comparision")

    network = net.ml.FullyConvolutionalNetwork(categories_count=len(categories))

    initialized_variables = tf.global_variables()

    session = tf.keras.backend.get_session()

    uninitialized_variables = set(tf.global_variables()).difference(initialized_variables)

    session.run(tf.variables_initializer(uninitialized_variables))

    for _ in range(10):

        image, segmentation = next(generator)
        segmentation_cube = net.voc.get_segmentation_cube(segmentation, indices_to_colors_map)

        feed_dictionary = {network.input_placeholder: np.array([image])}

        results = session.run(network.predictions, feed_dictionary)

        print()
        print(results.shape)
        print(segmentation_cube.shape)

        # print(results[0].shape)
        # print(results[1].shape)
        # print(get_simple_statistics(results[0]))
        # print(get_simple_statistics(results[1]))
        # print(get_simple_statistics(results[2]))
        # print(get_simple_statistics(scaled_block_4_head))


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

    # log_data(logger, generator, categories, indices_to_colors_map, void_color)
    compare_predictions(generator, categories, indices_to_colors_map)


if __name__ == "__main__":

    main()
