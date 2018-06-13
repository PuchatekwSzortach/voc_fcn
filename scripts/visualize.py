"""
Module for visualization of data generators outputs, model prediction, etc
"""

import argparse
import sys

import yaml
import vlogging
import tqdm
import tensorflow as tf

import net.utilities
import net.data
import net.ml


def log_voc_samples_generator_output(logger, config):
    """
    Logs voc samples generator output
    """

    voc_config = {
        "data_directory": config["voc"]["data_directory"],
        "data_set_path": config["voc"]["train_set_path"],
    }

    hariharan_config = {
        "data_directory": config["hariharan"]["data_directory"],
        "data_set_path": config["hariharan"]["train_set_path"],
    }

    generator = net.data.CombinedPASCALDatasetsGeneratorFactory(
        voc_config, hariharan_config, config["size_factor"],
        len(config["categories"]), use_augmentation=True).get_generator()

    ids_to_colors_map, void_color = net.data.get_colors_info(len(config["categories"]))
    ids_to_categories_map = net.utilities.get_ids_to_values_map(config["categories"])

    for _ in tqdm.tqdm(range(40)):

        image, segmentation = next(generator)

        image, segmentation = net.utilities.DataAugmenter.rotate_samples(image, segmentation, void_color)

        segmentation_cube = net.data.get_segmentation_cube(segmentation, ids_to_colors_map)

        categories, segmentation_layers = zip(*net.utilities.get_categories_segmentations_maps(
            segmentation_cube, ids_to_categories_map).items())

        images_to_display = \
            [image, segmentation, 255 * net.data.get_void_mask(segmentation, void_color)] + \
            net.utilities.get_uint8_images(segmentation_layers)

        logger.info(vlogging.VisualRecord("Data", images_to_display, footnotes=str(["void"] + list(categories))))


def log_one_hot_encoded_voc_samples_generator_output(logger, configuration):
    """
    Logs one hot encoded voc samples generator output
    """

    indices_to_colors_map, void_color = net.data.get_colors_info(len(configuration["categories"]))

    data_segmentation_samples_generator_factory = net.data.VOCSamplesGeneratorFactory(
        configuration["voc"]["data_directory"], configuration["voc"]["validation_set_path"],
        configuration["size_factor"], use_augmentation=False)

    generator_factory = net.data.VOCOneHotEncodedSamplesGeneratorFactory(
        data_segmentation_samples_generator_factory, indices_to_colors_map)

    generator = generator_factory.get_generator()

    for _ in tqdm.tqdm(range(40)):

        image, segmentation_cube = next(generator)
        segmentation_image = net.data.get_segmentation_image(segmentation_cube, indices_to_colors_map, void_color)

        logger.info(vlogging.VisualRecord("Data", [image, segmentation_image]))


def log_trained_model_predictions(logger, configuration):
    """
    Logs trained model's predictions
    """

    indices_to_colors_map, void_color = net.data.get_colors_info(len(configuration["categories"]))

    network = net.ml.FullyConvolutionalNetwork(categories_count=len(configuration["categories"]))

    session = tf.keras.backend.get_session()
    model = net.ml.Model(session, network, configuration["categories"])

    model.load(configuration["model_checkpoint_path"])

    generator_factory = net.data.VOCSamplesGeneratorFactory(
        configuration["voc"]["data_directory"], configuration["voc"]["validation_set_path"],
        configuration["size_factor"], use_augmentation=False)

    generator = generator_factory.get_generator()

    for _ in tqdm.tqdm(range(40)):

        image, segmentation_image = next(generator)

        predicted_segmentation_cube = model.predict(image)

        predicted_segmentation_image = net.data.get_segmentation_image(
            predicted_segmentation_cube, indices_to_colors_map, void_color)

        logger.info(vlogging.VisualRecord("Data", [image, segmentation_image, predicted_segmentation_image]))


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

    log_voc_samples_generator_output(logger, config)
    # log_one_hot_encoded_voc_samples_generator_output(logger, config)
    # log_trained_model_predictions(logger, config)


if __name__ == "__main__":

    main()
