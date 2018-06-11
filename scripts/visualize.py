"""
Module for visualization of data generators outputs, model prediction, etc
"""

import argparse
import sys
import os
import random

import yaml
import vlogging
import tqdm
import tensorflow as tf
import cv2
import scipy.io
import numpy as np

import net.utilities
import net.voc
import net.ml


def log_voc_samples_generator_output(logger, configuration):
    """
    Logs voc samples generator output
    """

    generator = net.voc.VOCSamplesGeneratorFactory(
        configuration["voc"]["data_directory"], configuration["voc"]["validation_set_path"],
        configuration["size_factor"], use_augmentation=True).get_generator()

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

    generator_factory = net.voc.VOCOneHotEncodedSamplesGeneratorFactory(
        configuration["voc"]["data_directory"], configuration["voc"]["validation_set_path"],
        configuration["size_factor"], indices_to_colors_map, use_augmentation=False)

    generator = generator_factory.get_generator()

    for _ in tqdm.tqdm(range(40)):

        image, segmentation_cube = next(generator)
        segmentation_image = net.voc.get_segmentation_image(segmentation_cube, indices_to_colors_map, void_color)

        logger.info(vlogging.VisualRecord("Data", [image, segmentation_image]))


def log_trained_model_predictions(logger, configuration):
    """
    Logs trained model's predictions
    """

    indices_to_colors_map, void_color = net.voc.get_colors_info(len(configuration["categories"]))

    network = net.ml.FullyConvolutionalNetwork(categories_count=len(configuration["categories"]))

    session = tf.keras.backend.get_session()
    model = net.ml.Model(session, network, configuration["categories"])

    model.load(configuration["model_checkpoint_path"])

    generator_factory = net.voc.VOCSamplesGeneratorFactory(
        configuration["voc"]["data_directory"], configuration["voc"]["validation_set_path"],
        configuration["size_factor"], use_augmentation=False)

    generator = generator_factory.get_generator()

    for _ in tqdm.tqdm(range(40)):

        image, segmentation_image = next(generator)

        predicted_segmentation_cube = model.predict(image)

        predicted_segmentation_image = net.voc.get_segmentation_image(
            predicted_segmentation_cube, indices_to_colors_map, void_color)

        logger.info(vlogging.VisualRecord("Data", [image, segmentation_image, predicted_segmentation_image]))


def log_hariharan_dataset(logger, config):

    data_directory = config["hariharan"]["data_directory"]
    dataset_path = config["hariharan"]["validation_set_path"]

    with open(os.path.join(data_directory, dataset_path)) as file:

        images_filenames = [line.split()[0] for line in file.readlines()]

    random.shuffle(images_filenames)

    indices_to_colors_map, void_color = net.voc.get_colors_info(len(config["categories"]))

    for image_filename in tqdm.tqdm(images_filenames[:20]):

        image_path = os.path.join(data_directory, "dataset/img", image_filename + ".jpg")
        image = cv2.imread(image_path)

        segmentation_path = os.path.join(data_directory, "dataset/cls", image_filename + ".mat")
        segmentation_data = scipy.io.loadmat(segmentation_path)
        segmentation_matrix = segmentation_data["GTcls"][0][0][1]

        segmentation = np.zeros(shape=image.shape, dtype=np.uint8)

        for category_index in set(segmentation_matrix.reshape(-1)):

            segmentation[segmentation_matrix == category_index] = indices_to_colors_map[category_index]

        logger.info(vlogging.VisualRecord("Data", [image, segmentation]))


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
    # log_one_hot_encoded_voc_samples_generator_output(logger, config)
    # log_trained_model_predictions(logger, config)
    log_hariharan_dataset(logger, config)


if __name__ == "__main__":

    main()
