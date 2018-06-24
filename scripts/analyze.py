"""
Script for analyzing model's performance
"""

import argparse
import sys
import collections

import yaml
import tensorflow as tf
import tqdm
import numpy as np

import net.data
import net.ml
import net.utilities


def report_iou_results(categories_intersections_counts_map, categories_unions_counts_map):
    """
    Reports iou analysis results
    :param categories_intersections_counts_map: dictionary mapping categories to a list of intersection counts
     for different images for that category
    :param categories_unions_counts_map: dictionary mapping categories to a list of unions counts
     for different images for that category
    """

    categories = sorted(categories_intersections_counts_map.keys())
    categories_means = []

    for category in categories:

        category_intersections_counts = categories_intersections_counts_map[category]
        category_unions_counts = categories_unions_counts_map[category]

        category_mean = np.sum(category_intersections_counts) / np.sum(category_unions_counts)
        print("{} mean iou -> {:.5f}".format(category, category_mean))

        categories_means.append(category_mean)

    print("\nMean iou across all categories: {:.5f}".format(np.mean(categories_means)))


def get_segmentation_cubes_generator(samples_generator, model, indices_to_colors_map, void_color):
    """
    Get a generator that uses samples_generator to obtain (image, segmentation) tuple and yields a tuple
    (ground_truth_segmentation_cube, predicted_segmentation_cube)
    :param samples_generator: generator that yields (image, segmentation) tuple
    :param model: net.ml.Model instance
    :param indices_to_colors_map: dictionary mapping categories indices to their colors in segmentation images
    :param void_color: 3-elements tuple that represents color of pixels without a category
    :return: generator that yields (ground_truth_segmentation_cube, predicted_segmentation_cube) tuples
    """

    while True:

        image, segmentation = next(samples_generator)
        ground_truth_segmentation_cube = net.data.get_segmentation_cube(segmentation, indices_to_colors_map)

        # Raw predictions are floats before thresholding
        raw_predicted_segmentation_cube = model.predict(image)

        predicted_segmentation_image = net.data.get_segmentation_image(
            raw_predicted_segmentation_cube, indices_to_colors_map, void_color)

        predicted_segmentation_cube = net.data.get_segmentation_cube(
            predicted_segmentation_image, indices_to_colors_map)

        yield ground_truth_segmentation_cube, predicted_segmentation_cube


def analyze_iou(model, generator_factory, config):
    """
    Analyses intersection over union of model predictions with ground truth using VOC validation dataset
    :param model: net.ml.Model instance
    :param generator_factory: VOCSamplesGeneratorFactory instance
    :param config: object with configuration details
    """

    indices_to_colors_map, void_color = net.data.get_colors_info(len(config["categories"]))

    segmentation_cubes_generator = get_segmentation_cubes_generator(
        generator_factory.get_generator(), model, indices_to_colors_map, void_color)

    categories_intersections_counts_map = collections.defaultdict(list)
    categories_unions_counts_map = collections.defaultdict(list)

    # for _ in tqdm.tqdm(range(10)):
    for _ in tqdm.tqdm(range(generator_factory.get_size())):

        ground_truth_segmentation_cube, predicted_segmentation_cube = next(segmentation_cubes_generator)

        # Get iou for each category that is present in ground truth cube
        for index, category in enumerate(config["categories"]):

            intersection_pixels = np.logical_and(
                ground_truth_segmentation_cube[:, :, index], predicted_segmentation_cube[:, :, index])

            categories_intersections_counts_map[category].append(np.sum(intersection_pixels))

            union_pixels = np.logical_or(
                ground_truth_segmentation_cube[:, :, index], predicted_segmentation_cube[:, :, index])

            categories_unions_counts_map[category].append(np.sum(union_pixels))

    report_iou_results(categories_intersections_counts_map, categories_unions_counts_map)


def main():
    """
    Script entry point
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('--config', action="store", required=True)
    arguments = parser.parse_args(sys.argv[1:])

    with open(arguments.config) as file:
        config = yaml.safe_load(file)

    network = net.ml.FullyConvolutionalNetwork(categories_count=len(config["categories"]))
    session = tf.keras.backend.get_session()
    model = net.ml.Model(session, network, config["categories"])
    model.load(config["model_checkpoint_path"])

    generator_factory = net.data.VOCSamplesGeneratorFactory(
        config["voc"]["data_directory"], config["voc"]["validation_set_path"], config["size_factor"])

    analyze_iou(model, generator_factory, config)


if __name__ == "__main__":
    main()
