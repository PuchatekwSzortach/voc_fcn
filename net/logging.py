"""
Module with logging utilities
"""

import vlogging

import net.utilities
import net.data


def log_model_analysis(
        logger, image, segmentation_image, model, indices_to_colors_map, void_color, colors_to_ignore):
    """
    Log analysis of model's prediction on a single image.
    Logs input image, ground truth segmentation, predicted segmentation, ground truth segmentation overlaid
    over input image, and predicted segmentation overlaid over input image
    :param logger: logger instance to which analysis should be logged
    :param image: input image for model to predict on
    :param segmentation_image: ground truth segmentation for input image
    :param model: net.ml.Model instance
    :param indices_to_colors_map: dictionary mapping categories ids to segmentation colors
    :param void_color: color representing pixels without any category assigned
    :param colors_to_ignore: list of colors that should be ignored when constructing overlays
    """

    ground_truth_overlay_image = net.utilities.get_segmentation_overlaid_image(
        image, segmentation_image, colors_to_ignore)

    predicted_segmentation_cube = model.predict(image)

    predicted_segmentation_image = net.data.get_segmentation_image(
        predicted_segmentation_cube, indices_to_colors_map, void_color)

    predicted_overlay_image = net.utilities.get_segmentation_overlaid_image(
        image, predicted_segmentation_image, colors_to_ignore)

    logger.info(vlogging.VisualRecord(
        "Data", [image, segmentation_image, predicted_segmentation_image,
                 ground_truth_overlay_image, predicted_overlay_image]))
