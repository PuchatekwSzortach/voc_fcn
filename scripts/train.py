"""
Script for training FCN net
"""

import argparse
import sys

import yaml
import tensorflow as tf

import net.voc
import net.ml
import net.callbacks


def main():
    """
    Main driver
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('--config', action="store", required=True)
    arguments = parser.parse_args(sys.argv[1:])

    with open(arguments.config) as file:
        config = yaml.safe_load(file)

    categories = config["categories"]

    indices_to_colors_map, _ = net.voc.get_colors_info(len(categories))

    training_data_generator_factory = net.voc.VOCOneHotEncodedSamplesGeneratorFactory(
        config["data_directory"], config["train_set_path"], config["size_factor"], indices_to_colors_map)

    validation_data_generator_factory = net.voc.VOCOneHotEncodedSamplesGeneratorFactory(
        config["data_directory"], config["validation_set_path"], config["size_factor"], indices_to_colors_map)

    network = net.ml.FullyConvolutionalNetwork(categories_count=len(categories))

    initialized_variables = tf.global_variables()

    session = tf.keras.backend.get_session()

    callbacks = [
        net.callbacks.ModelCheckpoint(
            config["model_checkpoint_path"], config["train"]["model_checkpoint_skip_epochs"]),
        net.callbacks.EarlyStopping(config["train"]["early_stopping_patience"]),
        net.callbacks.ReduceLearningRateOnPlateau(
            config["train"]["reduce_learning_rate_patience"],
            config["train"]["reduce_learning_rate_factor"])
    ]

    model = net.ml.Model(session, network, categories)

    # Load previous checkpoint if requested
    if config["train"]["restore_last_checkpoint"] is True:

        print("Loading existing weights")
        model.load(config["model_checkpoint_path"])

    uninitialized_variables = set(tf.global_variables()).difference(initialized_variables)
    session.run(tf.variables_initializer(uninitialized_variables))

    model.train(training_data_generator_factory, validation_data_generator_factory, config["train"], callbacks)


if __name__ == "__main__":
    main()
