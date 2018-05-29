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

    data_generator_factory = net.voc.VOCSamplesGeneratorFactory(config["data_directory"])
    generator = data_generator_factory.get_generator(size_factor=config["size_factor"])

    network = net.ml.FullyConvolutionalNetwork(categories_count=len(categories))

    initialized_variables = tf.global_variables()

    session = tf.keras.backend.get_session()
    model = net.ml.Model(session, network, categories)

    uninitialized_variables = set(tf.global_variables()).difference(initialized_variables)
    session.run(tf.variables_initializer(uninitialized_variables))

    callbacks = [net.callbacks.LearningRateManager()]

    model.train(
        generator, data_generator_factory.get_size(),
        indices_to_colors_map, config["train"], callbacks)


if __name__ == "__main__":
    main()
