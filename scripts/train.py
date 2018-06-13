"""
Script for training FCN net
"""

import argparse
import sys

import yaml
import tensorflow as tf

import net.data
import net.ml
import net.callbacks


def get_data_generators_factories(config):
    """
    Simple helper to prepare data generators factories used to train model
    :param config: script configuration object
    :return: two instances of VOCOneHotEncodedSamplesGeneratorFactoryTwo, one configured for training
    and one configured for validation
    """

    categories = config["categories"]

    indices_to_colors_map, _ = net.data.get_colors_info(len(categories))

    voc_train_config = {
        "data_directory": config["voc"]["data_directory"],
        "data_set_path": config["voc"]["train_set_path"],
    }

    hariharan_train_config = {
        "data_directory": config["hariharan"]["data_directory"],
        "data_set_path": config["hariharan"]["train_set_path"],
    }

    training_data_segmentation_samples_generator_factory = net.data.CombinedPASCALDatasetsGeneratorFactory(
        voc_train_config, hariharan_train_config, config["size_factor"],
        len(config["categories"]), use_augmentation=True)

    training_data_generator_factory = net.data.VOCOneHotEncodedSamplesGeneratorFactoryTwo(
        training_data_segmentation_samples_generator_factory, indices_to_colors_map, config["train"]["batch_size"])

    validation_data_segmentation_samples_generator_factory = net.data.VOCSamplesGeneratorFactory(
        config["voc"]["data_directory"], config["voc"]["validation_set_path"], config["size_factor"],
        use_augmentation=False)

    validation_data_generator_factory = net.data.VOCOneHotEncodedSamplesGeneratorFactoryTwo(
        validation_data_segmentation_samples_generator_factory, indices_to_colors_map, config["train"]["batch_size"])

    return training_data_generator_factory, validation_data_generator_factory


def main():
    """
    Main driver
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('--config', action="store", required=True)
    arguments = parser.parse_args(sys.argv[1:])

    with open(arguments.config) as file:
        config = yaml.safe_load(file)

    network = net.ml.FullyConvolutionalNetwork(categories_count=len(config["categories"]))
    initialized_variables = tf.global_variables()
    session = tf.keras.backend.get_session()
    model = net.ml.Model(session, network, config["categories"])

    # Load previous checkpoint if requested
    if config["train"]["restore_last_checkpoint"] is True:

        print("Loading existing weights")
        model.load(config["model_checkpoint_path"])

    uninitialized_variables = set(tf.global_variables()).difference(initialized_variables)
    session.run(tf.variables_initializer(uninitialized_variables))

    training_data_generator_factory, validation_data_generator_factory = get_data_generators_factories(config)

    callbacks = [
        net.callbacks.ModelCheckpoint(
            config["model_checkpoint_path"], config["train"]["model_checkpoint_skip_epochs"]),
        net.callbacks.EarlyStopping(config["train"]["early_stopping_patience"]),
        net.callbacks.ReduceLearningRateOnPlateau(
            config["train"]["reduce_learning_rate_patience"],
            config["train"]["reduce_learning_rate_factor"])
    ]

    model.train(training_data_generator_factory, validation_data_generator_factory, config["train"], callbacks)


if __name__ == "__main__":
    main()
