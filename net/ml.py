"""
Module with machine learning code
"""

import os

import tensorflow as tf
import numpy as np
import tqdm

import net.data
import net.utilities


class FullyConvolutionalNetwork:
    """
    Fully Convolutional Network implementation based on VGG
    """

    def __init__(self, categories_count):
        """
        Constructor
        :param categories_count: number of categories to detect
        """

        vgg = tf.keras.applications.VGG16(include_top=False)
        self.input_placeholder = vgg.input

        self.ops_map = {
            "block3_pool": vgg.get_layer("block3_pool").output,
            "block4_pool": vgg.get_layer("block4_pool").output,
            "block5_pool": vgg.get_layer("block5_pool").output,
        }

        filters_count = 128

        self.ops_map["main_head"] = tf.keras.layers.Conv2D(
            filters_count, kernel_size=(1, 1), activation=tf.nn.swish, padding="same")(self.ops_map["block5_pool"])

        self.ops_map["main_head"] = tf.keras.layers.Conv2DTranspose(
            filters_count, kernel_size=(4, 4), strides=(2, 2), activation=tf.nn.swish,
            kernel_initializer=net.utilities.bilinear_initializer)(self.ops_map["main_head"])

        self.ops_map["main_head"] = tf.keras.layers.Cropping2D(cropping=((1, 1), (1, 1)))(self.ops_map["main_head"])
        self.ops_map["main_head"] = tf.Variable(1, dtype=tf.float32) * self.ops_map["main_head"]

        self.ops_map["block4_head"] = tf.keras.layers.Conv2D(
            filters_count, kernel_size=(3, 3), activation=tf.nn.swish, padding="same")(self.ops_map["block4_pool"])

        self.ops_map["main_head"] = self.ops_map["main_head"] + self.ops_map["block4_head"]

        self.ops_map["main_head"] = tf.keras.layers.Conv2D(
            filters_count, kernel_size=(1, 1), activation=tf.nn.swish, padding="same")(self.ops_map["main_head"])

        self.ops_map["main_head"] = tf.keras.layers.Conv2DTranspose(
            filters_count, kernel_size=(4, 4), strides=(2, 2), activation=tf.nn.swish,
            kernel_initializer=net.utilities.bilinear_initializer)(self.ops_map["main_head"])

        self.ops_map["main_head"] = tf.keras.layers.Cropping2D(cropping=((1, 1), (1, 1)))(self.ops_map["main_head"])
        self.ops_map["main_head"] = tf.Variable(0.1, dtype=tf.float32) * self.ops_map["main_head"]

        self.ops_map["block3_head"] = tf.keras.layers.Conv2D(
            filters_count, kernel_size=(3, 3), activation=tf.nn.swish, padding="same")(self.ops_map["block3_pool"])

        self.ops_map["main_head"] = self.ops_map["main_head"] + self.ops_map["block3_head"]

        self.ops_map["main_head"] = tf.keras.layers.Conv2D(
            filters_count, kernel_size=(1, 1), activation=tf.nn.swish, padding="same")(self.ops_map["main_head"])

        self.ops_map["main_head"] = tf.keras.layers.Conv2DTranspose(
            categories_count, kernel_size=(16, 16), strides=(8, 8), activation=tf.nn.swish,
            kernel_initializer=net.utilities.bilinear_initializer)(self.ops_map["main_head"])

        self.ops_map["logits"] = tf.keras.layers.Cropping2D(cropping=((4, 4), (4, 4)))(self.ops_map["main_head"])
        self.ops_map["predictions"] = tf.nn.softmax(self.ops_map["logits"], axis=-1)


class Model:
    """
    Model wraps up a network, providing training and evaluation functions
    """

    def __init__(self, session, network, categories):
        """
        Constructor
        :param session: tensorflow session
        :param network: FullyConvolutionalNetwork instance
        :param categories: categories to detect
        """

        self.session = session
        self.network = network

        self.categories_count = len(categories)
        self.should_continue_training = None
        self.learning_rate = None

        self.ops = {
            "labels_placeholder": tf.placeholder(dtype=tf.int32, shape=[None, None, None]),
            "masks_placeholder": tf.placeholder(dtype=tf.int32, shape=[None, None, None]),
            "learning_rate_placeholder": tf.placeholder(shape=[], dtype=tf.float32)
        }

        self.ops["loss_op"] = tf.losses.sparse_softmax_cross_entropy(
            labels=self.ops["labels_placeholder"], logits=self.network.ops_map["logits"],
            weights=self.ops["masks_placeholder"])

        self.ops["train_op"] = tf.train.AdamOptimizer(
            self.ops["learning_rate_placeholder"]).minimize(self.ops["loss_op"])

    def train(self, training_data_generator_factory, validation_data_generator_factory, configuration, callbacks=None):
        """
        Trains network
        :param training_data_generator_factory:
        factory for creating training data generator and inquiring about its size
        :param validation_data_generator_factory:
        factory for creating validation data generator and inquiring about its size
        :param configuration: dictionary with training options
        :param callbacks: list of callbacks to call at end of each epoch
        """

        self.learning_rate = configuration["learning_rate"]
        self.should_continue_training = True
        epoch_index = 0

        model_callbacks = callbacks if callbacks is not None else []

        for callback in model_callbacks:
            callback.model = self

        while epoch_index < configuration["epochs"] and self.should_continue_training is True:

            print("Epoch {}/{}".format(epoch_index, configuration["epochs"]))

            epoch_log = {
                "epoch_index": epoch_index,
                "training_loss": self._train_for_one_epoch(
                    training_data_generator_factory, configuration["batch_size"]),
                "validation_loss": self._get_validation_loss(
                    validation_data_generator_factory, configuration["batch_size"])
            }

            print(epoch_log)

            for callback in model_callbacks:
                callback.on_epoch_end(epoch_log)

            epoch_index += 1

        training_data_generator_factory.stop_generator()
        validation_data_generator_factory.stop_generator()

    def predict(self, image):
        """
        Computes prediction on a single image
        :param image: numpy array
        :return: segmentation prediction cube
        """

        feed_dictionary = {
            self.network.input_placeholder: np.array([image])
        }

        return self.session.run(self.network.ops_map["predictions"], feed_dictionary)[0]

    def _train_for_one_epoch(self, training_data_generator_factory, batch_size):

        training_data_generator = training_data_generator_factory.get_generator()

        training_losses = []

        for _ in tqdm.tqdm(range(training_data_generator_factory.get_size() // batch_size)):

            images_batch, segmentations_labels_batch, masks_batch = next(training_data_generator)

            feed_dictionary = {
                self.network.input_placeholder: images_batch,
                self.ops["labels_placeholder"]: segmentations_labels_batch,
                self.ops["masks_placeholder"]: masks_batch,
                self.ops["learning_rate_placeholder"]: self.learning_rate
            }

            _, loss = self.session.run([self.ops["train_op"], self.ops["loss_op"]], feed_dictionary)

            training_losses.append(loss)

        return np.mean(training_losses)

    def _get_validation_loss(self, validation_data_generator_factory, batch_size):

        validation_data_generator = validation_data_generator_factory.get_generator()
        validation_losses = []

        for _ in tqdm.tqdm(range(validation_data_generator_factory.get_size() // batch_size)):

            images_batch, segmentations_labels_batch, masks_batch = next(validation_data_generator)

            feed_dictionary = {
                self.network.input_placeholder: images_batch,
                self.ops["labels_placeholder"]: segmentations_labels_batch,
                self.ops["masks_placeholder"]: masks_batch,
            }

            loss = self.session.run(self.ops["loss_op"], feed_dictionary)
            validation_losses.append(loss)

        return np.mean(validation_losses)

    def save(self, save_path):
        """
        Save model's network
        :param save_path: prefix for filenames created for the checkpoint
        """

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        tf.train.Saver().save(self.session, save_path)

    def load(self, save_path):
        """
        Save model's network
        :param save_path: prefix for filenames created for the checkpoint
        """

        tf.train.Saver().restore(self.session, save_path)
