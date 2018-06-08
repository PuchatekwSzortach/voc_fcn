"""
Module with machine learning code
"""

import os

import tensorflow as tf
import numpy as np
import tqdm

import net.voc
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

        self.ops_map["main_head"] = tf.keras.layers.Conv2DTranspose(
            512, kernel_size=(4, 4), strides=(2, 2), activation=tf.nn.swish,
            kernel_initializer=net.utilities.bilinear_initializer)(self.ops_map["block5_pool"])

        self.ops_map["main_head"] = tf.keras.layers.Cropping2D(cropping=((1, 1), (1, 1)))(self.ops_map["main_head"])
        self.ops_map["main_head"] = tf.Variable(1, dtype=tf.float32) * self.ops_map["main_head"]

        self.ops_map["main_head"] = self.ops_map["main_head"] + self.ops_map["block4_pool"]

        self.ops_map["main_head"] = tf.keras.layers.Conv2DTranspose(
            256, kernel_size=(4, 4), strides=(2, 2), activation=tf.nn.swish,
            kernel_initializer=net.utilities.bilinear_initializer)(self.ops_map["main_head"])

        self.ops_map["main_head"] = tf.keras.layers.Cropping2D(cropping=((1, 1), (1, 1)))(self.ops_map["main_head"])
        self.ops_map["main_head"] = tf.Variable(0.01, dtype=tf.float32) * self.ops_map["main_head"]

        self.ops_map["main_head"] = self.ops_map["main_head"] + self.ops_map["block3_pool"]

        self.ops_map["main_head"] = tf.keras.layers.Conv2DTranspose(
            categories_count, kernel_size=(16, 16), strides=(8, 8),
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

        self.labels_placeholder = tf.placeholder(dtype=np.float32, shape=[1, None, None, self.categories_count])

        self.loss_op = tf.losses.softmax_cross_entropy(
            onehot_labels=self.labels_placeholder, logits=self.network.ops_map["logits"])

        self.train_op = tf.train.AdamOptimizer(learning_rate=0.00001).minimize(self.loss_op)
        self.should_continue_training = None

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

        self.should_continue_training = True
        epoch_index = 0

        model_callbacks = callbacks if callbacks is not None else []

        for callback in model_callbacks:
            callback.model = self

        while epoch_index < configuration["epochs"] and self.should_continue_training is True:

            print("Epoch {}/{}".format(epoch_index, configuration["epochs"]))

            epoch_log = {
                "training_loss": self._train_for_one_epoch(training_data_generator_factory),
                "validation_loss": self._get_validation_loss(validation_data_generator_factory)
            }

            print(epoch_log)

            for callback in model_callbacks:
                callback.on_epoch_end(epoch_log)

            epoch_index += 1

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

    def _train_for_one_epoch(self, training_data_generator_factory):

        training_data_generator = training_data_generator_factory.get_generator()

        training_losses = []

        # for _ in tqdm.tqdm(range(10)):
        for _ in tqdm.tqdm(range(training_data_generator_factory.get_size())):

            image, segmentation_cube = next(training_data_generator)

            feed_dictionary = {
                self.network.input_placeholder: np.array([image]),
                self.labels_placeholder: np.array([segmentation_cube], dtype=np.float32),
            }

            _, loss = self.session.run([self.train_op, self.loss_op], feed_dictionary)
            training_losses.append(loss)

        return np.mean(training_losses)

    def _get_validation_loss(self, validation_data_generator_factory):

        validation_data_generator = validation_data_generator_factory.get_generator()
        validation_losses = []

        # for _ in tqdm.tqdm(range(10)):
        for _ in tqdm.tqdm(range(validation_data_generator_factory.get_size())):

            image, segmentation_cube = next(validation_data_generator)

            feed_dictionary = {
                self.network.input_placeholder: np.array([image]),
                self.labels_placeholder: np.array([segmentation_cube], dtype=np.float32),
            }

            loss = self.session.run(self.loss_op, feed_dictionary)
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
