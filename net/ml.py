"""
Module with machine learning code
"""

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

    def train(self, generator, data_size, indices_to_colors_map, configuration, callbacks=None):
        """
        Trains network
        :param generator: training samples generator
        :param data_size: training dataset size
        :param indices_to_colors_map:
        :param configuration: dictionary with training options
        :param callbacks: list of callbacks to call at end of each epoch
        """

        model_callbacks = callbacks if callbacks is not None else []

        for callback in model_callbacks:
            callback.model = self

        for _ in range(configuration["epochs"]):

            losses = []

            for _ in tqdm.tqdm(range(data_size)):

                image, segmentation = next(generator)

                segmentation_cube = net.voc.get_segmentation_cube(segmentation, indices_to_colors_map)

                feed_dictionary = {
                    self.network.input_placeholder: np.array([image]),
                    self.labels_placeholder: np.array([segmentation_cube], dtype=np.float32),
                }

                _, loss = self.session.run([self.train_op, self.loss_op], feed_dictionary)

                losses.append(loss)

            print("Loss: {}".format(np.mean(losses)))
