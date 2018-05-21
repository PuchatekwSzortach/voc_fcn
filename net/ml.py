"""
Module with machine learning code
"""

import time

import tensorflow as tf
import numpy as np
import tqdm

import net.voc
import net.utilities


class FullyConvolutionalNetwork:

    def __init__(self, categories_count):

        x = tf.keras.applications.VGG16(include_top=False)
        self.input_placeholder = x.input

        self.ops_map = {
            "block3_pool": x.get_layer("block3_pool").output,
            "block4_pool": x.get_layer("block4_pool").output,
            "block5_pool": x.get_layer("block5_pool").output,
        }

        block5_upscale = tf.keras.layers.Conv2DTranspose(
            filters=256, kernel_size=(4, 4), strides=(2, 2), activation=tf.nn.swish)(self.ops_map["block5_pool"])

        self.ops_map["cropped_block5_upscale"] = tf.keras.layers.Cropping2D(cropping=((1, 1), (1, 1)))(block5_upscale)

        self.ops_map["block_4_head"] = tf.keras.layers.Conv2D(
            filters=256, kernel_size=(3, 3), activation=tf.nn.swish, padding='same')(self.ops_map["block4_pool"])

        self.ops_map["head_4_scaling"] = tf.Variable(0.1, dtype=tf.float32)
        self.ops_map["scaled_block_4_head"] = self.ops_map["head_4_scaling"] * self.ops_map["block_4_head"]

        head_4_and_5 = self.ops_map["cropped_block5_upscale"] + self.ops_map["scaled_block_4_head"]

        head_4_and_5 = tf.keras.layers.Conv2DTranspose(
            filters=256, kernel_size=(4, 4), strides=(2, 2), activation=tf.nn.swish)(head_4_and_5)

        self.ops_map["head_4_and_5"] = tf.keras.layers.Cropping2D(cropping=((1, 1), (1, 1)))(head_4_and_5)

        self.ops_map["head_3"] = self.ops_map["block3_pool"]

        self.ops_map["head_3_scaling"] = tf.Variable(0.01, dtype=tf.float32)
        self.ops_map["scaled_head_3"] = self.ops_map["head_3_scaling"] * self.ops_map["head_3"]

        full_head = self.ops_map["head_4_and_5"] + self.ops_map["scaled_head_3"]

        full_head = tf.keras.layers.Conv2DTranspose(
            filters=256, kernel_size=(4, 4), strides=(4, 4), activation=tf.nn.swish)(full_head)

        full_head = tf.keras.layers.Conv2DTranspose(
            filters=categories_count, kernel_size=(4, 4), strides=(2, 2))(full_head)

        self.logits = tf.keras.layers.Cropping2D(cropping=((1, 1), (1, 1)))(full_head)
        self.predictions = tf.nn.softmax(self.logits, axis=-1)


class Model:
    """
    Model wraps up a network, providing training and evaluation functions
    """

    def __init__(self, session, network, categories):

        self.session = session
        self.network = network

        self.labels_placeholder = tf.placeholder(dtype=np.float32, shape=[1, None, None, len(categories)])

        self.loss_op = tf.losses.softmax_cross_entropy(
            onehot_labels=self.labels_placeholder, logits=self.network.logits)

        self.train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.loss_op)

    def train(self, generator, data_size, indices_to_colors_map, configuration):

        for epoch_index in range(configuration["epochs"]):

            losses = []

            # head_4_scaling, head_3_scaling = self.session.run(
            #     [self.network.ops_map["head_4_scaling"], self.network.ops_map["head_3_scaling"]])

            # print("Scales: {}, {}".format(head_4_scaling, head_3_scaling))

            for step in tqdm.tqdm(range(data_size)):
            # for step in tqdm.tqdm(range(100)):

                image, segmentation = next(generator)

                segmentation_cube = net.voc.get_segmentation_cube(segmentation, indices_to_colors_map)

                feed_dictionary = {
                    self.network.input_placeholder: np.array([image]),
                    self.labels_placeholder: np.array([segmentation_cube])
                }

                _, loss = self.session.run([self.train_op, self.loss_op], feed_dictionary)

                losses.append(loss)

            print("Loss: {}".format(np.mean(losses)))


