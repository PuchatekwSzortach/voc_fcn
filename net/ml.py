"""
Module with machine learning code
"""

import tensorflow as tf


class FCNModel:

    def __init__(self, categories_count):

        x = tf.keras.applications.VGG16(include_top=False)
        self.input = x.input

        net = {
            "block3_pool": x.get_layer("block3_pool").output,
            "block4_pool": x.get_layer("block4_pool").output,
            "block5_pool": x.get_layer("block5_pool").output,
        }

        block5_upscale = tf.keras.layers.Conv2DTranspose(
            filters=256, kernel_size=(4, 4), strides=(2, 2), activation=tf.nn.swish)(net["block5_pool"])

        cropped_block5_upscale = tf.keras.layers.Cropping2D(cropping=((1, 1), (1, 1)))(block5_upscale)

        block_4_head = tf.keras.layers.Conv2D(
            filters=256, kernel_size=(3, 3), activation=tf.nn.swish, padding='same')(net["block4_pool"])

        block_4_5_scaling = tf.Variable(tf.constant(1, dtype=tf.float32))
        head_4_and_5 = cropped_block5_upscale + (block_4_5_scaling * block_4_head)

        upscaled_head_4_and_5 = tf.keras.layers.Conv2DTranspose(
            filters=256, kernel_size=(4, 4), strides=(2, 2), activation=tf.nn.swish)(head_4_and_5)

        cropped_upscaled_head_4_and_5 = tf.keras.layers.Cropping2D(cropping=((1, 1), (1, 1)))(upscaled_head_4_and_5)

        full_head_scaling = tf.Variable(tf.constant(1, dtype=tf.float32))
        full_head = cropped_upscaled_head_4_and_5 + (full_head_scaling * net["block3_pool"])

        full_head = tf.keras.layers.Conv2DTranspose(
            filters=256, kernel_size=(4, 4), strides=(4, 4), activation=tf.nn.swish)(full_head)

        full_head = tf.keras.layers.Conv2DTranspose(
            filters=categories_count, kernel_size=(4, 4), strides=(4, 4), activation=tf.nn.softmax)(full_head)

        self.output = full_head
