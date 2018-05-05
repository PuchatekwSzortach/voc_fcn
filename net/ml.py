"""
Module with machine learning code
"""

import tensorflow as tf


class FCNModel:

    def __init__(self):

        x = tf.keras.applications.VGG16(include_top=True)

        self.input = x.input
        self.output = x.output
