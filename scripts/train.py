import argparse
import sys
import glob
import os

import yaml
import tensorflow as tf
import cv2
import numpy as np

import net.voc
import net.ml
import net.imagenet


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--config', action="store", required=True)
    arguments = parser.parse_args(sys.argv[1:])

    with open(arguments.config) as file:
        config = yaml.safe_load(file)

    categories = config["categories"]

    generator = net.voc.BatchesGeneratorFactory(config["data_directory"]).get_generator(
        size_factor=config["size_factor"], batch_size=config["batch_size"])

    model = net.ml.FCNModel(categories_count=len(categories))

    session = tf.keras.backend.get_session()

    for _ in range(2):

        # images, segmentations = next(generator)
        images = np.random.randint(0, 255, size=(1, 320, 480, 3))

        feed_dictionary = {model.input: images}
        prediction = session.run(model.output, feed_dictionary)
        print(images.shape)
        print(prediction.shape)
        print()


if __name__ == "__main__":
    main()
