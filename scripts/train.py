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

    # generator = net.voc.BatchesGeneratorFactory(config["data_directory"]).get_generator(batch_size=1)

    model = net.ml.FCNModel()

    paths = glob.glob("./data/*.jpg")
    print(*paths, sep="\n")

    images = np.array([cv2.imread(path) for path in paths])
    print(images.shape)

    session = tf.keras.backend.get_session()

    feed_dictionary = {
        model.input: images
    }

    predictions = session.run(model.output, feed_dictionary)

    for path, prediction in zip(paths, predictions):

        prediction_index = np.argmax(prediction)
        category = net.imagenet.mapping[prediction_index]

        print("{} -> {}".format(os.path.basename(path), category))


if __name__ == "__main__":
    main()
