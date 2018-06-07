""""
Callbacks definitions. Largely based on Keras callbacks
"""

import numpy as np


class Callback:
    """
    Base class for model callbacks
    """

    def __init__(self):
        """
        Constructor
        """

        self.model = None

    def on_epoch_end(self, epoch_log):
        """
        Callback called on epoch end
        :param epoch_log: dictionary with epoch data
        """

        raise NotImplementedError()


class ModelCheckpoint(Callback):
    """
    Callback for saving network parameters.
    Only saves network if its validation loss improved by at least 0.1% over previous result
    """

    def __init__(self, save_path, verbose=True):
        """
        Constructor
        :param save_path: prefix for filenames created for the checkpoint
        :param verbose: bool, sets callback's verbosity
        """

        super().__init__()

        self.save_path = save_path
        self.best_validation_loss = np.inf
        self.verbose = verbose

    def on_epoch_end(self, epoch_log):

        if epoch_log["validation_loss"] < 0.999 * self.best_validation_loss:

            if self.verbose:
                print("Validation loss improved from {} to {}, saving model".format(
                    self.best_validation_loss, epoch_log["validation_loss"]))

            self.best_validation_loss = epoch_log["validation_loss"]
            self.model.save(self.save_path)
