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


class EarlyStopping(Callback):
    """
    Callback that instructs model to stop training if validation loss didn't improve for too many epochs
    """

    def __init__(self, patience, verbose=True):
        """
        Constructor
        :param patience: int, number of epochs that have to pass without validation loss
        improvement before callback will ask model to stop training
        :param verbose: bool, sets callback's verbosity
        """

        super().__init__()

        self.patience = patience
        self.best_validation_loss = np.inf
        self.verbose = verbose

        self.epochs_since_last_improvement = 0

    def on_epoch_end(self, epoch_log):

        if epoch_log["validation_loss"] < 0.999 * self.best_validation_loss:

            self.best_validation_loss = epoch_log["validation_loss"]
            self.epochs_since_last_improvement = 0

        else:

            self.epochs_since_last_improvement += 1

        if self.epochs_since_last_improvement > self.patience:

            self.model.should_continue_training = False

            if self.verbose is True:
                print("Early stopping!!!")


class ReduceLearningRateOnPlateau(Callback):
    """
    Callback that reduces learning rate if validation loss doesn't improve for too many epochs
    """

    def __init__(self, patience, factor, verbose=True):
        """
        Constructor
        :param patience: int, number of epochs that have to pass without validation loss
        improvement before callback will ask model to adjust learning rate
        :param factor: float, factor by which learning rate should be changed
        :param verbose: bool, sets callback's verbosity
        """

        super().__init__()

        self.patience = patience
        self.factor = factor
        self.verbose = verbose

        self.best_validation_loss = np.inf
        self.epoch_before_patience_runs_out = 0

    def on_epoch_end(self, epoch_log):

        if epoch_log["validation_loss"] < 0.999 * self.best_validation_loss:

            self.best_validation_loss = epoch_log["validation_loss"]
            self.epoch_before_patience_runs_out = 0

        else:

            self.epoch_before_patience_runs_out += 1

        if self.epoch_before_patience_runs_out > self.patience:

            self.model.learning_rate *= self.factor
            self.epoch_before_patience_runs_out = 0

            if self.verbose is True:
                print("Changed learning rate to {}".format(self.model.learning_rate))
