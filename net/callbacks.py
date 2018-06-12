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

    def __init__(self, save_path, skip_epochs_count, verbose=True):
        """
        Constructor
        :param save_path: prefix for filenames created for the checkpoint
        :param skip_epochs_count: number of epochs from the start during which callback won't ask model to save itself
        :param verbose: bool, sets callback's verbosity
        """

        super().__init__()

        self.save_path = save_path
        self.skip_epochs_count = skip_epochs_count
        self.best_validation_loss = np.inf
        self.verbose = verbose

    def on_epoch_end(self, epoch_log):

        has_loss_improved = epoch_log["validation_loss"] < 0.999 * self.best_validation_loss
        should_save_model = has_loss_improved and epoch_log["epoch_index"] >= self.skip_epochs_count

        # Save model if loss improved and we passed skip epochs count
        if should_save_model:

            if self.verbose:
                print("Validation loss improved from {} to {}, saving model".format(
                    self.best_validation_loss, epoch_log["validation_loss"]))

            self.model.save(self.save_path)

        # If loss improved, update our best loss
        if has_loss_improved:
            self.best_validation_loss = epoch_log["validation_loss"]


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
                print("ReduceLearningRateOnPlateau changed learning rate to {}".format(self.model.learning_rate))


class LearningRateScheduler(Callback):
    """
    Callback for setting learning rate at fixed epochs
    """

    def __init__(self, schedule, verbose):
        """
        Constructor
        :param schedule: dictionary mapping epoch indices to learning rates
        :param verbose: bool, sets callback's verbosity
        """

        super().__init__()

        self.schedule = schedule
        self.verbose = verbose

    def on_epoch_end(self, epoch_log):

        if epoch_log["epoch_index"] in self.schedule:

            self.model.learning_rate = self.schedule[epoch_log["epoch_index"]]

            if self.verbose:
                print("LearningRateScheduler changed learning rate to {}".format(self.model.learning_rate))
