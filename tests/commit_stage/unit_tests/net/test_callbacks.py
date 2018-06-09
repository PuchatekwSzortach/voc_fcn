"""
Tests for callbacks module
"""

import unittest.mock

import net.callbacks


def test_model_checkpoint():
    """
    Test ModelCheckpoint callback
    """

    callback = net.callbacks.ModelCheckpoint(save_path=None, skip_epochs_count=1, verbose=False)

    callback.model = unittest.mock.Mock()
    callback.on_epoch_end({"epoch_index": 0, "validation_loss": 100})

    # Since we are skipping over first epoch, we expect model.save wasn't called
    assert not callback.model.save.called
    callback.on_epoch_end({"epoch_index": 1, "validation_loss": 100})

    # Reset mock state, call on_epoch_end with larger loss
    callback.model.reset_mock()
    callback.on_epoch_end({"epoch_index": 2, "validation_loss": 200})

    assert callback.model.save.called is False
    assert callback.best_validation_loss == 100

    # Reset mock state, call on_epoch_end with smaller loss
    callback.model.reset_mock()
    callback.on_epoch_end({"epoch_index": 3, "validation_loss": 50})

    assert callback.model.save.called is True
    assert callback.best_validation_loss == 50


def test_early_stopping():
    """
    Tests EarlyStopping callback
    """

    callback = net.callbacks.EarlyStopping(patience=3, verbose=False)

    callback.model = unittest.mock.Mock()
    callback.model.should_continue_training = True

    # Loss improved from infinity
    callback.on_epoch_end({"validation_loss": 100})
    assert callback.model.should_continue_training is True

    # Loss didn't improve for one epoch
    callback.on_epoch_end({"validation_loss": 100})
    assert callback.model.should_continue_training is True

    # Loss didn't improve for two epochs
    callback.on_epoch_end({"validation_loss": 100})
    assert callback.model.should_continue_training is True

    # Loss didn't improve for three epochs
    callback.on_epoch_end({"validation_loss": 100})
    assert callback.model.should_continue_training is True

    # Loss improved
    callback.on_epoch_end({"validation_loss": 50})
    assert callback.model.should_continue_training is True

    # Loss didn't improve for one epoch
    callback.on_epoch_end({"validation_loss": 50})
    assert callback.model.should_continue_training is True

    # Loss didn't improve for two epochs
    callback.on_epoch_end({"validation_loss": 50})
    assert callback.model.should_continue_training is True

    # Loss didn't improve for three epochs
    callback.on_epoch_end({"validation_loss": 50})
    assert callback.model.should_continue_training is True

    # Loss didn't improve for four epochs - we went over patience period
    callback.on_epoch_end({"validation_loss": 50})
    assert callback.model.should_continue_training is False


def test_reduce_learning_rate_on_plateau():
    """
    Tests ReduceLearningRateOnPlateau callback
    """

    callback = net.callbacks.ReduceLearningRateOnPlateau(patience=1, factor=0.25, verbose=False)

    callback.model = unittest.mock.Mock()
    callback.model.learning_rate = 1

    # Loss improved from infinity
    callback.on_epoch_end({"validation_loss": 100})
    assert callback.model.learning_rate == 1

    # Loss didn't improve for one epoch
    callback.on_epoch_end({"validation_loss": 100})
    assert callback.model.learning_rate == 1

    # Loss improved
    callback.on_epoch_end({"validation_loss": 50})
    assert callback.model.learning_rate == 1

    # Loss didn't improve for one epoch
    callback.on_epoch_end({"validation_loss": 50})
    assert callback.model.learning_rate == 1

    # Loss didn't improve for two epochs, callback should change learning rate
    callback.on_epoch_end({"validation_loss": 50})
    assert callback.model.learning_rate == 0.25

    # One epoch since last change of loss
    callback.on_epoch_end({"validation_loss": 50})
    assert callback.model.learning_rate == 0.25

    # Two epochs since last change of loss - should adjust loss again
    callback.on_epoch_end({"validation_loss": 50})
    assert callback.model.learning_rate == 0.0625
