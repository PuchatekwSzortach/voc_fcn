"""
Tests for callbacks module
"""

import unittest.mock

import net.callbacks


def test_model_checkpoint():
    """
    Test ModelCheckpoint
    """

    callback = net.callbacks.ModelCheckpoint(save_path=None, verbose=False)

    callback.model = unittest.mock.Mock()
    callback.on_epoch_end({"validation_loss": 100})

    # On first epoch we always expect save to be called, as initial loss is assumed to be infinit
    assert callback.model.save.called
    callback.on_epoch_end({"validation_loss": 100})

    # Reset mock state, call on_epoch_end with larger loss
    callback.model.reset_mock()
    callback.on_epoch_end({"validation_loss": 200})

    assert callback.model.save.called is False
    assert callback.best_validation_loss == 100

    # Reset mock state, call on_epoch_end with smaller loss
    callback.model.reset_mock()
    callback.on_epoch_end({"validation_loss": 50})

    assert callback.model.save.called is True
    assert callback.best_validation_loss == 50
