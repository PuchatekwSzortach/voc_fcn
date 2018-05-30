""""
Callbacks definitions. Largely based on Keras callbacks
"""


class Callback:
    """
    Base class for model callbacks
    """

    def __init__(self):
        """
        Constructor
        """

        print("I was called :)")

        self.model = None

    def on_epoch_end(self, epoch_log):
        """
        Callback called on epoch end
        :param epoch_log: dictionary with epoch data
        """

        raise NotImplementedError()
