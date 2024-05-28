from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Dense,
    Softmax,
    Conv2D,
    MaxPool2D,
    ReLU,
    Flatten,
    Dropout,
)


class MyModel(Model):
    """
    TensorFlow convolutional neural network which is used as classifier for
    MNIST images.

    Parameters
    ----------
    name : string, optional
        Name of model. The default is None.
    **kwargs :
        See description of tf.keras.Model.

    """

    def __init__(self, name=None, **kwargs):
        super(MyModel, self).__init__(name=name, **kwargs)

        # TODO: Initialize layers

    def call(self, x, training=False):
        """
        Forward pass of MyModel with specific input x.

        Parameters
        ----------
        x : Tensor float32 (None, 28, 28, 1)
            Input to MyModel.
        training : bool, optional
            training=True is only needed if there are layers with different
            behavior during training versus inference (e.g. Dropout).
            The default is False.

        Returns
        -------
        out : tensor float32 (None, 10)
            Output of MyModel.

        """

        # TODO: Call layers
        pass
