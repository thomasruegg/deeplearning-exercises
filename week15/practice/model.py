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
        self.conv_1 = Conv2D(filters=32, kernel_size=5, padding='same')
        self.relu_1 = ReLU()
        self.pool_1 = MaxPool2D(pool_size=(2, 2))

        self.conv_2 = Conv2D(filters=64, kernel_size=5, padding='same')
        self.relu_2 = ReLU()
        self.pool_2 = MaxPool2D(pool_size=(2, 2))

        self.flat_1 = Flatten()

        self.drop_1 = Dropout(0.5)
        self.dense_1 = Dense(1024, activation='relu')
        self.drop_2 = Dropout(0.5)
        self.dense_2 = Dense(10, activation='relu')
        self.softmax_1 = Softmax()


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
        x = self.conv_1(x)
        x = self.relu_1(x)
        x = self.pool_1(x)
        x = self.conv_2(x)
        x = self.relu_2(x)
        x = self.pool_2(x)
        x = self.flat_1(x)
        x = self.drop_1(x, training=training)
        x = self.dense_1(x)
        x = self.drop_2(x, training=training)
        x = self.dense_2(x)
        x = self.softmax_1(x)

        return x
