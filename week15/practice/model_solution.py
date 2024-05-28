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

        self.conv_1 = Conv2D(filters=32, kernel_size=5, padding="same")
        self.conv_2 = Conv2D(filters=64, kernel_size=5, padding="same")

        self.relu_1 = ReLU()
        self.relu_2 = ReLU()
        self.relu_3 = ReLU()

        self.pool_1 = MaxPool2D(pool_size=(2, 2))
        self.pool_2 = MaxPool2D(pool_size=(2, 2))

        self.flat_1 = Flatten()

        self.drop_1 = Dropout(0.5)
        self.drop_2 = Dropout(0.5)

        self.dense_1 = Dense(1024)
        self.dense_2 = Dense(10)

        self.softm_1 = Softmax()

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

        # First convolutional stage
        t_conv_1 = self.conv_1(x)
        t_relu_1 = self.relu_1(t_conv_1)
        t_pool_1 = self.pool_1(t_relu_1)

        # Second convolutional stage
        t_conv_2 = self.conv_2(t_pool_1)
        t_relu_2 = self.relu_2(t_conv_2)
        t_pool_2 = self.pool_2(t_relu_2)

        # Flatten
        t_flat_1 = self.flat_1(t_pool_2)

        # First dense stage
        t_drop_1 = self.drop_1(t_flat_1, training)
        t_dens_1 = self.dense_1(t_drop_1)
        t_relu_3 = self.relu_3(t_dens_1)

        # Second dense stage
        t_drop_2 = self.drop_2(t_relu_3, training)
        t_dens_2 = self.dense_2(t_drop_2)
        out = self.softm_1(t_dens_2)

        return out
