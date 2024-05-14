from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Softmax, ReLU


class MyModel(Model):
    """
    TensorFlow neural network which is used as classifier for flattened
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

        self.d1 = Dense(10, use_bias=True, kernel_regularizer='l2')

        self.r1 = ReLU()
        self.d2 = Dense(100, use_bias=True, kernel_regularizer='l2')
        self.r2 = ReLU()
        self.d3 = Dense(100, use_bias=True, kernel_regularizer='l2')
        self.r3 = ReLU()
        self.d4 = Dense(100, use_bias=True, kernel_regularizer='l2')
        self.r4 = ReLU()
        self.d5 = Dense(10, use_bias=True)

        self.s1 = Softmax(name="Softmax_1")

    def call(self, x, training=False):
        """
        Forward pass of MyModel with specific input x.

        Parameters
        ----------
        x : Tensor float32 (None, 784)
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
        x = self.d1(x)
        x = self.r1(x)
        x = self.d2(x)
        x = self.r2(x)
        x = self.d3(x)
        x = self.r3(x)
        x = self.d4(x)
        x = self.r4(x)
        x = self.d5(x)
        x = self.s1(x)
        return x
