import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout


class MyModel(Model):
    """
    Classifier for rock-paper-scissors images. This is a transfer-learned
    neural network with a ResNet50 as backbone and three fully-connected layers
    at the end.
    """

    def __init__(self, **kwargs):
        super(MyModel, self).__init__(**kwargs)
        # TODO: Instantiate model / layers

    def call(self, x, training=False):
        """
        Forward pass through neural network. The first part is a pretrained
        ResNet50 network which was trained on ImageNet dataset.

        Parameters
        ----------
        x : Tensor (None,300,300,3)
            Input to CNN
        training : bool, optional
            Network behavior is different in training mode than it is in
            inference mode (dropout), by default False

        Returns
        -------
        Tensor (None, 1)
            Logits of classifier (pre-softmax activations)
        """
        out = x
        # TODO: Define forward pass through network
        return out


if __name__ == "__main__":
    model = MyModel()
    features = model(tf.ones([32, 300, 300, 3]))
    model.summary()
