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

        IMAGE_SHAPE = (300, 300, 3)

        self.preprocess_layer = preprocess_input

        self.backbone = tf.keras.applications.ResNet50(
            include_top=False, weights="imagenet", input_shape=IMAGE_SHAPE
        )
        self.backbone.trainable = False

        self.avg_pool_1 = GlobalAveragePooling2D()

        self.drop_1 = Dropout(0.5)
        self.drop_2 = Dropout(0.5)
        self.drop_3 = Dropout(0.5)

        self.dense_1 = Dense(1024, activation="relu")
        self.dense_2 = Dense(64, activation="relu")
        self.dense_3 = Dense(3)

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
        x = self.preprocess_layer(x)
        x = self.backbone(x)
        x = self.avg_pool_1(x)

        x = self.drop_1(x, training=training)
        x = self.dense_1(x)

        x = self.drop_2(x, training=training)
        x = self.dense_2(x)

        x = self.drop_3(x, training=training)
        out = self.dense_3(x)

        return out


if __name__ == "__main__":
    model = MyModel()
    features = model(tf.ones([32, 300, 300, 3]))
    model.summary()
