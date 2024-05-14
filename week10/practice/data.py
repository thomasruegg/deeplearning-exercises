import numpy as np
import tensorflow as tf

class DataLoaderMNIST:
    """
    Prepare TensorFlow iterator of MNIST dataset and split data into train,
    valid, and test subsets.

    """

    def __init__(self):
        VALIDATION_DATASET_SIZE = 5000
        MINI_BATCH_SIZE = 32

        # Load MNIST data
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        # Scale images
        x_train = x_train / 255.0
        x_test = x_test / 255.0

        # Flatten images
        x_train = x_train.reshape([len(x_train), -1]).astype("float32")
        x_test = x_test.reshape([len(x_test), -1]).astype("float32")

        # Split off validation dataset from training dataset
        indices = np.random.choice(len(y_train), VALIDATION_DATASET_SIZE, replace=False)
        x_valid = x_train[indices, :]
        y_valid = y_train[indices]
        x_train = np.delete(x_train, indices, axis=0)
        y_train = np.delete(y_train, indices, axis=0)

        # Convert labels to one-hot tensor
        y_train = tf.one_hot(y_train, 10)
        y_test = tf.one_hot(y_test, 10)
        y_valid = tf.one_hot(y_valid, 10)

        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.2),
        ])

        # Create datasets
        self._train_dataset = (
            tf.data.Dataset.from_tensor_slices((x_train, y_train))
            .shuffle(len(y_train))
            .batch(MINI_BATCH_SIZE)
        )
        self._valid_dataset = tf.data.Dataset.from_tensor_slices(
            (x_valid, y_valid)
        ).batch(MINI_BATCH_SIZE)
        self._test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(
            MINI_BATCH_SIZE
        )

    @property
    def train_dataset(self):
        return self._train_dataset

    @property
    def valid_dataset(self):
        return self._valid_dataset

    @property
    def test_dataset(self):
        return self._test_dataset
