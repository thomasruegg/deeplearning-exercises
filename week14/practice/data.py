import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds

tfds.disable_progress_bar()


class DataLoaderRPS:
    """
    This class downloads and prepares the rock-paper-scissors dataset.
    It splits the dataset into train, valid and test subsets.
    """

    def __init__(self):
        MINI_BATCH_SIZE = 64
        VALID_SPLIT = 0.33

        # Augmentation layer
        data_augmentation = tf.keras.Sequential(
            [
                MyRandomRotation(),
                tf.keras.layers.RandomZoom((-0.1, 0.1)),
            ]
        )

        dataset = tfds.load("rock_paper_scissors", as_supervised=True)

        # Split off valid dataset from train dataset
        self._train_dataset, self._valid_dataset = tf.keras.utils.split_dataset(
            dataset["train"], right_size=VALID_SPLIT, shuffle=True
        )

        self._test_dataset = dataset["test"]

        self._train_dataset = self._train_dataset.shuffle(10000)

        # Batch
        self._train_dataset = self._train_dataset.batch(MINI_BATCH_SIZE)
        self._valid_dataset = self._valid_dataset.batch(MINI_BATCH_SIZE)
        self._test_dataset = self._test_dataset.batch(MINI_BATCH_SIZE)

        # Augment
        self._train_dataset = self._train_dataset.map(
            lambda x, y: (data_augmentation(x), y), num_parallel_calls=tf.data.AUTOTUNE
        )
        self._valid_dataset = self._valid_dataset.map(
            lambda x, y: (data_augmentation(x), y), num_parallel_calls=tf.data.AUTOTUNE
        )

        # Prefetch
        self._train_dataset = self._train_dataset.prefetch(tf.data.AUTOTUNE)
        self._valid_dataset = self._valid_dataset.prefetch(tf.data.AUTOTUNE)
        self._test_dataset = self._test_dataset.prefetch(tf.data.AUTOTUNE)

    @property
    def train_dataset(self):
        return self._train_dataset

    @property
    def valid_dataset(self):
        return self._valid_dataset

    @property
    def test_dataset(self):
        return self._test_dataset

    def plot_samples(self):
        samples_count = 16
        images, labels = next(self.valid_dataset.as_numpy_iterator())
        if len(images) < samples_count:
            samples_count = len(images)

        LEGEND = ["Rock", "Paper", "Scissors"]
        fig = plt.figure()
        for i in range(samples_count):
            ax = fig.add_subplot(4, 4, i + 1)
            ax.imshow(images[i, ...] / 255.0)
            ax.set_title(LEGEND[labels[i]])
            ax.axis("off")
        plt.tight_layout()
        plt.show()


class MyRandomRotation(tf.keras.layers.Layer):
    def __init__(self):
        super(MyRandomRotation, self).__init__()

    def call(self, input):
        rnd_num = tf.random.uniform(shape=(), minval=0, maxval=4, dtype=tf.int32)
        out = tf.image.rot90(input, rnd_num)
        return out


if __name__ == "__main__":
    data_loader = DataLoaderRPS()
    data_loader.plot_samples()
