import tensorflow as tf
import tensorflow_datasets as tfds

tfds.disable_progress_bar()


class DataLoaderIMDB:
    """
    This class downloads and prepares the IMDB large movie dataset.
    It splits the dataset into train, valid, test, and unsupervised subsets.
    """

    def __init__(self):
        MINI_BATCH_SIZE = 64
        VALID_SPLIT = 0.1

        dataset, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)

        valid_size = int(info.splits["train"].num_examples * VALID_SPLIT)

        self._valid_dataset = dataset["train"].take(valid_size)
        self._train_dataset = dataset["train"].skip(valid_size)
        self._test_dataset = dataset["test"]
        self._unsupervised_dataset = dataset["unsupervised"]

        # example = next(self._train_dataset.as_numpy_iterator())
        # print("text:", example[0])
        # print("label:", example[1])

        self._train_dataset = self._train_dataset.shuffle(10000)

        # Batch
        self._train_dataset = self._train_dataset.batch(MINI_BATCH_SIZE)
        self._valid_dataset = self._valid_dataset.batch(MINI_BATCH_SIZE)
        self._test_dataset = self._test_dataset.batch(MINI_BATCH_SIZE)
        self._unsupervised_dataset = self._unsupervised_dataset.batch(MINI_BATCH_SIZE)

        # Prefetch
        self._train_dataset = self._train_dataset.prefetch(tf.data.AUTOTUNE)
        self._valid_dataset = self._valid_dataset.prefetch(tf.data.AUTOTUNE)
        self._test_dataset = self._test_dataset.prefetch(tf.data.AUTOTUNE)
        self._unsupervised_dataset = self._unsupervised_dataset.prefetch(
            tf.data.AUTOTUNE
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

    @property
    def unsupervised_dataset(self):
        return self._unsupervised_dataset


if __name__ == "__main__":
    data_loader = DataLoaderIMDB()
