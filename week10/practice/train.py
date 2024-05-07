import tensorflow as tf
import os

from early_stopping import EarlyStoppingCounter


class Training:
    """
    Training class, for model, logging metrics in tensorboard_dir.

    Parameters
    ----------
    model : tf.keras.Model
        TensorFlow model to be trained.
    tensorboard_dir : string
        Path to tensorboard directory.

    """

    def __init__(self, model, tensorboard_dir):
        self._model = model
        self._optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
        self._loss = tf.keras.losses.CategoricalCrossentropy()
        self._init_metrics()
        self._init_tensorboard(tensorboard_dir)
        self._early_stopping_counter = EarlyStoppingCounter()

    def __call__(
        self,
        train_dataset,
        valid_dataset,
        test_dataset,
        max_epochs=50,
        evaluate_every=5,
    ):
        """
        This method implements the training loop using train_dataset and
        evaluetes the valid_dataset every evaluate_every epochs. At the end
        it tests the classifiers performance based on test_dataset.

        Parameters
        ----------
        train_dataset : TF Dataset
            Training dataset which consists of pairs of input and labels.
        valid_dataset : TF Dataset
            Validation dataset which consists of pairs of input and labels.
        test_dataset : TF Dataset
            Test dataset which consists of pairs of input and labels.
        max_epochs : int, optional
            Maximum number of times train_dataset is iterated over.
            The default is 50.
        evaluate_every : int, optional
            Every evaluate_every'th epoch the model is evaluated based on
            valid_dataset. The default is 5.

        Returns
        -------
        None.

        """
        for epoch in range(1, max_epochs + 1):
            self._reset_train_metrics()
            for images, labels in train_dataset:
                self._train_step(images, labels)

            if epoch % evaluate_every == 0:
                self._reset_valid_metrics()
                for images, labels in valid_dataset:
                    self._valid_step(images, labels)

                self._print_train_and_valid_results(epoch, max_epochs)
                self._update_tensorboard(epoch)

                self._early_stopping_counter.update(self._valid_accuracy.result())
                if self._early_stopping_counter.is_stopping_criteria_reached():
                    print("Early stopping at epoch {:2d}.".format(epoch))
                    self._early_stopping_counter.reset()
                    break

        self._reset_test_metrics()
        for images, labels in test_dataset:
            self._test_step(images, labels)

        self._print_test_results()

    @tf.function
    def _train_step(self, images, labels):
        with tf.GradientTape() as tape:
            predictions = self._model(images, training=True)
            loss = self._loss(labels, predictions)

        gradients = tape.gradient(loss, self._model.trainable_variables)
        self._optimizer.apply_gradients(zip(gradients, self._model.trainable_variables))
        self._update_train_metrics(loss, labels, predictions)

    @tf.function
    def _valid_step(self, images, labels):
        predictions = self._model(images, training=False)
        loss = self._loss(labels, predictions)
        self._update_valid_metrics(loss, labels, predictions)

    @tf.function
    def _test_step(self, images, labels):
        predictions = self._model(images, training=False)
        loss = self._loss(labels, predictions)
        self._update_test_metrics(loss, labels, predictions)

    def _init_metrics(self):
        self._init_train_metrics()
        self._init_valid_metrics()
        self._init_test_metrics()

    def _init_train_metrics(self):
        self._train_loss = tf.keras.metrics.Mean(name="train_loss")
        self._train_accuracy = tf.keras.metrics.CategoricalAccuracy(
            name="train_accuracy"
        )

    def _init_valid_metrics(self):
        self._valid_loss = tf.keras.metrics.Mean(name="valid_loss")
        self._valid_accuracy = tf.keras.metrics.CategoricalAccuracy(
            name="valid_accuracy"
        )

    def _init_test_metrics(self):
        self._test_loss = tf.keras.metrics.Mean(name="test_loss")
        self._test_accuracy = tf.keras.metrics.CategoricalAccuracy(name="test_accuracy")

    def _reset_train_metrics(self):
        self._train_loss.reset_state()
        self._train_accuracy.reset_state()

    def _reset_valid_metrics(self):
        self._valid_loss.reset_state()
        self._valid_accuracy.reset_state()

    def _reset_test_metrics(self):
        self._test_loss.reset_state()
        self._test_accuracy.reset_state()

    def _update_train_metrics(self, loss, labels, predictions):
        self._train_loss(loss)
        self._train_accuracy(labels, predictions)

    def _update_valid_metrics(self, loss, labels, predictions):
        self._valid_loss(loss)
        self._valid_accuracy(labels, predictions)

    def _update_test_metrics(self, loss, labels, predictions):
        self._test_loss(loss)
        self._test_accuracy(labels, predictions)

    def _print_train_and_valid_results(self, epoch, max_epochs):
        print(
            "Epoch {:3d} of {:3d}, ".format(epoch, max_epochs),
            "Train Loss: {:3.3f}, ".format(self._train_loss.result()),
            "Train Accuracy: {:3.3f}%, ".format(self._train_accuracy.result() * 100),
            "Valid Loss: {:3.3f}, ".format(self._valid_loss.result()),
            "Valid Accuracy: {:3.3f}%".format(self._valid_accuracy.result() * 100),
        )

    def _print_test_results(self):
        print(
            "\nTest Loss: {:3.3f}, ".format(self._test_loss.result()),
            "Test Accuracy: {:3.3f}%".format(self._test_accuracy.result() * 100),
        )

    def _init_tensorboard(self, tensorboard_dir):
        self._train_summary_writer = tf.summary.create_file_writer(
            os.path.join(tensorboard_dir, "train")
        )
        self._valid_summary_writer = tf.summary.create_file_writer(
            os.path.join(tensorboard_dir, "valid")
        )

    def _update_tensorboard(self, epoch):
        with self._train_summary_writer.as_default():
            tf.summary.scalar("Train loss", self._train_loss.result(), step=epoch)
            tf.summary.scalar(
                "Train accuracy", self._train_accuracy.result(), step=epoch
            )

        with self._valid_summary_writer.as_default():
            tf.summary.scalar("Valid loss", self._valid_loss.result(), step=epoch)
            tf.summary.scalar(
                "Valid accuracy", self._valid_accuracy.result(), step=epoch
            )
