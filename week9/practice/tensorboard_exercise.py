import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Softmax
from tensorflow.keras import Model


# Load and prepare data
#############################################################################

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
indices = np.random.choice(len(y_train), 5000, replace=False)
x_valid = x_train[indices, :]
y_valid = y_train[indices]
x_train = np.delete(x_train, indices, axis=0)
y_train = np.delete(y_train, indices, axis=0)

# Convert labels to one-hot tensor
y_train = tf.one_hot(y_train, 10)
y_test = tf.one_hot(y_test, 10)
y_valid = tf.one_hot(y_valid, 10)

# Create datasets
BATCH_SIZE = 32
train_ds = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train))
    .shuffle(len(y_train), reshuffle_each_iteration=True)
    .batch(BATCH_SIZE)
)
valid_ds = tf.data.Dataset.from_tensor_slices((x_valid, y_valid)).batch(BATCH_SIZE)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE)


# Create model, define optimizer, loss and metrics
#############################################################################


class MyModel(Model):
    def __init__(self, name=None, **kwargs):
        super(MyModel, self).__init__(name=name, **kwargs)
        self.d1 = Dense(10, use_bias=True, name="Dense_1")
        self.s1 = Softmax(name="Softmax_1")

    @tf.function
    def call(self, x, training=False):
        x = self.d1(x)
        out = self.s1(x)
        return out


# Create an instance of the model
model = MyModel(name="MNISTClassifier")

# Choose an optimizer and loss function for training
loss_object = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

# Select metrics to measure the loss and the accuracy of the model.
# These metrics accumulate the values over epochs and then print the overall result.
train_loss = tf.keras.metrics.Mean(name="train_loss")
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name="train_accuracy")

valid_loss = tf.keras.metrics.Mean(name="valid_loss")
valid_accuracy = tf.keras.metrics.CategoricalAccuracy(name="valid_accuracy")

test_loss = tf.keras.metrics.Mean(name="test_loss")
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name="test_accuracy")

# Define steps
#############################################################################


# Use tf.GradientTape to train the model
@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)


# Validate the model
@tf.function
def valid_step(images, labels):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(images, training=False)
    loss = loss_object(labels, predictions)

    valid_loss(loss)
    valid_accuracy(labels, predictions)

    mislabeled = tf.not_equal(tf.argmax(predictions, axis=1), tf.argmax(labels, axis=1))
    mislabeled_images = tf.boolean_mask(images, mislabeled)

    return mislabeled_images


# Test the model
@tf.function
def test_step(images, labels):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(images, training=False)
    loss = loss_object(labels, predictions)

    test_loss(loss)
    test_accuracy(labels, predictions)


# Prepare TensorBoard
#############################################################################

# TODO: Visualize the following aspects in TensorBoard:
# Scalar: Train loss, train accuracy, validation loss, validation accuracy
scalar_log_dir = "logs/scalars/"
scalar_writer = tf.summary.create_file_writer(scalar_log_dir)

# Graph: Graph illustration of your model
graph_log_dir = "logs/graph/"
graph_writer = tf.summary.create_file_writer(graph_log_dir)

# Histogram: Network weights, network bias
histogram_log_dir = "logs/histograms/"
histogram_writer = tf.summary.create_file_writer(histogram_log_dir)

# Images: 12 misclassified validation images
image_log_dir = "logs/images/"
image_writer = tf.summary.create_file_writer(image_log_dir)


# Run training
#############################################################################

EPOCHS = 25
for epoch in range(EPOCHS):
    # Reset the metrics at the start of the next epoch
    train_loss.reset_state()
    train_accuracy.reset_state()
    valid_loss.reset_state()
    valid_accuracy.reset_state()

    # Train
    for images, labels in train_ds:
        train_step(images, labels)

    # Validate
    mislabeled_images_list = []
    for valid_images, valid_labels in valid_ds:
        mislabeled_images = valid_step(valid_images, valid_labels)
        mislabeled_images_list.append(mislabeled_images)

    # TODO: Write scalars to TensorBoard
    with scalar_writer.as_default():
        tf.summary.scalar("train_loss", train_loss.result(), step=epoch)
        tf.summary.scalar("train_accuracy", train_accuracy.result(), step=epoch)
        tf.summary.scalar("valid_loss", valid_loss.result(), step=epoch)
        tf.summary.scalar("valid_accuracy", valid_accuracy.result(), step=epoch)

    # TODO: Write histograms to TensorBoard
    with histogram_writer.as_default():
        for layer in model.layers:
            for weight in layer.weights:
                tf.summary.histogram(weight.name, weight, step=epoch)

    # TODO: Write images to TensorBoard
    with image_writer.as_default():
        for i, mislabeled_images in enumerate(mislabeled_images_list):
            if i < 12:
                tf.summary.image(
                    "mislabeled_images_{}".format(i),
                    tf.reshape(mislabeled_images, [-1, 28, 28, 1]),
                    step=epoch,
                )

    print(
        "Epoch {:2d}: ".format(epoch + 1),
        "Train Loss: {:3.3f}, ".format(train_loss.result()),
        "Train Accuracy: {:3.3f}%, ".format(train_accuracy.result() * 100),
        "Validation Loss: {:3.3f}, ".format(valid_loss.result()),
        "Validation Accuracy: {:3.3f}%".format(valid_accuracy.result() * 100),
    )

# Test resulting classifier
test_loss.reset_state()
test_accuracy.reset_state()
for test_images, test_labels in test_ds:
    test_step(test_images, test_labels)

print(
    "\nTesting result: ",
    "Test Loss: {:3.3f}, ".format(test_loss.result()),
    "Test Accuracy: {:3.3f}%".format(test_accuracy.result() * 100),
)