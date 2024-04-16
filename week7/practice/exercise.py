"""Implementation of a tiny logistic regression model using TensorFlow."""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def noisy_data(n=100):
    x = tf.random.uniform(shape=(n,))
    y = tf.convert_to_tensor(np.random.normal(x, 0.1) > 0.5, dtype=tf.float32)
    return x, y


def main():
    # create training dataset
    x_train, y_train = noisy_data()

    # visualize the data as scatter plot
    plt.scatter(x_train, y_train)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


if __name__ == "__main__":
    main()
