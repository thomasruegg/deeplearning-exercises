import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


def noisy_data(m=0.3, b=.3, n=100):
    x = tf.random.uniform(shape=(n,))
    some_noise = tf.random.normal(shape=(len(x),), stddev=0.01)
    y = m * x + b + some_noise
    return x, y


x_train, y_train = noisy_data()

plt.plot(x_train, y_train, 'b.')
plt.show()


m = tf.Variable(0.)
b = tf.Variable(0.)

@tf.function
def predict_response(x):
    return m * x + b

@tf.function
def squared_error(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_pred - y_true))

# compute loss prior to training
y_pred = predict_response(x_train)
loss = squared_error(y_pred, y_train)
print(f"Loss prior to training: {loss.numpy():.6f}")


learning_rate = 0.05
steps = 201

for i in range(steps):
    with tf.GradientTape() as tape:
        y_pred = predict_response(x_train)
        loss = squared_error(y_pred, y_train)

    gradient = tape.gradient(loss, [m, b])

    m.assign_sub(gradient[0] * learning_rate)
    b.assign_sub(gradient[1] * learning_rate)

    if i % 50 == 0:
        print(f"Step number {i}: loss = {loss.numpy():.6f}")

print(f'\nGradient after {steps} steps:', gradient)