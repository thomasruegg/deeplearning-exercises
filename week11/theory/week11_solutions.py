import numpy as np
from numpy.polynomial.polynomial import polyval
import matplotlib.pyplot as plt
from matplotlib import cm


N = 10
w_init = [4, 4]
epsilon = 0.05
alpha = 0.5

# loss function
W1 = np.arange(-7, 5, 0.1)
W2 = np.arange(-7, 5, 0.1)
W1, W2 = np.meshgrid(W1, W2)
F = (
    1 / 4 * (W1**4)
    + (W1**3)
    - 17 / 4 * (W1**2)
    - 6 * W1
    + 1 / 5 * (W2**4)
    + 6 / 5 * (W2**3)
    + 89
)


def gradient(w):
    return np.array(
        [
            polyval(w[0], [-6, -17 / 2, 3, 1]),
            polyval(w[1], [0, 0, 18 / 5, 4 / 5]),
        ]
    )


# gradient descent
w_gradient_descent = np.zeros([2, N + 1])
w_gradient_descent[:, 0] = w_init
print(f"Gradient descent with {epsilon=}:")
print(f"w(0) = {w_gradient_descent[:, 0]}")
for i in range(N):
    w_gradient_descent[:, i + 1] = w_gradient_descent[:, i] - epsilon * gradient(w_gradient_descent[:, i])
    print(f"w({i + 1}) = {w_gradient_descent[:, i + 1]}")

# momentum
v = np.zeros(2)
w_momentum = np.zeros([2, N + 1])
w_momentum[:, 0] = w_init
print()
print(f"Momentum with {alpha=} and {epsilon=}:")
print(f"w(0) = {w_momentum[:, 0]}")
for i in range(N):
    v = alpha * v - epsilon * gradient(w_momentum[:, i])
    w_momentum[:, i + 1] = w_momentum[:, i] + v
    print(f"w({i + 1}) = {w_momentum[:, i + 1]}")

# plot results
plt.figure(figsize=(13, 10))
plt.contour(W1, W2, F, 100, cmap=cm.coolwarm)
plt.colorbar(shrink=0.5, aspect=5)
plt.plot(w_gradient_descent[0, :], w_gradient_descent[1, :], "ro-", label="Gradient Descent")
plt.plot(w_momentum[0, :], w_momentum[1, :], "bo-", label="Momentum")
plt.axis("equal")
plt.axis("tight")
plt.xlabel("w1")
plt.ylabel("w2")
plt.legend(loc="lower center", ncol=3, mode="expand", shadow=True, fancybox=True)
plt.show()
