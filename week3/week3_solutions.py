import numpy as np
import numdifftools as nd
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial


def main():
    def f(x):
        return (
            1 / 4 * x[0] ** 4
            + x[0] ** 3
            - 17 / 4 * x[0] ** 2
            - 6 * x[0]
            + 1 / 5 * x[1] ** 4
            + 6 / 5 * x[1] ** 3
            + 89
        )

    # Exercise 1. (b)
    grad_x0 = Polynomial([-6, -17 / 2, 3, 1])
    grad_x1 = Polynomial([0, 0, 18 / 5, 4 / 5])
    print(f"Roots of grad_x0 = {grad_x0.roots()}")
    print(f"Roots of grad_x1 = {grad_x1.roots()}")

    # Exercise 1. (d) i.
    x0 = np.array([0, -0.5])
    eps = 0.1
    gradient_descent(f, x0, eps)

    # Exercise 1. (d) ii.
    x0 = np.array([-2.5, 4])
    eps = 0.05
    gradient_descent(f, x0, eps)

    # Exercise 1. (e) i.
    x0 = np.array([0, -0.5])
    newtons_method(f, x0)

    # Exercise 1. (e) ii.
    x0 = np.array([-2.5, 4])
    newtons_method(f, x0)


def gradient_descent(f, x0, eps):
    title = f"Gradient descent with learning rate = {eps}"
    print(f"\n{title}:")

    iterations = 10

    x = x0
    g = nd.Gradient(f)

    fig, ax = init_plot(f, x0, title)
    print_step(f, g, x0, 0)

    for i in range(iterations):
        x_new = x - eps * g(x)

        update_plot(fig, ax, x, x_new)
        print_step(f, g, x_new, i + 1)

        x = x_new


def newtons_method(f, x0):
    title = "Newton's method"
    print(f"\n{title}:")

    iterations = 10

    x = x0
    g = nd.Gradient(f)
    H = nd.Hessian(f)

    fig, ax = init_plot(f, x0, title)
    print_step(f, g, x0, 0)

    for i in range(iterations):
        x_new = x - np.linalg.inv(H(x)) @ g(x)

        update_plot(fig, ax, x, x_new)
        print_step(f, g, x_new, i + 1)

        x = x_new


def init_plot(f, x0, title):
    plt.ion()

    x1 = np.linspace(-6, 6, 200)
    x2 = np.linspace(-6, 6, 200)
    X1, X2 = np.meshgrid(x1, x2)
    Z = f(np.vstack([np.ravel(X1), np.ravel(X2)])).reshape([len(x2), len(x1)])

    fig, ax = plt.subplots()
    cont = ax.contourf(X1, X2, Z, levels=25)
    ax.plot(x0[0], x0[1], "ro")
    ax.set_title(title)
    ax.set_aspect(1)
    ax.grid(which="major")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    fig.colorbar(cont)

    return fig, ax


def update_plot(fig, ax, x, x_new):
    ax.plot(x_new[0], x_new[1], "ko", markersize=5, zorder=2)
    ax.plot([x[0], x_new[0]], [x[1], x_new[1]], "k", lw=0.8, zorder=1)
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(1)


def print_step(f, g, x, index):
    x_str = np.array2string(
        x, precision=4, floatmode="fixed", suppress_small=True, sign=" "
    )
    g_str = np.array2string(
        g(x), precision=4, floatmode="fixed", suppress_small=True, sign=" "
    )
    print(
        f"x({index}) = {x_str}, f(x({index})) = {f(x):.2f}, grad_f(x({index})) = {g_str}"
    )


if __name__ == "__main__":
    main()
