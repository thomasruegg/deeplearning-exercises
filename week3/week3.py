import numpy as np


def f(x):
    return (1 / 4) * x[0] ** 4 + x[0] ** 3 - (17 / 4) * x[0] ** 2 - 6 * x[0] + (1 / 5) * x[1] ** 4 + (6 / 5) * x[
        1] ** 3 + 89


def gradient_f(x):
    return np.array([x[0] ** 3 + 3 * x[0] ** 2 - (17 / 2) * x[0] - 6, (4 / 5) * x[1] ** 3 + (18 / 5) * x[1] ** 2])


def hessian_f(x):
    return np.array([[3 * x[0] ** 2 + 6 * x[0] - (17 / 2), 0], [0, 12 / 5 * x[1] ** 2 + 36 / 5 * x[1]]])


def check_critical_point(hessian):
    eigenvalues = np.linalg.eigvals(hessian)
    if np.all(eigenvalues > 0):
        return 'minimum'
    elif np.all(eigenvalues < 0):
        return 'maximum'
    else:
        return 'saddle point'


def steepest_descent(start_point, learning_rate, iterations=10):
    x = start_point
    for i in range(iterations):
        x = x - learning_rate * gradient_f(x)
    return x


if __name__ == '__main__':

    print(f'ex1a')
    zeroes = np.solve(gradient_f, (x_1, x_2))
    print(f'ex1b')

    print(f'ex1c')
