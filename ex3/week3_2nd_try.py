from sympy import symbols, solve
import numpy as np

if __name__ == '__main__':
    x_1, x_2 = symbols('x_1 x_2')
    f_x = (1 / 4) * x_1 ** 4 + x_1 ** 3 - (17 / 4) * x_1 ** 2 - 6 * x_1 + (1 / 5) * x_2 ** 4 + (6 / 5) * x_2 ** 3 + 89

    gradient_f_x = [f_x.diff(x_1), f_x.diff(x_2)]
    hessian_f_x = [[f_x.diff(x_1, x_1), f_x.diff(x_1, x_2)], [f_x.diff(x_2, x_1), f_x.diff(x_2, x_2)]]

    critical_points = solve(gradient_f_x, (x_1, x_2))
    real_critical_points = np.real(critical_points)
    print(f'critical_points: {real_critical_points}')
    print(f'critical_point_1: {f_x.subs({x_1: critical_points[0][0], x_2: critical_points[0][1]})}')

    # hessian_of_critical_points = [hessian_f_x.subs({x_1: critical_point[0], x_2: critical_point[1]}) for critical_point in critical_points]
    # print(f'hessian_of_critical_points: {hessian_of_critical_points}')

