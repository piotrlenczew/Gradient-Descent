import numpy as np
import functools
import matplotlib.pyplot as plt
from autograd import grad
from typing import List


def g(x, alpha):
    i = np.arange(len(x))
    exponent = (i - 1) / 9
    result = np.sum(np.power(alpha, exponent) * np.square(x))
    return result


g_alpha_1 = functools.partial(g, alpha=1)
g_alpha_10 = functools.partial(g, alpha=10)
g_alpha_100 = functools.partial(g, alpha=100)


# def h(x):
#    return (x + 2) ** 2 + 2


class GradParam:
    def __init__(self, tolerance=None, max_iter=None, learning_rate=None):
        if tolerance:
            self.tolerance = tolerance
        else:
            self.tolerance = 1e-6
        if max_iter:
            self.max_iter = max_iter
        else:
            self.max_iter = 100
        if learning_rate:
            self.learning_rate = learning_rate
        else:
            self.learning_rate = 0.01


class GradResults:
    def __init__(self, iterations, values, reason_for_stop):
        self.iterations = iterations
        self.values = values
        self.reason_for_stop = reason_for_stop


def gradient_descent(f, x0, grad_param: GradParam) -> GradResults:
    values = [f(x0)]
    iterations = [0]
    reason_for_stop = None
    current_x = np.array(x0)

    gradient_f = grad(f)

    for i in range(grad_param.max_iter - 1):
        gradient = np.array(gradient_f(current_x))
        if np.linalg.norm(gradient) <= grad_param.tolerance:
            reason_for_stop = "Small gradient. Close to local minimum."
            break
        next_x = current_x - grad_param.learning_rate * gradient
        if np.linalg.norm(next_x - current_x) <= grad_param.tolerance:
            reason_for_stop = "Small difference between points. CLose to local minimum."
            break
        values.append(f(next_x))
        iterations.append(i + 1)
        current_x = next_x
    if not reason_for_stop:
        reason_for_stop = "Reached max iterations."
    return GradResults(iterations, values, reason_for_stop)


def plot_convergence(data_list: List[GradResults], labels):
    plt.figure(figsize=(10, 6))
    graph_colors = ["b", "g", "r", "c", "m", "y"]

    for i, data in enumerate(data_list):
        iterations = data.iterations
        values = data.values
        plt.plot(
            iterations,
            values,
            #marker="o",
            linestyle="-",
            color=graph_colors[i % len(graph_colors)],
            label=labels[i],
        )

    plt.xlabel("Iterations")
    plt.ylabel("Function Value")
    plt.title("Convergence of Gradient Descent")
    plt.grid(True)
    plt.legend()
    plt.show()


grad_result_alpha_1 = gradient_descent(
    g_alpha_1,
    [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
    GradParam(),
)
# print(grad_result_alpha_1.values)
# print(grad_result_alpha_1.iterations)
# print(grad_result_alpha_1.reason_for_stop)
#
grad_result_alpha_10 = gradient_descent(
    g_alpha_10,
    [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
    GradParam(),
)
# print(grad_result_alpha_10.values)
# print(grad_result_alpha_10.iterations)
# print(grad_result_alpha_10.reason_for_stop)
#
grad_result_alpha_100 = gradient_descent(
    g_alpha_100,
    [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
    GradParam(),
)
# print(grad_result_alpha_100.values)
# print(grad_result_alpha_100.iterations)
# print(grad_result_alpha_100.reason_for_stop)

# path1 = gradient_descent(h, 22.0, GradParam())
# print(path1.values)
# print(path1.iterations)
# print(path1.reason_for_stop)

plot_convergence(
    [grad_result_alpha_1, grad_result_alpha_10, grad_result_alpha_100],
    ["alpha_1", "alpha_10", "alpha_100"],
)

# do czasu użyj time
# zrób 2 wykresy na każdym dla 3 alpha jeden od iteracji i drugi od czasu
