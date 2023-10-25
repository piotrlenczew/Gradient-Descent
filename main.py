import functools
import numpy as np
import time
from Gradient_descent import gradient_descent, GradParam, GradResults
from Plotting import plot_convergence


def g(x: np.array, alpha: int):
    i = np.arange(len(x))
    exponent = (i - 1) / 9
    result = np.sum(np.power(alpha, exponent) * np.square(x))
    return result


functions = []

functions.append(functools.partial(g, alpha=1))
functions.append(functools.partial(g, alpha=10))
functions.append(functools.partial(g, alpha=100))


start = time.time()
grad_result_alpha_1 = gradient_descent(
    g_alpha_1,
    [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
    GradParam(),
)
end = time.time()
time_for_alpha_1 = end - start

start = time.time()
grad_result_alpha_10 = gradient_descent(
    g_alpha_10,
    [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
    GradParam(),
)
end = time.time()
time_for_alpha_10 = end - start

start = time.time()
grad_result_alpha_100 = gradient_descent(
    g_alpha_100,
    [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
    GradParam(),
)
end = time.time()
time_for_alpha_100 = end - start

print(f'Alpha=1\nReason for stop: {grad_result_alpha_1.reason_for_stop}\nTime in seconds: {time_for_alpha_1:.4f}\n')
print(f'Alpha=10\nReason for stop: {grad_result_alpha_10.reason_for_stop}\nTime in seconds: {time_for_alpha_10:.4f}\n')
print(f'Alpha=100\nReason for stop: {grad_result_alpha_100.reason_for_stop}\nTime in seconds: {time_for_alpha_100:.4f}\n')

plot_convergence(
    [grad_result_alpha_1, grad_result_alpha_10, grad_result_alpha_100],
    ["alpha_1", "alpha_10", "alpha_100"],
)
