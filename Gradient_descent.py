import numpy as np
from autograd import grad


def g(x):
    return x[0] ** 2 + 2 * x[1] ** 2


def solver(f, x0, alpha, tolerance=1e-6, max_iter=100):
    x0 = np.array(x0)
    x1 = x0
    iteration = 0

    gradient_f = grad(f)

    while iteration < max_iter:
        gradient = gradient_f(x0)
        if np.linalg.norm(gradient) <= tolerance:
            break
        x1 = x0 - alpha * gradient
        if np.linalg.norm(x1-x0) <= tolerance:
            break
        x0 = x1
        iteration += 1
    return x1


print(solver(g, [3.0, 4.0], 1))
