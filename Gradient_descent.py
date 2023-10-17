import numpy as np
from autograd import grad


alpha = 0.5


def g(x):
    result = 0

    for i, xi in enumerate(x):
        result += pow(alpha, (i - 1) / 9) * (xi**2)
    return result


def h(x):
    return (x + 2) ** 2 + 2


def solver(f, x0, tolerance=1e-6, max_iter=10):
    result = [x0]
    current_x = np.array(x0)

    gradient_f = grad(f)

    for i in range(max_iter - 1):
        gradient = np.array(gradient_f(current_x))
        if np.linalg.norm(gradient) <= tolerance:
            break
        next_x = current_x - alpha * gradient
        if np.linalg.norm(next_x - current_x) <= tolerance:
            break
        result.append(next_x.tolist())
        current_x = next_x
    return result


path = solver(g, [1.0, 3.0, 2.0, 4.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0])
print(path)

path1 = solver(h, 1.0)
print(path1)
