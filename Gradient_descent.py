import numpy as np
from autograd import grad


def g(x, alpha):
    result = 0

    for i, xi in enumerate(x):
        result += pow(alpha, (i-1)/9)*(xi**2)
    return result


def h(x):
    return (x+2) ** 2 + 2


def solver(f, x0, alpha, tolerance=1e-6, max_iter=10):
    result = [x0]
    current_x = x0

    gradient_f = grad(f)

    for i in range(max_iter-1):
        gradient = gradient_f(current_x)
        if gradient <= tolerance:
            break
        next_x = current_x - alpha * gradient
        if np.abs(next_x-current_x) <= tolerance:
            break
        result.append(next_x)
        current_x = next_x
    print(len(result))
    return result


path = solver(h, 3.0, 0.5)
print(path)
print(g(path, 0.5))
