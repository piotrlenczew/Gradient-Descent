import numpy as np
from autograd import grad
from typing import Optional, Callable, Any


class GradParam:
    def __init__(self, max_iter: Optional[int] = None, learning_rate: Optional[float] = None, tolerance: Optional[float] = None):
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
            self.learning_rate = 0.001


class GradResults:
    def __init__(self, iterations: [int], values: [float], reason_for_stop: str):
        self.iterations = iterations
        self.values = values
        self.reason_for_stop = reason_for_stop


def gradient_descent(f: Callable, x0: Any, grad_param: GradParam) -> GradResults:
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
            reason_for_stop = "Small difference between points. Close to local minimum."
            break
        values.append(f(next_x))
        iterations.append(i + 1)
        current_x = next_x
    if not reason_for_stop:
        reason_for_stop = "Reached max iterations."
    return GradResults(iterations, values, reason_for_stop)
