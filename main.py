import functools
import numpy as np
import time
import pandas as pd
from Gradient_descent import gradient_descent, GradParam
from Plotting import plot_convergence


def g(x: np.array, alpha: int):
    i = np.arange(len(x))
    exponent = (i - 1) / 9
    result = np.sum(np.power(alpha, exponent) * np.square(x))
    return result


learning_rates = [0.1, 0.01, 0.001]
alphas = [1, 10, 100]
starting_point = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]

functions = [functools.partial(g, alpha=alpha) for alpha in alphas]

for learning_rate in learning_rates:
    results_for_table = []
    results_for_graph = []

    for index, function in enumerate(functions):
        start = time.time()
        grad_result = gradient_descent(
            function,
            starting_point,
            GradParam(learning_rate),
        )
        end = time.time()
        elapsed_time = end - start

        results_for_table.append(
            {
                "Alpha": alphas[index],
                "Reason for stop": grad_result.reason_for_stop,
                "Time (s)": elapsed_time,
            }
        )

        results_for_graph.append(grad_result)

    print(f"Table for learning rate = {learning_rate}")
    table = pd.DataFrame(results_for_table)
    print(f"{table}\n")

    labels = [f"alpha_{alpha}" for alpha in alphas]
    plot_convergence(
        results_for_graph,
        labels,
        f"Convergence of Gradient Descent for learning rate = {learning_rate}",
    )
