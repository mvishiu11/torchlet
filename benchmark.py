import datetime
import json
import os
import time

from torchlet.engine import Element
from torchlet.nn import MLP


"""
This script benchmarks the performance of the engine and neural network modules.
Benchmarks are run 1000 times (or 100 times for specific functions) to calculate the average time per run.
The benchmarks are executed on a local CPU, and the results are compared with the best previous results saved in a JSON file.
If the current benchmarks are better, the user is prompted to update the JSON file and optionally the README.
This script is intended for internal use only.
"""


def load_json(filename=os.path.join("benchmark_data", "benchmarks.json")):
    """Load benchmark data from a JSON file.

    Args:
        filename (str): The name of the JSON file to load.

    Returns:
        dict: A dictionary containing the benchmark results.
    """
    if os.path.exists(filename):
        with open(filename, "r") as f:
            return json.load(f)
    return {}


def save_json(data, filename=os.path.join("benchmark_data", "benchmarks.json")):
    """Save benchmark data to a JSON file.

    Args:
        data (dict): The benchmark data to save.
        filename (str): The name of the JSON file to save the data.
    """
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)


def benchmark_addition():
    """Benchmark the addition operation in the Element class.

    Returns:
        float: The average duration of the addition operation in microseconds.
    """
    a = Element(1.0)
    b = Element(2.0)
    start_time = time.perf_counter()
    for _ in range(1000):
        c = a + b  # noqa F841
    duration = (time.perf_counter() - start_time) / 1000 * 1e6
    return duration


def benchmark_multiplication():
    """Benchmark the multiplication operation in the Element class.

    Returns:
        float: The average duration of the multiplication operation in microseconds.
    """
    a = Element(1.0)
    b = Element(2.0)
    start_time = time.perf_counter()
    for _ in range(1000):
        c = a * b  # noqa F841
    duration = (time.perf_counter() - start_time) / 1000 * 1e6
    return duration


def benchmark_backward():
    """Benchmark the backward pass in the computational graph.

    Returns:
        float: The duration of the backward pass in microseconds.
    """
    a = Element(1.0)
    b = Element(2.0)
    c = a * b + a
    start_time = time.perf_counter()
    c.backward()
    duration = (time.perf_counter() - start_time) * 1e6
    return duration


def benchmark_mlp_forward():
    """Benchmark the forward pass of a Multi-Layer Perceptron (MLP).

    Returns:
        float: The average duration of the MLP forward pass in microseconds.
    """
    mlp = MLP(3, [4, 4, 1])
    x = [Element(1.0), Element(2.0), Element(3.0)]
    start_time = time.perf_counter()
    for _ in range(100):
        y = mlp(x)  # noqa F841
    duration = (time.perf_counter() - start_time) / 100 * 1e6
    return duration


def benchmark_mlp_backward():
    """Benchmark the backward pass of a Multi-Layer Perceptron (MLP).

    Returns:
        float: The duration of the MLP backward pass in microseconds.
    """
    mlp = MLP(3, [4, 4, 1])
    x = [Element(1.0), Element(2.0), Element(3.0)]
    y = mlp(x)

    if isinstance(y, Element):
        loss = y * Element(1.0)
    else:
        loss = y[0] * Element(1.0)
    start_time = time.perf_counter()
    loss.backward()
    duration = (time.perf_counter() - start_time) * 1e6
    return duration


def benchmark_zero_grad():
    """Benchmark the zero_grad function in the MLP class.

    Returns:
        float: The average duration of the zero_grad operation in microseconds.
    """
    mlp = MLP(3, [4, 4, 1])
    start_time = time.perf_counter()
    for _ in range(100):
        mlp.zero_grad()
    duration = (time.perf_counter() - start_time) / 100 * 1e6
    return duration


def run_benchmarks():
    """Run all the benchmarks and return the results.

    Returns:
        dict: A dictionary containing the results of all benchmarks.
    """
    return {
        "Addition Benchmark": benchmark_addition(),
        "Multiplication Benchmark": benchmark_multiplication(),
        "Backward Pass Benchmark": benchmark_backward(),
        "MLP Forward Pass Benchmark": benchmark_mlp_forward(),
        "MLP Backward Pass Benchmark": benchmark_mlp_backward(),
        "Zero Grad Benchmark": benchmark_zero_grad(),
    }


def compare_benchmarks(current, best):
    """Compare the current benchmark results with the best saved results.

    Args:
        current (dict): The current benchmark results.
        best (dict): The best benchmark results saved in the JSON file.

    Returns:
        dict: A dictionary containing the benchmarks that have improved.
    """
    improvements = {}
    for key in current:
        if key not in best or current[key] < best[key]:
            improvements[key] = current[key]
    return improvements


def update_benchmarks(best, improvements):
    """Update the benchmark JSON file with improved results.

    Args:
        best (dict): The best benchmark results saved in the JSON file.
        improvements (dict): The improved benchmark results.
    """
    best.update(improvements)
    save_json(best)


def log_positive_changes(
    improvements, log_file=os.path.join("benchmark_data", "benchmark_log.json")
):
    """Log the positive changes in the benchmark results to a JSON file.

    Args:
        improvements (dict): A dictionary containing the positive changes in the benchmark results.
        log_file (str): The name of the JSON file to log the improvements.
    """
    if not improvements:
        return

    log_entry = {
        "datetime": datetime.datetime.now().isoformat(),
        "title": input("Enter a title for this benchmark improvement: "),
        "improvements": improvements,
    }

    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            logs = json.load(f)
    else:
        logs = []

    logs.append(log_entry)

    with open(log_file, "w") as f:
        json.dump(logs, f, indent=4)

    print(f"Logged benchmark improvements to {log_file}")


def main():
    """Main function to run benchmarks and handle user interactions."""
    best_results = load_json()
    print(best_results)
    current_results = run_benchmarks()
    improvements = compare_benchmarks(current_results, best_results)

    if improvements:
        print("New benchmark improvements found:")
        for k, v in improvements.items():
            print(f"{k}: {v:.4f} µs (previous best: {best_results.get(k, 'N/A')} µs)")

        user_input = input("Do you want to update the best benchmarks? (y/n): ")
        if user_input.lower() == "y":
            update_benchmarks(best_results, improvements)
            log_positive_changes(improvements)

    else:
        print("No improvements in the benchmark results.")


if __name__ == "__main__":
    main()
