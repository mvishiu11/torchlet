import time

from torchlet.engine import Element
from torchlet.nn import MLP


def benchmark_addition():
    a = Element(1.0)
    b = Element(2.0)
    start_time = time.perf_counter()
    for _ in range(1000):
        c = a + b  # noqa F841
    duration = (
        (time.perf_counter() - start_time) / 1000 * 1e6
    )  # Convert to microseconds
    print(f"Addition benchmark: {duration:.3f} µs per operation")


def benchmark_multiplication():
    a = Element(1.0)
    b = Element(2.0)
    start_time = time.perf_counter()
    for _ in range(1000):
        c = a * b  # noqa F841
    duration = (
        (time.perf_counter() - start_time) / 1000 * 1e6
    )  # Convert to microseconds
    print(f"Multiplication benchmark: {duration:.3f} µs per operation")


def benchmark_backward():
    a = Element(1.0)
    b = Element(2.0)
    c = a * b + a
    start_time = time.perf_counter()
    c.backward()
    duration = (time.perf_counter() - start_time) * 1e6  # Convert to microseconds
    print(f"Backward pass benchmark: {duration:.6f} µs")


def benchmark_mlp_forward():
    mlp = MLP(3, [4, 4, 1])
    x = [Element(1.0), Element(2.0), Element(3.0)]
    start_time = time.perf_counter()
    for _ in range(100):
        y = mlp(x)  # noqa F841
    duration = (time.perf_counter() - start_time) / 100 * 1e6  # Convert to microseconds
    print(f"MLP forward pass benchmark: {duration:.3f} µs per run")


def benchmark_mlp_backward():
    mlp = MLP(3, [4, 4, 1])
    x = [Element(1.0), Element(2.0), Element(3.0)]
    y = mlp(x)

    if isinstance(y, Element):
        loss = y * Element(1.0)  # Simplified loss
    else:
        loss = y[0] * Element(1.0)  # In case y is a list

    start_time = time.perf_counter()
    loss.backward()
    duration = (time.perf_counter() - start_time) * 1e6  # Convert to microseconds
    print(f"MLP backward pass benchmark: {duration:.6f} µs")


def benchmark_zero_grad():
    mlp = MLP(3, [4, 4, 1])
    start_time = time.perf_counter()
    for _ in range(100):
        mlp.zero_grad()
    duration = (time.perf_counter() - start_time) / 100 * 1e6  # Convert to microseconds
    print(f"Zero grad benchmark: {duration:.3f} µs per run")


if __name__ == "__main__":
    benchmark_addition()
    benchmark_multiplication()
    benchmark_backward()
    benchmark_mlp_forward()
    benchmark_mlp_backward()
    benchmark_zero_grad()
