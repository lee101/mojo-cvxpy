"""Benchmark covered canonicalization stages against CVXPY."""

from __future__ import annotations

import math
import os
import platform
import sys
import time

import cvxpy
import numpy as np
import scipy.sparse as sp
from cvxpy.cvxcore.python import canonInterface as upstream

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"),
)

import mojocvxpy as mcp  # noqa: E402


def timeit(fn, repeat=5):
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def machine():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def make_tensor():
    rng = np.random.default_rng(2026)
    var_length = 48
    constraints = 30_000
    params = 64
    entries = 500_000
    shape = (constraints * (var_length + 1), params + 1)
    rows = rng.integers(0, shape[0], entries)
    cols = rng.integers(0, shape[1], entries)
    values = rng.normal(size=entries)
    tensor = sp.csc_array((values, (rows, cols)), shape=shape)
    param = rng.normal(size=params + 1)
    return tensor, param, var_length


def main():
    tensor, param, var_length = make_tensor()
    reduced, indices, indptr, shape = upstream.reduce_problem_data_tensor(
        tensor.copy(), var_length
    )
    index = (indices, indptr, shape)

    cases = [
        (
            "reduce problem-data tensor (500k triplets)",
            lambda: mcp.reduce_problem_data_tensor(tensor.copy(), var_length),
            lambda: upstream.reduce_problem_data_tensor(tensor.copy(), var_length),
        ),
        (
            "apply parameter vector and rebuild CSC",
            lambda: mcp.get_matrix_from_tensor(
                reduced, param, var_length, problem_data_index=index
            ),
            lambda: upstream.get_matrix_from_tensor(
                reduced, param, var_length, problem_data_index=index
            ),
        ),
        (
            "20 cached parameter applications",
            lambda: [
                mcp.get_matrix_from_tensor(
                    reduced, param, var_length, problem_data_index=index
                )
                for _ in range(20)
            ],
            lambda: [
                upstream.get_matrix_from_tensor(
                    reduced, param, var_length, problem_data_index=index
                )
                for _ in range(20)
            ],
        ),
    ]

    print(f"Machine: {machine()}")
    print(f"Software: CVXPY {cvxpy.__version__}, Python {platform.python_version()}")
    print()
    print("| case | mojo-cvxpy | CVXPY | result |")
    print("| --- | ---: | ---: | ---: |")
    for name, ours, theirs in cases:
        ours()
        theirs()
        mojo_time = timeit(ours, repeat=3)
        cvxpy_time = timeit(theirs, repeat=3)
        speedup = cvxpy_time / mojo_time
        label = f"{speedup:.2f}x faster" if speedup >= 1 else f"{1/speedup:.2f}x slower"
        print(
            f"| {name} | {mojo_time * 1e3:.2f} ms | "
            f"{cvxpy_time * 1e3:.2f} ms | {label} |"
        )


if __name__ == "__main__":
    main()
