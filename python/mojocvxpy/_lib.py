"""Load the Mojo shared library and define its C signatures."""

from __future__ import annotations

import ctypes
import os
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get("MOJOCVXPY_LIB") or os.path.join(
    ROOT, "dist", "libmojo-cvxpy.so"
)

I = ctypes.c_int64

_SIGNATURES = {
    "mcvx_csr_matvec": ([I, I, I, I, I, I, I], None),
    "mcvx_csr_matvec_i32": ([I, I, I, I, I, I, I], None),
    "mcvx_csr_nonempty_rows": ([I, I, I, I], I),
    "mcvx_csr_nonempty_rows_i32": ([I, I, I, I], I),
    "mcvx_csc_compact": ([I] * 11, I),
    "mcvx_csc_compact_i32": ([I] * 11, I),
}


class BuildError(RuntimeError):
    pass


def build(force: bool = False) -> str:
    """Build the shared library when it is absent or older than its source."""
    source = os.path.join(ROOT, "src", "capi.mojo")
    if os.environ.get("MOJOCVXPY_LIB"):
        if os.path.exists(LIB):
            return LIB
        raise BuildError(f"MOJOCVXPY_LIB does not exist: {LIB}")
    stale = not os.path.exists(LIB) or os.path.getmtime(LIB) < os.path.getmtime(source)
    if force or stale:
        proc = subprocess.run(
            ["bash", os.path.join(ROOT, "build", "build.sh")],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        if proc.returncode != 0 or not os.path.exists(LIB):
            raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_library: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_library, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _library


def addr(array: np.ndarray) -> int:
    """Return a non-null address for a non-empty, C-contiguous NumPy buffer."""
    if not isinstance(array, np.ndarray):
        raise TypeError("FFI buffers must be NumPy arrays")
    if array.size == 0:
        raise ValueError("empty buffers must not cross the Mojo FFI boundary")
    if not array.flags.c_contiguous:
        raise ValueError("FFI buffers must be C-contiguous")
    return int(array.ctypes.data)


def i64(array) -> np.ndarray:
    return np.ascontiguousarray(array, dtype=np.int64)


def f64(array) -> np.ndarray:
    source = np.asarray(array)
    if np.iscomplexobj(source):
        raise TypeError("complex values are not supported")
    return np.ascontiguousarray(array, dtype=np.float64)
