# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "zarr==3.4.2",
#     "numpy",
#     "cupy-cuda12x[ctk]==14.2.0",
# ]
# ///
"""Check that zarr-python's GPU mode (zarr.config.enable_gpu()) works on this machine.

Writes and reads a 64 MiB float32 array through MemoryStore and LocalStore,
with and without zstd, on the CPU and GPU paths. It checks that the GPU path
returns CuPy arrays with identical contents, that stores interoperate between
the two paths, and prints rough timings.

Usage:  uv run zarr_gpu_check.py [store_dir]

Uses CUDA 12 CuPy wheels, so it covers Maxwell (sm_5x) GPUs that CUDA 13 dropped.
The [ctk] extra installs the CUDA toolkit (NVRTC, headers, cuBLAS) from pip, so
no system CUDA install is needed, only the NVIDIA driver.
"""

import sys
import tempfile
import time

import cupy as cp
import numpy as np
import zarr
from zarr.storage import LocalStore, MemoryStore

SHAPE, CHUNKS = (4096, 4096), (1024, 1024)


def device_info() -> None:
    props = cp.cuda.runtime.getDeviceProperties(0)
    major, minor = cp.cuda.Device(0).compute_capability
    print(f"GPU: {props['name'].decode()} (compute capability {major}.{minor})")
    print(f"CUDA runtime {cp.cuda.runtime.runtimeGetVersion()}, driver {cp.cuda.runtime.driverGetVersion()}")
    print(f"zarr {zarr.__version__}, cupy {cp.__version__}\n")


def sync() -> None:
    cp.cuda.Device().synchronize()


def run(label, store, compressors, gpu, host, dev) -> bool:
    ctx = zarr.config.enable_gpu() if gpu else zarr.config.set({})
    with ctx:
        a = zarr.create_array(store, shape=SHAPE, chunks=CHUNKS, dtype="f4",
                              compressors=compressors, overwrite=True)
        t = time.perf_counter()
        a[:] = dev if gpu else host
        sync()
        tw = time.perf_counter() - t
        t = time.perf_counter()
        out = a[:]
        sync()
        tr = time.perf_counter() - t
        sl = a[100:200, 3000:3100]
    expected = cp.ndarray if gpu else np.ndarray
    ok = isinstance(out, expected) and isinstance(sl, expected)
    ok &= np.array_equal(cp.asnumpy(out) if gpu else out, host)
    print(f"{'PASS' if ok else 'FAIL'}  {label:28s} {'gpu' if gpu else 'cpu'}  "
          f"returns={type(out).__module__.split('.')[0]:5s}  write={tw:.3f}s read={tr:.3f}s")
    return ok


def main() -> int:
    device_info()
    host = np.random.default_rng(0).random(SHAPE, dtype="f4")
    dev = cp.asarray(host)
    with tempfile.TemporaryDirectory() as tmp:
        root = sys.argv[1] if len(sys.argv) > 1 else tmp
        results = []
        for gpu in (False, True):
            results += [
                run("memory, no compression", MemoryStore(), None, gpu, host, dev),
                run("memory, zstd", MemoryStore(), "auto", gpu, host, dev),
                run("local, zstd", LocalStore(f"{root}/zstd_{'gpu' if gpu else 'cpu'}"), "auto", gpu, host, dev),
                run("local, no compression", LocalStore(f"{root}/raw_{'gpu' if gpu else 'cpu'}"), None, gpu, host, dev),
            ]
        with zarr.config.enable_gpu():
            g = zarr.open_array(LocalStore(f"{root}/zstd_cpu"))[:]
        c = zarr.open_array(LocalStore(f"{root}/zstd_gpu"))[:]
        cross = [isinstance(g, cp.ndarray) and bool(cp.array_equal(g, dev)),
                 isinstance(c, np.ndarray) and np.array_equal(c, host)]
        print(f"{'PASS' if cross[0] else 'FAIL'}  CPU-written store read on GPU")
        print(f"{'PASS' if cross[1] else 'FAIL'}  GPU-written store read on CPU")
        results += cross
    print(f"\n{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
