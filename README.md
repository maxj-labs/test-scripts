# test-scripts

Small self-contained checks, each runnable with [uv](https://docs.astral.sh/uv/)
(dependencies are declared inline, PEP 723).

| Script | What it checks |
|---|---|
| [`zarr-gpu/zarr_gpu_check.py`](zarr-gpu/zarr_gpu_check.py) | zarr-python `enable_gpu()` round-trips with CuPy (CUDA 12) |

```sh
uv run zarr-gpu/zarr_gpu_check.py
```
