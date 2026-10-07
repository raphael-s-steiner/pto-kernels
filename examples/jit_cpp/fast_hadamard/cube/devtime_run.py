"""Inner driver for devtime_bench.py: run each config as one contiguous block.

Device time is read back from msprof's op_summary, so this process only has to
enqueue work; host cost does not matter as long as every launch is identical.
"""
import ctypes
import json
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch_npu  # noqa: F401

from cube.jit_util_hadamard_cube import hadamard_factors
from jit_util_common import get_current_stream_ptr, load_cdll, load_required_symbol

BASE = pathlib.Path(__file__).resolve().parents[1]

CUBE_ARGS = [
    ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32,
]
STD_ARGS = [
    ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
]

_LIBS: dict = {}


def kernel_for(so_path, argtypes):
    if so_path not in _LIBS:
        lib = load_cdll(so_path)
        _LIBS[so_path] = load_required_symbol(lib, "call_kernel", argtypes)
    return _LIBS[so_path]


def main():
    configs = json.loads(pathlib.Path(sys.argv[1]).read_text())
    device = "npu:0"
    torch.npu.set_device(device)
    for cfg in configs:
        batch, n = int(cfg["batch"]), int(cfg["n"])
        reps, warmup = int(cfg.get("reps", 30)), int(cfg.get("warmup", 5))
        block_dim = int(cfg.get("bd", 24))
        kind = cfg.get("kind", "cube")
        so = str(BASE / cfg.get("so", "cube/outputs/so/fast_hadamard_cube_jit.so"))

        x = torch.randn(batch, n, device=device, dtype=torch.float16)
        y = torch.empty_like(x)

        # Drains torch_npu's async task queue so the raw ctypes launches below
        # are ordered after the tensor setup. Must be re-resolved here for each
        # config -- but never inside the timed loop, which has no torch ops.
        stream_ptr = get_current_stream_ptr()

        if kind == "cube":
            fn = kernel_for(so, CUBE_ARGS)
            h_left, h_right = hadamard_factors(n, x.device)
            args = (block_dim, stream_ptr,
                    ctypes.c_void_p(x.data_ptr()), ctypes.c_void_p(y.data_ptr()),
                    ctypes.c_void_p(h_left.data_ptr()),
                    ctypes.c_void_p(h_right.data_ptr()), batch, n)
        else:
            so = str(BASE / "standard" / "outputs" / "so" / "fast_hadamard_jit.so")
            fn = kernel_for(so, STD_ARGS)
            args = (block_dim, stream_ptr, ctypes.c_void_p(x.data_ptr()),
                    batch, n, int(math.log2(n)))

        for _ in range(warmup):
            fn(*args)
        torch.npu.synchronize()
        for _ in range(reps):
            fn(*args)
        torch.npu.synchronize()
        print(f"CFG {cfg.get('tag', '')} batch={batch} n={n} bd={block_dim} "
              f"kind={kind} warmup={warmup} reps={reps}", flush=True)


if __name__ == "__main__":
    main()
