"""Inner driver for devtime_bench.py: run each config as one contiguous block.

Device time is read back from msprof's op_summary, so host cost does not matter
as long as every launch is identical.
"""
import ctypes
import json
import math
import pathlib
import sys

# msprof launches this with its own cwd, so the example root goes on the path
# explicitly -- bench_common and jit_util_common live there, not in cube/.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch_npu  # noqa: F401

from cube.jit_util_hadamard_twosided import (bitplan_operands,
                                            hadamard_factors)
from cube.jit_util_hadamard_onesided import hadamard_full
from jit_util_common import get_current_stream_ptr, load_cdll, load_required_symbol

BASE = pathlib.Path(__file__).resolve().parents[1]

# One list per kernel's call_kernel, each of which MUST mirror the C++
# signature: ctypes does not check, so a stale list passes the wrong arguments
# and the kernel reads whatever is in the register.
TWOSIDED_ARGS = [
    ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32,
]
STD_ARGS = [
    ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
]
ONESIDED_ARGS = [
    ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32,
]

# One .so per kind, so a config only needs `so` to point at a variant build.
DEFAULT_SO = {
    "twosided": "cube/outputs/so/fast_hadamard_twosided_jit.so",
    "bitplan": "cube/outputs/so/fast_hadamard_twosided_jit.so",
    "onesided": "cube/outputs/so/fast_hadamard_onesided_jit.so",
    "std": "standard/outputs/so/fast_hadamard_jit.so",
}

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
        kind = cfg.get("kind", "twosided")
        so = str(BASE / cfg.get("so", DEFAULT_SO.get(kind, DEFAULT_SO["std"])))

        # The vector kernel is IN PLACE, so relaunching it on one buffer makes
        # the values diverge (H*H = n*I, so fp16 saturates to inf within a few
        # reps). Cycle a pool of pre-filled buffers, allocated OUTSIDE the timed
        # loop so no torch op lands between launches.
        pool = int(cfg.get("pool", 1))
        xs = [torch.randn(batch, n, device=device, dtype=torch.float16)
              for _ in range(max(1, pool))]
        x = xs[0]
        y = torch.empty_like(x)

        # Drains torch_npu's async task queue, so the raw ctypes launches below
        # are ordered after the tensor setup. Re-resolved per config, never
        # inside the timed loop.
        stream_ptr = get_current_stream_ptr()

        if kind == "onesided":
            fn = kernel_for(so, ONESIDED_ARGS)
            h = hadamard_full(n, x.device, cfg.get("block"))
            args = (block_dim, stream_ptr,
                    ctypes.c_void_p(x.data_ptr()), ctypes.c_void_p(y.data_ptr()),
                    ctypes.c_void_p(h.data_ptr()), batch, n)
        elif kind == "twosided":
            fn = kernel_for(so, TWOSIDED_ARGS)
            h_left, h_right = hadamard_factors(n, x.device, cfg.get("block"))
            args = (block_dim, stream_ptr,
                    ctypes.c_void_p(x.data_ptr()), ctypes.c_void_p(y.data_ptr()),
                    ctypes.c_void_p(h_left.data_ptr()),
                    ctypes.c_void_p(h_right.data_ptr()), batch, n)
        elif kind == "bitplan":
            # The conversion-free single-view tile: `batch` samples of width n
            # driven as batch/samples_per_tile rows of T = A*B elements against
            # the tensor-product operands. Same .so and entry point.
            fn = kernel_for(so, TWOSIDED_ARGS)
            op_a, op_b, spt = bitplan_operands(n, x.device)
            tiles, tile_n = batch // spt, spt * n
            assert tiles, f"batch {batch} is below one bit-plan tile ({spt})"
            args = (block_dim, stream_ptr,
                    ctypes.c_void_p(x.data_ptr()), ctypes.c_void_p(y.data_ptr()),
                    ctypes.c_void_p(op_a.data_ptr()),
                    ctypes.c_void_p(op_b.data_ptr()), tiles, tile_n)
        else:
            fn = kernel_for(so, STD_ARGS)
            args = (block_dim, stream_ptr, ctypes.c_void_p(x.data_ptr()),
                    batch, n, int(math.log2(n)))

        # One argument tuple per pool slot, so the loop does nothing but launch.
        if len(xs) > 1:
            arg_sets = [tuple(ctypes.c_void_p(xs[k].data_ptr()) if j == 2 else a
                              for j, a in enumerate(args))
                        for k in range(len(xs))]
        else:
            arg_sets = [args]

        for i in range(warmup):
            fn(*arg_sets[i % len(arg_sets)])
        torch.npu.synchronize()
        for i in range(reps):
            fn(*arg_sets[i % len(arg_sets)])
        torch.npu.synchronize()
        print(f"CFG {cfg.get('tag', '')} batch={batch} n={n} bd={block_dim} "
              f"kind={kind} warmup={warmup} reps={reps}", flush=True)


if __name__ == "__main__":
    main()
