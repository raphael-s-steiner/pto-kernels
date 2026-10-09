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

from cube.jit_util_hadamard_cube import (bitplan_operands,
                                            hadamard_factors)
from onesided.jit_util_hadamard_onesided import hadamard_full
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
ONESIDED_ARGS = [
    ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32,
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

        # The vector kernel is IN PLACE, so relaunching it on one buffer makes
        # the values diverge (H*H = n*I, so magnitude grows by n every two
        # launches and fp16 saturates to inf within a few reps). Cycle a pool
        # of pre-filled buffers instead -- allocated OUTSIDE the timed loop, so
        # no torch op lands between launches. The established benchmark in
        # bench_common does the same thing for the same reason.
        pool = int(cfg.get("pool", 1))
        xs = [torch.randn(batch, n, device=device, dtype=torch.float16)
              for _ in range(max(1, pool))]
        x = xs[0]
        y = torch.empty_like(x)

        # Drains torch_npu's async task queue so the raw ctypes launches below
        # are ordered after the tensor setup. Must be re-resolved here for each
        # config -- but never inside the timed loop, which has no torch ops.
        stream_ptr = get_current_stream_ptr()

        if kind == "onesided":
            fn = kernel_for(so, ONESIDED_ARGS)
            h = hadamard_full(n, x.device, cfg.get("block"))
            args = (block_dim, stream_ptr,
                    ctypes.c_void_p(x.data_ptr()), ctypes.c_void_p(y.data_ptr()),
                    ctypes.c_void_p(h.data_ptr()), batch, n)
        elif kind == "cube":
            fn = kernel_for(so, CUBE_ARGS)
            h_left, h_right = hadamard_factors(n, x.device, cfg.get("block"))
            args = (block_dim, stream_ptr,
                    ctypes.c_void_p(x.data_ptr()), ctypes.c_void_p(y.data_ptr()),
                    ctypes.c_void_p(h_left.data_ptr()),
                    ctypes.c_void_p(h_right.data_ptr()), batch, n)
        elif kind == "bitplan":
            # The conversion-free single-view tile: `batch` samples of width n
            # are driven as batch/samples_per_tile rows of T = A*B elements,
            # against the tensor-product operands. Same .so and same entry
            # point as `cube` -- only the operands and the declared shape move.
            fn = kernel_for(so, CUBE_ARGS)
            op_a, op_b, spt = bitplan_operands(n, x.device)
            tiles, tile_n = batch // spt, spt * n
            assert tiles, f"batch {batch} is below one bit-plan tile ({spt})"
            args = (block_dim, stream_ptr,
                    ctypes.c_void_p(x.data_ptr()), ctypes.c_void_p(y.data_ptr()),
                    ctypes.c_void_p(op_a.data_ptr()),
                    ctypes.c_void_p(op_b.data_ptr()), tiles, tile_n)
        else:
            so = str(BASE / "standard" / "outputs" / "so" / "fast_hadamard_jit.so")
            fn = kernel_for(so, STD_ARGS)
            args = (block_dim, stream_ptr, ctypes.c_void_p(x.data_ptr()),
                    batch, n, int(math.log2(n)))

        # Pre-build one argument tuple per pool slot so the loop stays free of
        # any work beyond the launch itself.
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
