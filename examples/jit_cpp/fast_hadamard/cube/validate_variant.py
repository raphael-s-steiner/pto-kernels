"""Validate prebuilt cube-kernel variants against an exact fp64 FWHT.

The pytest suite JIT-compiles the shipped configuration, so it cannot check a
variant built by build_variant.sh. Run this on every variant BEFORE timing it:
a switch that computes the wrong answer is a broken lever, not a slow
technique, and its timing means nothing.

Covers 39 shapes x 3 block_dims per .so, chosen to exercise the tile
pipeline's prologue, steady state, ragged tail and power-of-two fallback.

  python3 cube/validate_variant.py cube/outputs/so_var/base.so [...]
"""
import ctypes, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch, torch_npu  # noqa: F401
from cube.jit_util_hadamard_cube import hadamard_factors
from jit_util_common import get_current_stream_ptr, load_cdll, load_required_symbol

BASE = pathlib.Path(__file__).resolve().parents[1]
ARGS = [ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32]

def fwht(x):                      # exact, natural (Sylvester) order
    y = x.to(torch.float64).clone()
    n = y.shape[-1]; h = 1
    while h < n:
        y = y.view(-1, n // (2 * h), 2, h)
        a, b = y[:, :, 0, :].clone(), y[:, :, 1, :].clone()
        y[:, :, 0, :], y[:, :, 1, :] = a + b, a - b
        y = y.reshape(-1, n); h *= 2
    return y

def main():
    torch.npu.set_device("npu:0")
    # batches: 1 tile, exact multiples, ragged tails, fallback chains, deep runs
    # the whole FAST_HADAMARD_CUBE_CASES table, not just the benchmarked sizes
    SHAPES = [(b, n) for n in (256, 512, 1024, 2048, 4096, 8192, 16384, 32768)
              for b in (1, 3, 31, 32, 33, 64, 96, 97, 127, 256, 1000, 1024)]
    worst = {}
    for so in sys.argv[1:]:
        fn = load_required_symbol(load_cdll(str(BASE / so)), "call_kernel", ARGS)
        bad = 0
        for batch, n in SHAPES:
            for bd in (1, 3, 24):
                torch.manual_seed(batch * 1000 + n + bd)
                x = torch.randn(batch, n, device="npu:0", dtype=torch.float16)
                y = torch.full_like(x, float("nan"))
                hl, hr = hadamard_factors(n, x.device)
                sp = get_current_stream_ptr()
                fn(bd, sp, x.data_ptr(), y.data_ptr(), hl.data_ptr(),
                   hr.data_ptr(), batch, n)
                torch.npu.synchronize()
                ref = fwht(x)
                got = y.to(torch.float64)
                if not torch.isfinite(got).all():
                    print(f"  NONFINITE b={batch} n={n} bd={bd}"); bad += 1; continue
                rms = ((got - ref) ** 2).mean().sqrt() / (ref ** 2).mean().sqrt()
                mx = (got - ref).abs().max() / ref.abs().max()
                key = (n,)
                worst[key] = max(worst.get(key, 0.0), float(rms))
                if rms > 1e-3 or mx > 1e-2:
                    print(f"  FAIL b={batch} n={n} bd={bd} rel_rms={rms:.3e} max={mx:.3e}")
                    bad += 1
        print(f"{so}: {'OK' if bad == 0 else str(bad) + ' FAILURES'}  "
              f"({len(SHAPES) * 3} cases)  worst rel_rms per n: "
              + ", ".join(f"n={k[0]}:{v:.2e}" for k, v in sorted(worst.items())))

main()
