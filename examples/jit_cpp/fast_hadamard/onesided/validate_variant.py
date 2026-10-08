"""Validate prebuilt one-sided kernel variants against an exact fp64 FWHT.

The pytest suite JIT-compiles the shipped configuration, so it cannot check a
variant built by build_variant.sh. Run this on every variant BEFORE timing it:
a switch that computes the wrong answer is a broken lever, not a slow
technique, and its timing means nothing.

  python3 onesided/validate_variant.py onesided/outputs/so_var/base.so [...]
"""
import ctypes, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch, torch_npu  # noqa: F401
from onesided.jit_util_hadamard_onesided import hadamard_full
from jit_util_common import get_current_stream_ptr, load_cdll, load_required_symbol

BASE = pathlib.Path(__file__).resolve().parents[1]
ARGS = [ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32]


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
    # batch >= 16 only: the host wrapper pads below that, and these call the
    # kernel directly. Straddle every tile size and the backed-up tail.
    SHAPES = [(b, n) for n in (64, 128, 256, 512)
              for b in (16, 17, 31, 32, 33, 48, 64, 65, 127, 128, 255, 256,
                        1000, 1024)]
    for so in sys.argv[1:]:
        fn = load_required_symbol(load_cdll(str(BASE / so)), "call_kernel", ARGS)
        bad, worst = 0, {}
        for batch, n in SHAPES:
            for bd in (1, 3, 24):
                torch.manual_seed(batch * 1000 + n + bd)
                x = torch.randn(batch, n, device="npu:0", dtype=torch.float16)
                y = torch.full_like(x, float("nan"))
                h = hadamard_full(n, x.device)
                sp = get_current_stream_ptr()
                fn(bd, sp, x.data_ptr(), y.data_ptr(), h.data_ptr(), batch, n)
                torch.npu.synchronize()
                ref, got = fwht(x), y.to(torch.float64)
                if not torch.isfinite(got).all():
                    print(f"  NONFINITE b={batch} n={n} bd={bd}"); bad += 1; continue
                rms = ((got - ref) ** 2).mean().sqrt() / (ref ** 2).mean().sqrt()
                mx = (got - ref).abs().max() / ref.abs().max()
                worst[n] = max(worst.get(n, 0.0), float(rms))
                if rms > 1e-3 or mx > 1e-2:
                    print(f"  FAIL b={batch} n={n} bd={bd} rel_rms={rms:.3e} max={mx:.3e}")
                    bad += 1
        print(f"{so}: {'OK' if bad == 0 else str(bad) + ' FAILURES'}  "
              f"({len(SHAPES) * 3} cases)  worst rel_rms per n: "
              + ", ".join(f"n={k}:{v:.2e}" for k, v in sorted(worst.items())))


# Guarded so that _bigcheck.py can import fwht without re-running the whole
# sweep as an import side effect.
if __name__ == "__main__":
    main()
