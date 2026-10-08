"""Correctness of the PTO_OS_BIG sizes (N = 1024, 2048), which the shipped case
table does not include and validate_variant.py therefore does not cover.

  bash onesided/build_variant.sh big -DPTO_OS_BIG=1
  python3 onesided/bigcheck_variant.py onesided/outputs/so_var/big.so

These sizes compile and are correct, but the two-sided kernel in cube/ is
1.7x-4.7x faster there, which is why they stay off. See the README.
"""
import ctypes, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch, torch_npu  # noqa: F401
from onesided.jit_util_hadamard_onesided import hadamard_full
from onesided.validate_variant import fwht
from jit_util_common import get_current_stream_ptr, load_cdll, load_required_symbol
BASE = pathlib.Path(__file__).resolve().parents[1]
ARGS = [ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32]
torch.npu.set_device("npu:0")
for so in sys.argv[1:]:
    fn = load_required_symbol(load_cdll(str(BASE / so)), "call_kernel", ARGS)
    bad, worst = 0, {}
    for n in (1024, 2048):
        for batch in (16, 17, 64, 100, 256, 1024):
            for bd in (1, 24):
                torch.manual_seed(batch + n + bd)
                x = torch.randn(batch, n, device="npu:0", dtype=torch.float16)
                y = torch.full_like(x, float("nan"))
                h = hadamard_full(n, x.device)
                fn(bd, get_current_stream_ptr(), x.data_ptr(), y.data_ptr(),
                   h.data_ptr(), batch, n)
                torch.npu.synchronize()
                ref, got = fwht(x), y.to(torch.float64)
                rms = float(((got-ref)**2).mean().sqrt() / (ref**2).mean().sqrt())
                worst[n] = max(worst.get(n, 0.0), rms)
                if not torch.isfinite(got).all() or rms > 2e-3:
                    print(f"  FAIL n={n} b={batch} bd={bd} rel_rms={rms:.3e}"); bad += 1
    print(f"{so}: {'OK' if bad == 0 else str(bad)+' FAILURES'}  "
          + ", ".join(f"n={k}:{v:.2e}" for k, v in sorted(worst.items())))
