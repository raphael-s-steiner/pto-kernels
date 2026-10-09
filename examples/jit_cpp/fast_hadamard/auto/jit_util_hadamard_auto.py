"""One Hadamard entry point that branches on N to the fastest kernel.

Three implementations exist in this example, and which one wins is decided
almost entirely by N:

  - `onesided/`  Y = X * H_N as a single right multiplication, H_N = H_B (x) H_M
                 applied as fixed-size matmuls accumulated in L0C. Costs N*N
                 MACs per sample but touches each pipe once per tile.
  - `cube/`      Y = H_R * X * H_C, the balanced Kronecker split. Costs only
                 N*(R+C) MACs per sample but needs a turnaround between the two
                 multiplies.
  - `standard/`  the vector-unit butterfly, log2(N) stages in UB.

The one-sided product's MAC count grows 4x per doubling of N against 2x for its
bytes, so it wins while N is small and loses once arithmetic dominates; the
split's extra turnaround costs more than it saves while N is small. They cross
between 512 and 1024, which is where this module switches.

**The vector kernel is not a branch here: it is never the fastest choice in the
supported range.** Measured over the required grid (N 128..16384 x batch
1/16/64/256/1024, msprof device time on 910B2, each kernel at its own best
block_dim) it loses every one of the 40 cells, by 1.17x to 3.58x. It stays in
`standard/` as the baseline these kernels are measured against, and because it
is the only one that runs N below 64.

Supported N is the union of the two cube kernels: powers of two in
[64, 32768]. The required grid, `bench_common.BENCH_HIDDEN_DIMS` =
[128 .. 16384], sits strictly inside that, with N=64 reachable only by the
one-sided kernel and N=32768 only by the two-sided one.
"""
from pathlib import Path

from cube.jit_util_hadamard_cube import (
    MAX_CUBE_HADAMARD_N,
    MIN_CUBE_HADAMARD_N,
    jit_compile as jit_compile_cube,
)
from jit_util_common import DEFAULT_DEVICE, normalize_npu_device
from onesided.jit_util_hadamard_onesided import (
    MAX_ONESIDED_N,
    MIN_ONESIDED_N,
    jit_compile as jit_compile_onesided,
)

MIN_AUTO_N = MIN_ONESIDED_N          # 64
MAX_AUTO_N = MAX_CUBE_HADAMARD_N     # 32768

# The crossover, measured rather than reasoned. Device time in us (msprof,
# 910B2, pool of one buffer so all kernels see the same footprint, each kernel
# at its own best block_dim; the one-sided host pads batch < 16 up to 16, so its
# batch-1 cost IS its batch-16 cost):
#
#     N  batch   vector  onesided  twosided
#   256      16    3.160     1.700     2.121
#   256    1024    4.100     2.880     6.160
#   512      16    3.100     2.180     2.420
#   512    1024    5.460     4.321     8.760
#  1024      16    3.280        --     1.980
#  1024    1024    8.480        --     4.680
#
# Those two-sided numbers at N=512 are the CLASSIC path and are now stale: the
# bit-plan tile made that size 1.73-2.47x faster, which is what turned N=512
# from a clear one-sided win into the batch-dependent crossover handled by
# BITPLAN_BATCH_MIN below. At N=1024 and above the split still wins everywhere,
# and below N=512 the one-sided kernel still does, by enough that neither needs
# a batch term.
ONESIDED_MAX_N = MAX_ONESIDED_N      # 512

# N=512 is the one size where batch moves the choice, because the two-sided
# kernel's bit-plan tile (see `bitplan_operands` in cube/) made it 1.73-2.47x
# faster there and so turned a clear loss into a crossover. It needs 16 samples
# to fill one tile, and a few tiles to fill the pipes; below that the one-sided
# kernel's single-pipe-pass tile still wins. Device time in us, each at its own
# best block_dim:
#
#   batch   onesided   twosided+bitplan
#      16      2.140   2.260
#      32      2.220   2.320
#      64      2.310   2.440
#     128      2.520   2.620
#     256      2.840   2.820
#    1024      4.300   3.850
#    4096      8.901   6.430
#   16384     27.090  16.020
#
# One-sided holds a flat 1.04-1.06x below 256; above it the bit-plan runs away,
# to 1.69x at batch 16384. 256 is where they cross.
BITPLAN_BATCH_MIN = 256


def supported_auto_sizes() -> list[int]:
    """Every N this dispatcher accepts, ascending."""
    sizes, n = [], MIN_AUTO_N
    while n <= MAX_AUTO_N:
        sizes.append(n)
        n *= 2
    return sizes


def pick_implementation(n: int, batch: int | None = None) -> str:
    """'onesided' or 'cube' -- the faster kernel at this N (and batch).

    N decides everywhere except N=512, where the two-sided kernel's bit-plan
    tile moved the crossover into the middle of the batch range; see
    BITPLAN_BATCH_MIN. Omitting `batch` keeps the small-batch answer, which is
    the conservative one.
    """
    n = int(n)
    if not MIN_AUTO_N <= n <= MAX_AUTO_N:
        raise ValueError(
            f"n must be in [{MIN_AUTO_N}, {MAX_AUTO_N}]; got {n}. Below "
            f"{MIN_AUTO_N} only the vector kernel in standard/ applies; above "
            f"{MAX_AUTO_N} no kernel here does -- the two-sided split would "
            "need an accumulator larger than L0C."
        )
    if n & (n - 1):
        raise ValueError(f"n must be a power of two; got {n}.")
    if n > ONESIDED_MAX_N:
        return "cube"
    if n == ONESIDED_MAX_N and batch is not None and int(batch) >= BITPLAN_BATCH_MIN:
        return "cube"
    return "onesided"


def jit_compile(
    base_dir=None,
    verbose: bool = True,
    clean_up: bool = False,
    device: str | int = DEFAULT_DEVICE,
):
    """Compile on demand and return one `fn(x, y)` that routes on N.

    Each implementation is compiled the first time an N that needs it is seen,
    not up front: both are cube-arch kernels but they are separate translation
    units, and a caller that only ever uses one size should only pay for one
    compile.
    """
    resolved_device = normalize_npu_device(device)
    base = Path(base_dir) if base_dir else Path(__file__).resolve().parents[1]
    builders = {
        "onesided": (
            jit_compile_onesided,
            base / "onesided" / "fast_hadamard_onesided.cpp",
        ),
        "cube": (jit_compile_cube, base / "cube" / "fast_hadamard_cube.cpp"),
    }
    compiled: dict = {}

    def kernel_for(impl: str):
        if impl not in compiled:
            build, src = builders[impl]
            compiled[impl] = build(
                str(src), verbose=verbose, clean_up=clean_up,
                device=resolved_device,
            )
        return compiled[impl]

    def hadamard_func(
        x,
        y,
        batch=None,
        n=None,
        block_dim=None,
        stream_ptr=None,
    ):
        """Y = X * H_N, out of place, for 2D contiguous fp16 [batch, n].

        Both back ends take this signature and validate the tensors themselves,
        so the only thing done here is the choice between them. Each applies its
        own block_dim schedule unless one is passed.
        """
        if x.dim() != 2:
            raise ValueError(
                f"x must be 2D with shape [batch, n]; got {tuple(x.shape)}."
            )
        resolved_n = int(x.shape[1])
        impl = pick_implementation(resolved_n, int(x.shape[0]))
        return kernel_for(impl)(
            x, y, batch=batch, n=n, block_dim=block_dim, stream_ptr=stream_ptr,
        )

    hadamard_func.pick_implementation = pick_implementation
    hadamard_func.compiled = compiled
    hadamard_func.kernel_for = kernel_for
    return hadamard_func


# Re-exported so callers can check a size without compiling anything.
__all__ = [
    "MIN_AUTO_N",
    "MAX_AUTO_N",
    "ONESIDED_MAX_N",
    "BITPLAN_BATCH_MIN",
    "MIN_CUBE_HADAMARD_N",
    "jit_compile",
    "pick_implementation",
    "supported_auto_sizes",
]
