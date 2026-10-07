import ctypes
import math

import torch

from jit_util_common import (
    CUBE_ARCH,
    DEFAULT_DEVICE,
    get_cube_block_dim,
    jit_compile_with_loader,
    load_cdll,
    load_required_symbol,
    normalize_npu_device,
    resolve_launch_block_dim,
    resolve_stream_ptr,
    torch_to_ctypes,
)

# The cube path needs N = R * C with both factors 16-aligned, so N >= 256.
MIN_CUBE_HADAMARD_N = 256
MAX_CUBE_HADAMARD_N = 16384

_FACTOR_CACHE: dict = {}

# Samples per tile, mirroring FAST_HADAMARD_CUBE_CASES in the kernel. A tile is
# S consecutive samples, so a launch has ceil(batch / S) tiles to spread over
# the cores. Keep in step with the kernel if that table changes.
_SAMPLES_PER_TILE = {256: 32, 512: 16, 1024: 8, 2048: 4, 4096: 2, 8192: 1,
                     16384: 1}

# Cores are only worth adding while each still gets a tile run long enough to
# keep its pipes full. Measured on 910B2: at N=256 a core moves 94 GB/s with
# only 2 tiles queued against 359 GB/s on a long run, so spreading 48 tiles over
# 24 cores is 1.87x SLOWER than over 8. bd = sqrt(k * tiles) trades the two off,
# and k is where a size sits on that curve: small k means high per-tile latency,
# so few cores with long runs.
#
# Fitted per N rather than by a rule, because the two R == 16 sizes do not
# behave like the rest and nothing about N predicts which way: N=512 wants k=1
# where N=1024 wants 8, and assuming 8 for it cost 1.51x. N=512's tile is 16
# TEXTRACTs plus 16 TSTOREs, the most per-tile work in the table, which is the
# likely reason -- but it was found by measuring, not by that argument.
#
# Last column is what the old constant 24 cost at that size, i.e. what the
# schedule buys:
#
#     N      k     shapes   worst regret   constant-24 worst
#   256   0.49        6         1.025x           1.834x
#   512   1.00        6         1.005x           1.908x
#  1024   8.00        6         1.000x           1.207x
#  2048  16.00        6         1.000x           1.085x
#  4096   8.00        6         1.000x           1.093x
#  8192   8.00        5         1.000x           1.000x
# 16384   8.00        5         1.000x           1.000x
_TILE_RUN_K = {256: 0.49, 512: 1.0, 1024: 8.0, 2048: 16.0, 4096: 8.0,
               8192: 8.0, 16384: 8.0}
# Only reached if FAST_HADAMARD_CUBE_CASES grows a size this table has not been
# fitted for. Then it is an extrapolation -- sweep block_dim and add the entry.
_DEFAULT_TILE_RUN_K = 8.0


def cube_block_dim_schedule(batch: int, n: int, max_block_dim: int) -> int:
    """Pick block_dim for one launch: enough cores to use the part, but not so
    many that each core's tile run is too short to fill its pipes.

    Fitted to block_dim sweeps in msprof device time on 910B2 covering every N in
    the kernel's case table x batch in {64, 256, 1024, 4096, 8192, 65536} -- 40
    shapes, no extrapolation. Lands within 1.025x of the best measured block_dim
    (mean 1.001x), against 1.91x worst / 1.19x mean for a constant 24.

    block_dim cannot change results, so an explicit block_dim always overrides
    this; test_block_dim_does_not_change_result is the gate on that.
    """
    tiles = -(-int(batch) // _SAMPLES_PER_TILE.get(int(n), 1))
    k = _TILE_RUN_K.get(int(n), _DEFAULT_TILE_RUN_K)
    return max(1, min(int(max_block_dim), round(math.sqrt(k * tiles))))


def hadamard_split(n: int) -> tuple[int, int]:
    """Balanced Kronecker split H_n = H_rows (x) H_cols, rows <= cols."""
    log2_n = n.bit_length() - 1
    rows_log2 = log2_n // 2
    return 1 << rows_log2, 1 << (log2_n - rows_log2)


def hadamard_matrix(size: int, device, dtype=torch.float16):
    """Sylvester (natural-order) Hadamard matrix with +-1 entries."""
    h = torch.ones((1, 1), dtype=torch.float32)
    while h.shape[0] < size:
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    return h.to(device=device, dtype=dtype).contiguous()


def hadamard_factors(n: int, device):
    """Cached (H_rows, H_cols) device tensors for the balanced split of H_n.

    Keyed on the torch device object directly: this sits on the per-call path,
    and stringifying the device costs more than the lookup it feeds.
    """
    key = (n, device)
    cached = _FACTOR_CACHE.get(key)
    if cached is None:
        rows, cols = hadamard_split(n)
        cached = (
            hadamard_matrix(rows, device),
            hadamard_matrix(cols, device),
        )
        _FACTOR_CACHE[key] = cached
    return cached


def validate_cube_hadamard_args(x, y):
    if x.dim() != 2 or y.dim() != 2:
        raise ValueError("x and y must be 2D tensors with shape [batch, n].")
    if x.shape != y.shape:
        raise ValueError("x and y must have matching shapes.")
    if x.dtype != torch.float16 or y.dtype != torch.float16:
        raise TypeError("x and y must use torch.float16.")
    if not x.is_contiguous() or not y.is_contiguous():
        raise ValueError("x and y must be contiguous.")
    if x.device != y.device:
        raise ValueError("x and y must be on the same device.")

    n = int(x.shape[1])
    if n & (n - 1):
        raise ValueError(f"n must be a power of two, got {n}.")
    if not MIN_CUBE_HADAMARD_N <= n <= MAX_CUBE_HADAMARD_N:
        raise ValueError(
            f"n must be in [{MIN_CUBE_HADAMARD_N}, {MAX_CUBE_HADAMARD_N}], got {n}."
        )
    return int(x.shape[0]), n


def load_lib(lib_path, block_dim=None, device: str | int = DEFAULT_DEVICE):
    lib = load_cdll(lib_path)
    resolved_block_dim = max(
        1, int(block_dim) if block_dim is not None else get_cube_block_dim(device)
    )

    kernel = load_required_symbol(
        lib,
        "call_kernel",
        [
            ctypes.c_uint32,  # blockDim
            ctypes.c_void_p,  # stream
            ctypes.c_void_p,  # x
            ctypes.c_void_p,  # y
            ctypes.c_void_p,  # h_left  (H_rows,  R x R)
            ctypes.c_void_p,  # h_right (H_cols,  C x C)
            ctypes.c_uint32,  # batch
            ctypes.c_uint32,  # n
        ],
    )

    def hadamard_func(
        x,
        y,
        batch=None,
        n=None,
        block_dim=None,
        stream_ptr=None,
    ):
        resolved_batch, resolved_n = validate_cube_hadamard_args(x, y)
        if batch is not None and int(batch) != resolved_batch:
            raise ValueError("batch must match the input tensor shape.")
        if n is not None and int(n) != resolved_n:
            raise ValueError("n must match the input tensor shape.")

        # block_dim is a per-launch argument, not a property of the compiled
        # kernel: the best value moves with the shape, and a constant costs up
        # to 1.83x at the small end of this kernel's own benchmark range.
        launch_block_dim = (
            cube_block_dim_schedule(resolved_batch, resolved_n, resolved_block_dim)
            if block_dim is None
            else resolve_launch_block_dim(block_dim, resolved_block_dim)
        )

        h_left, h_right = hadamard_factors(resolved_n, x.device)
        kernel(
            launch_block_dim,
            resolve_stream_ptr(stream_ptr),
            torch_to_ctypes(x),
            torch_to_ctypes(y),
            torch_to_ctypes(h_left),
            torch_to_ctypes(h_right),
            resolved_batch,
            resolved_n,
        )

    hadamard_func.block_dim = resolved_block_dim
    return hadamard_func


def jit_compile(
    src_path,
    verbose=True,
    clean_up=False,
    so_dir=None,
    device: str | int = DEFAULT_DEVICE,
):
    resolved_device = normalize_npu_device(device)
    return jit_compile_with_loader(
        src_path,
        lambda lib_path, block_dim=None: load_lib(
            lib_path, block_dim=block_dim, device=resolved_device
        ),
        verbose=verbose,
        clean_up=clean_up,
        so_dir=so_dir,
        device=resolved_device,
        arch=CUBE_ARCH,
    )
