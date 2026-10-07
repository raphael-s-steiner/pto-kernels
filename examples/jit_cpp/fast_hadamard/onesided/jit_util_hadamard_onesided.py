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

# Y = X * H_N as a single dense product. N must be 16-aligned for the cube, and
# the ceiling is where H stops fitting near the matmul -- see the kernel header.
MIN_ONESIDED_N = 64
MAX_ONESIDED_N = 512

# A tile's Rows is S here, and PTO requires 16-aligned rows (TMOV's TExtract
# rejects even Rows == 1), so the kernel's smallest tile is 16 samples and it
# needs batch >= 16. Shorter batches are padded up on the host.
MIN_ONESIDED_BATCH = 16

_H_CACHE: dict = {}
_PAD_CACHE: dict = {}

# Samples per tile, mirroring FAST_HADAMARD_ONESIDED_CASES in the kernel:
# S * N == PTO_OS_TILE_ELEMS, so two L0C accumulators fit 128 KB. Keep in step
# with the kernel if PTO_OS_TILE_ELEMS or the case table changes.
# N=512 runs a taller tile than TILE_ELEMS/N would give; see PTO_OS_S512.
_TILE_ELEMS = 16384
_SAMPLES_PER_TILE = {64: 256, 128: 128, 256: 64, 512: 64}

# Same shape of schedule as the two-sided kernel: cores are only worth adding
# while each still gets a tile run long enough to keep its pipes full, and
# bd = sqrt(k * tiles) trades those off.
#
# Fitted per N from a block_dim sweep in msprof device time on 910B2, over
# N x batch in {256, 1024, 8192, 65536}. Last column is what a constant 24
# costs, i.e. what the schedule buys:
#
#    N     k    worst regret   constant-24 worst
#   64     8        1.033x           1.281x
#  128    16        1.004x           1.144x
#  256    16        1.028x           1.157x
#  512    16        1.013x           1.077x
#
# The penalty for no schedule is much smaller than the two-sided kernel's 1.91x,
# because this tile body touches each pipe once and so reaches steady state after
# very few tiles -- it is far less sensitive to how thinly the work is spread.
_TILE_RUN_K = {64: 8.0, 128: 16.0, 256: 16.0, 512: 16.0}
# Only reached if the case table grows a size this table has not been fitted
# for. Then it is an extrapolation -- sweep block_dim and add the entry.
_DEFAULT_TILE_RUN_K = 8.0


def onesided_block_dim_schedule(batch: int, n: int, max_block_dim: int) -> int:
    """Pick block_dim for one launch: enough cores to use the part, but not so
    many that each core's tile run is too short to fill its pipes.

    block_dim cannot change results, so an explicit block_dim always overrides
    this; test_block_dim_does_not_change_result is the gate on that.
    """
    tiles = -(-int(batch) // _SAMPLES_PER_TILE.get(int(n), 1))
    k = _TILE_RUN_K.get(int(n), _DEFAULT_TILE_RUN_K)
    return max(1, min(int(max_block_dim), round(math.sqrt(k * tiles))))


def hadamard_matrix(size: int, device, dtype=torch.float16):
    """Sylvester (natural-order) Hadamard matrix with +-1 entries."""
    h = torch.ones((1, 1), dtype=torch.float32)
    while h.shape[0] < size:
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    return h.to(device=device, dtype=dtype).contiguous()


def hadamard_full(n: int, device):
    """Cached device copy of the whole H_n.

    H_n is symmetric, so the kernel can use it as the right operand with no
    transpose, and row-major order makes each contraction chunk H[kc:kc+Kc, :]
    one contiguous descriptor whatever Kc the kernel picks.

    Keyed on the torch device object directly: this sits on the per-call path,
    and stringifying the device costs more than the lookup it feeds.
    """
    key = (n, device)
    cached = _H_CACHE.get(key)
    if cached is None:
        cached = hadamard_matrix(n, device)
        _H_CACHE[key] = cached
    return cached


def _pad_pair(n: int, device):
    """Cached (x, y) scratch of MIN_ONESIDED_BATCH rows for short batches."""
    key = (n, device)
    cached = _PAD_CACHE.get(key)
    if cached is None:
        cached = (
            torch.zeros((MIN_ONESIDED_BATCH, n), device=device,
                        dtype=torch.float16),
            torch.empty((MIN_ONESIDED_BATCH, n), device=device,
                        dtype=torch.float16),
        )
        _PAD_CACHE[key] = cached
    return cached


def validate_onesided_hadamard_args(x, y):
    if x.dim() != 2 or y.dim() != 2:
        raise ValueError("x and y must be 2D tensors with shape [batch, n].")
    if x.shape != y.shape:
        raise ValueError("x and y must have matching shapes.")
    if x.dtype != torch.float16 or y.dtype != torch.float16:
        raise TypeError("x and y must use torch.float16.")
    if not x.is_contiguous() or not y.is_contiguous():
        raise ValueError("x and y must be contiguous.")

    n = int(x.shape[1])
    if n < MIN_ONESIDED_N or n > MAX_ONESIDED_N:
        raise ValueError(
            f"n must be in [{MIN_ONESIDED_N}, {MAX_ONESIDED_N}]; got {n}. "
            "The one-sided product reads all of H per tile, so beyond 512 it "
            "no longer fits near the matmul -- use the two-sided kernel in "
            "cube/ instead."
        )
    if n & (n - 1):
        raise ValueError(f"n must be a power of two; got {n}.")
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
            ctypes.c_void_p,  # h  (N x N, row-major)
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
        resolved_batch, resolved_n = validate_onesided_hadamard_args(x, y)
        if batch is not None and int(batch) != resolved_batch:
            raise ValueError("batch must match the input tensor shape.")
        if n is not None and int(n) != resolved_n:
            raise ValueError("n must match the input tensor shape.")

        if resolved_batch == 0:
            return
        if resolved_batch < MIN_ONESIDED_BATCH:
            # The smallest tile is 16 samples and the kernel's tail trick backs a
            # tile up to `end`, which needs batch >= 16. Run the padded scratch
            # instead; the rows past `batch` are zeros and are discarded.
            px, py = _pad_pair(resolved_n, x.device)
            px[:resolved_batch].copy_(x)
            hadamard_func(px, py, block_dim=block_dim, stream_ptr=stream_ptr)
            y.copy_(py[:resolved_batch])
            return

        launch_block_dim = (
            onesided_block_dim_schedule(
                resolved_batch, resolved_n, resolved_block_dim
            )
            if block_dim is None
            else resolve_launch_block_dim(block_dim, resolved_block_dim)
        )

        kernel(
            launch_block_dim,
            resolve_stream_ptr(stream_ptr),
            torch_to_ctypes(x),
            torch_to_ctypes(y),
            torch_to_ctypes(hadamard_full(resolved_n, x.device)),
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
