import ctypes
import math
import os

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

# Y = X * H_N as a single right-multiplication, with H_N factored
# H_B (x) H_M and applied as fixed-size M x M matmuls accumulated in L0C.
# N must be 16-aligned for the cube. 512 is a COMPUTE crossover, not a capacity
# limit -- see the kernel header and validate_onesided_hadamard_args.
MIN_ONESIDED_N = 64
MAX_ONESIDED_N = 512

# A tile's Rows is S here, and PTO requires 16-aligned rows (TMOV's TExtract
# rejects even Rows == 1), so the kernel's smallest tile is 16 samples and it
# needs batch >= 16. Shorter batches are padded up on the host.
MIN_ONESIDED_BATCH = 16

_H_CACHE: dict = {}
_PAD_CACHE: dict = {}

# Samples per tile, mirroring FAST_HADAMARD_ONESIDED_CASES in the kernel. Only
# N <= 128 follows S * N == PTO_OS_TILE_ELEMS; 256 and 512 run the blocked form
# and override it with a SMALLER tile (see PTO_OS_S512). Keep in step with the
# kernel -- the tests import this to place their tile-boundary batches.
_TILE_ELEMS = 16384
_SAMPLES_PER_TILE = {64: 256, 128: 128, 256: 32, 512: 16}

# Same shape of schedule as the two-sided kernel: cores are only worth adding
# while each still gets a tile run long enough to keep its pipes full, and
# bd = sqrt(k * tiles) trades those off.
#
# REFITTED for the blocked form. The old constants (8, 16, 16, 16) were fitted
# when the kernel was movement-bound at N>=256 and ran a 2-4x taller tile; the
# blocked form is matmul-bound there and tiles are smaller, so both the tile
# count and the per-tile cost moved. Resweep this whenever the tile geometry or
# PTO_OS_BLOCK changes -- it is calibration, not a rule.
#
# Fitted per N over EVERY integer block_dim in [1, 24] x batch in
# {64, 256, 1024, 8192, 65536}, msprof device time on 910B2, 15 reps each.
# Regret is against the best measured block_dim at that shape:
#
#    N      k    worst regret   mean    constant-24 worst
#   64    7.51      1.041x     1.012x        1.192x
#  128    7.23      1.021x     1.006x        1.176x
#  256    7.51      1.000x     1.000x        1.214x
#  512   22.57      1.028x     1.014x        1.181x
#
# Sweep every integer, not a subset. A first fit over
# {1,2,3,4,6,8,12,16,20,24} chose constants that emit bd = 15 and 23 -- values
# it had never measured -- and scored them against their nearest neighbour. The
# regret it reported was 1.013-1.031x; the real cost at N=64 batch 8192 was
# 1.12x, because 32 tiles divide evenly over 16 cores and badly over 15.
_TILE_RUN_K = {64: 7.51, 128: 7.23, 256: 7.51, 512: 22.57}
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


# Block size M of the kernel's blocked-accumulation form, mirroring
# PTO_OS_BLOCK. The buffer below carries +-H_M for it, so the two MUST agree:
# a mismatch hands the kernel the wrong-sized operand and it computes garbage
# (which is how the first M=64 A/B failed -- a host/device disagreement, not a
# bad block size). The env override exists so that A/B can set both at once.
_BLOCK = int(os.environ.get("PTO_OS_BLOCK", 64))


def hadamard_operands(n: int, block: int | None = None) -> tuple[int, int]:
    """(M, B) of the kernel's blocked form: H_n = H_B (x) H_M, M = min(64, n)."""
    m = min(_BLOCK if block is None else int(block), n)
    return m, n // m


def hadamard_full(n: int, device, block: int | None = None):
    """Cached device operand buffer: H_n, then H_M, then -H_M, flat.

    H_n is symmetric, so the kernel can use it as the right operand with no
    transpose, and row-major order makes each contraction chunk H[kc:kc+Kc, :]
    one contiguous descriptor whatever Kc the kernel picks.

    The +-H_M pair at offset n*n is what the blocked form multiplies by: with
    H_n = H_B (x) H_M those two are the only right operands it ever needs, since
    H_B's entries are +-1 (see the kernel header). They are APPENDED rather than
    substituted so a single host buffer feeds either form of the kernel, which
    keeps the A/B between them a compile-time switch and nothing else. The tail
    is 2*M*M <= 32768 halves whatever n is.

    Keyed on the torch device object directly: this sits on the per-call path,
    and stringifying the device costs more than the lookup it feeds.
    """
    key = (n, device, block)
    cached = _H_CACHE.get(key)
    if cached is None:
        m, _ = hadamard_operands(n, block)
        h_m = hadamard_matrix(m, device)
        cached = torch.cat(
            (hadamard_matrix(n, device).view(-1), h_m.view(-1), -h_m.view(-1))
        ).contiguous()
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
            "The one-sided product costs n*n MACs per sample however H is "
            "factored, so past 512 the MACs grow 4x per doubling while the "
            "bytes only double -- use the two-sided kernel in cube/, which is "
            "1.7x faster at 1024 and 4.8x at 2048."
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

        # Build H BEFORE resolving the stream pointer, and keep it on its own
        # line. On a cache miss hadamard_full enqueues torch ops (host->device
        # copies and a cat), while resolve_stream_ptr is what drains torch_npu's
        # async task queue so this raw ctypes launch is ordered after them.
        # Python evaluates call arguments left to right, so writing this inline
        # after the stream pointer enqueues those ops AFTER the drain and the
        # kernel can read H before it is written -- Y = X * 0 = 0, on the first
        # call for each n only, which is exactly as intermittent as it sounds.
        h = hadamard_full(resolved_n, x.device)
        kernel(
            launch_block_dim,
            resolve_stream_ptr(stream_ptr),
            torch_to_ctypes(x),
            torch_to_ctypes(y),
            torch_to_ctypes(h),
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
