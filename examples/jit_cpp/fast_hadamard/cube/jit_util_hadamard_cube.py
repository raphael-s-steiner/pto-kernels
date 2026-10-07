import ctypes

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
        block_dim=resolved_block_dim,
        stream_ptr=None,
    ):
        resolved_batch, resolved_n = validate_cube_hadamard_args(x, y)
        if batch is not None and int(batch) != resolved_batch:
            raise ValueError("batch must match the input tensor shape.")
        if n is not None and int(n) != resolved_n:
            raise ValueError("n must match the input tensor shape.")

        h_left, h_right = hadamard_factors(resolved_n, x.device)
        kernel(
            resolve_launch_block_dim(block_dim, resolved_block_dim),
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
