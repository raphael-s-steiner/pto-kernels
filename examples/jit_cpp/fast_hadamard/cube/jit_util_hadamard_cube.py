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

# The cube path needs N = R * C with both factors 16-aligned, so N >= 256.
MIN_CUBE_HADAMARD_N = 256
MAX_CUBE_HADAMARD_N = 32768

_FACTOR_CACHE: dict = {}

# Samples per tile, mirroring FAST_HADAMARD_CUBE_CASES in the kernel. A tile is
# S consecutive samples, so a launch has ceil(batch / S) tiles to spread over
# the cores. Keep in step with the kernel if that table changes.
_SAMPLES_PER_TILE = {256: 32, 512: 16, 1024: 8, 2048: 4, 4096: 2, 8192: 1,
                     16384: 1, 32768: 1}

# Cores are only worth adding while each still gets a tile run long enough to
# keep its pipes full. Measured on 910B2: at N=256 a core moves 94 GB/s with
# only 2 tiles queued against 359 GB/s on a long run, so spreading 48 tiles over
# 24 cores is 1.87x SLOWER than over 8. bd = sqrt(k * tiles) trades the two off,
# and k is where a size sits on that curve: small k means high per-tile latency,
# so few cores with long runs.
#
# Fitted per N rather than by a rule, because nothing about N predicts which
# way a size behaves: N=512 wants k well below 1 where N=1024 wants 8, and
# assuming 8 for it cost 1.51x.
#
# REFITTED for the two-phase deep path now that it runs at every size (see
# the two-phase tile body in the kernel). The deep body changes per-tile cost, so the
# shallow-fitted constants went stale: they cost the deep path up to 1.05x at
# N=512 and 1.045x at N=2048. Only those two moved; 1024, 4096 and 8192 were
# already within 0.005x of their own optimum.
#
# Fitted over EVERY integer block_dim in [1, 24] x batch in {64, 256, 1024,
# 8192}, msprof device time on 910B2, minimising worst-case regret against the
# best measured block_dim and then mean regret as a tiebreak. Sweep every
# integer, not a subset: a subset lets the fit emit values it never measured
# and score them against a neighbour.
#
#     N      k     worst regret   mean    constant-24 worst
#   256   0.49         1.025x       --         1.834x
#   512   0.60         1.016x     1.004x       1.908x
#  1024   8.00         1.000x     1.000x       1.207x
#  2048   8.27         1.029x     1.007x       1.085x
#  4096   8.00         1.000x     1.000x       1.093x
#  8192   8.00         1.145x     1.036x       1.000x
# 16384   8.00            --         --        1.000x
# 32768  16.54         1.029x     1.007x       1.074x
#
# N=8192's 1.145x is irreducible in this schedule's form: its best block_dim
# over the four batches is 23, 24, 24, 22 -- non-monotonic, so no single k
# reproduces it. It is the one size where a lookup would beat the formula.
_TILE_RUN_K = {256: 0.49, 512: 0.6, 1024: 8.0, 2048: 8.27, 4096: 8.0,
               8192: 8.0, 16384: 8.0, 32768: 16.54}
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


# Mat-factor of each stage, mirroring PTO_HAD_BLOCK in the kernel. Each factor
# H_f is applied as H_f = H_acc (x) H_mat with H_mat of size min(_BLOCK, f), so
# the matmul and its data slice are M-sized while the accumulation over blocks
# runs in L0C. The two MUST agree or the kernel multiplies by a wrong-sized
# operand and computes garbage; the env override exists so an A/B sets both.
_BLOCK = int(os.environ.get("PTO_HAD_BLOCK", 64))


def hadamard_mat_factor(f: int, block: int | None = None) -> int:
    """Size of the matmul sub-factor of H_f. A factor already at or below the
    block size is not split (M == f), which is the dense case."""
    return min(_BLOCK if block is None else int(block), f)


def _factor_buffer(f: int, device, block: int | None = None):
    """H_f, then +H_M and -H_M, flat -- the layout loadHadamardFactors reads.

    Appended rather than substituted so one buffer feeds either form and the
    A/B between them stays a compile-time switch. The tail is 2*M*M <= 32768
    halves whatever f is, and is unused when M == f.
    """
    m = hadamard_mat_factor(f, block)
    h_m = hadamard_matrix(m, device)
    return torch.cat(
        (hadamard_matrix(f, device).view(-1), h_m.view(-1), -h_m.view(-1))
    ).contiguous()


def hadamard_factors(n: int, device, block: int | None = None):
    """Cached (H_rows, H_cols) device operand buffers for the split of H_n.

    `block` MUST match whatever the kernel was compiled with (PTO_HAD_BLOCK):
    the buffers carry +-H_M, so a disagreement hands the kernel the wrong-sized
    operand and it computes garbage -- a broken switch, not a bad block size.
    The env default exists so one setting drives both sides.

    Keyed on the torch device object directly: this sits on the per-call path,
    and stringifying the device costs more than the lookup it feeds.
    """
    b = _BLOCK if block is None else int(block)
    key = (n, device, b)
    cached = _FACTOR_CACHE.get(key)
    if cached is None:
        rows, cols = hadamard_split(n)
        cached = (_factor_buffer(rows, device, b),
                  _factor_buffer(cols, device, b))
        _FACTOR_CACHE[key] = cached
    return cached


# --- the bit-plan tile -------------------------------------------------------
#
# A conversion-free GM transfer (Layout::NZ on both legs, one contiguous
# descriptor each) needs the kernel's view of the tile to be the NZ regrouping
# of the GM byte index. For a tile of S > 1 samples that regrouping lifts the
# high column bits above the SAMPLE index, which an ordinary H_R / H_C pair
# cannot undo -- see the note above PTO_HAD_BLOCK in the kernel.
#
# But H is a compile-time constant, so it does not have to be a Hadamard.
# Because H_N is a tensor power of H_2, the transform applies H_2 to each index
# bit independently, and a matmul against a chosen operand applies whatever we
# like to one FIELD of the byte index. For an (A x B) NZ view,
#
#     p = [ col_hi | row_hi | row_lo | col_lo ]
#           t-1..a+4  a+3..8   7..4     3..0
#
# so the left operand owns p[4, a+4) and the right operand owns
# p[0,4) u p[a+4, t). Correctness is then pure bit accounting:
#
#     every transform bit gets H_2 exactly once; every sample bit gets I_2.
#
# The operand for a field is the tensor product, in that field's bit order, of
# H_2 for a transform bit and I_2 for a sample bit. At N=512 with a T=8192 tile
# that is OpA = I_2 (x) H_32 and OpB = I_8 (x) H_16 -- both plain compile-time
# constants, and the view (64 x 128) is byte for byte the configuration the
# kernel already runs at N=8192. So this needs no new kernel code at all: it is
# the existing S == 1 body, driven with T-sized "samples".
#
# The price is MACs: the operands span the whole index, so the cost is
# A + B = 2*sqrt(T) per value instead of 2*sqrt(N), i.e. sqrt(S) more. That is
# only worth paying where the per-tile descriptor count dominates, which is
# N=512 and nowhere else: measured 2.19-2.42x there, against a best of
# 0.98-1.12x at N=1024 through 4096 swept over every tile size. See the README.
_BITPLAN_TILE = {512: 8192}
_BITPLAN_CACHE: dict = {}


def bitplan_view(n: int, tile: int):
    """(A, B, row_bits, col_bits) for the (A x B) NZ view of a `tile`-element tile.

    `a` is balanced so that A + B -- the MAC cost per value -- is minimal, and
    clamped so both operands stay within L0A/L0B (<= 128 wide for fp16).
    """
    t = tile.bit_length() - 1
    a = min(max(t // 2, max(4, t - 7)), min(7, t - 4))
    A, B = 1 << a, 1 << (t - a)
    if A * B != tile:
        raise ValueError(f"no valid view for tile={tile}")
    row_bits = list(range(a + 3, 7, -1)) + list(range(7, 3, -1))
    col_bits = list(range(t - 1, a + 3, -1)) + list(range(3, -1, -1))
    assert sorted(row_bits + col_bits) == list(range(t)), "fields must partition p"
    return A, B, row_bits, col_bits


def _bit_operand(bits, n_log2: int, device):
    """Tensor product over `bits` (high to low) of H_2 on a transform bit
    (index < n_log2, i.e. inside one sample) and I_2 on a sample bit."""
    m = torch.ones((1, 1), dtype=torch.float32)
    h2 = torch.tensor([[1.0, 1.0], [1.0, -1.0]])
    for b in bits:
        m = torch.kron(m, h2 if b < n_log2 else torch.eye(2))
    return m.to(device=device, dtype=torch.float16).contiguous().view(-1)


def bitplan_operands(n: int, device):
    """Cached (OpA, OpB) device buffers for n's bit-plan tile, or None."""
    tile = _BITPLAN_TILE.get(n)
    if tile is None:
        return None
    key = (n, device)
    cached = _BITPLAN_CACHE.get(key)
    if cached is None:
        A, B, row_bits, col_bits = bitplan_view(n, tile)
        n_log2 = n.bit_length() - 1
        cached = (_bit_operand(row_bits, n_log2, device),
                  _bit_operand(col_bits, n_log2, device), tile // n)
        _BITPLAN_CACHE[key] = cached
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

        # Build every device buffer BEFORE resolving a stream pointer, and
        # resolve one per launch. A raw ctypes launch can jump ahead of torch
        # ops issued after the pointer was read, so a hoisted pointer races the
        # operand tensors into existence -- which shows up only on the FIRST
        # call at each N, as a moving set of wrong answers.
        plan = bitplan_operands(resolved_n, x.device)
        h_left, h_right = hadamard_factors(resolved_n, x.device)
        bulk = 0
        if plan is not None:
            op_a, op_b, samples_per_tile = plan
            # Whole bit-plan tiles only; a ragged remainder goes to the classic
            # path below. The tile is `samples_per_tile` samples seen as ONE
            # T-element row, which is exactly what the kernel's S == 1 body
            # wants -- so this is a plain reinterpret of the same buffers.
            tiles = resolved_batch // samples_per_tile
            if tiles:
                bulk = tiles * samples_per_tile
                tile_n = samples_per_tile * resolved_n
                kernel(
                    cube_block_dim_schedule(tiles, tile_n, resolved_block_dim)
                    if block_dim is None
                    else launch_block_dim,
                    resolve_stream_ptr(stream_ptr),
                    torch_to_ctypes(x),
                    torch_to_ctypes(y),
                    torch_to_ctypes(op_a),
                    torch_to_ctypes(op_b),
                    tiles,
                    tile_n,
                )
        if bulk == resolved_batch:
            return

        rest = resolved_batch - bulk
        kernel(
            launch_block_dim if bulk == 0 else cube_block_dim_schedule(
                rest, resolved_n, resolved_block_dim),
            resolve_stream_ptr(stream_ptr),
            torch_to_ctypes(x[bulk:] if bulk else x),
            torch_to_ctypes(y[bulk:] if bulk else y),
            torch_to_ctypes(h_left),
            torch_to_ctypes(h_right),
            rest,
            resolved_n,
        )

    hadamard_func.block_dim = resolved_block_dim
    hadamard_func.bitplan_sizes = sorted(_BITPLAN_TILE)
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
