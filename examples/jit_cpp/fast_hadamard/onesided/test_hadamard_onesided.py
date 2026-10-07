import pytest
import torch

from onesided.jit_util_hadamard_onesided import (
    MAX_ONESIDED_N,
    MIN_ONESIDED_BATCH,
    MIN_ONESIDED_N,
    hadamard_full,
    hadamard_matrix,
    onesided_block_dim_schedule,
    validate_onesided_hadamard_args,
)
# Deliberately the wrapper's own table rather than TILE_ELEMS // N: S is a
# per-N override in the kernel (N=512 runs a taller tile than the capacity
# formula gives), and a test that recomputed it would silently start probing
# the wrong boundaries the next time that override moves.
from onesided.jit_util_hadamard_onesided import _SAMPLES_PER_TILE

# The one-sided kernel reads the whole of H per tile, so its case table stops at
# 512; everything larger belongs to the two-sided kernel in cube/.
TEST_HIDDEN_DIMS = [64, 128, 256, 512]
# Straddles both boundaries that matter: the S-sample tile (256/128/64/64
# across the grid) and the 16-sample tail the kernel mops up by backing one more
# tile up to the end of the range. Everything below MIN_ONESIDED_BATCH goes
# through the host's padded scratch instead.
TEST_BATCHES = [1, 3, 15, 16, 17, 31, 32, 33, 48, 63, 64, 65, 127, 128, 255,
                300, 1000]
TEST_SEEDS = [0, 1]
SHORT_BATCHES = list(range(1, MIN_ONESIDED_BATCH))
MAX_BLOCK_DIM = 24
DTYPE = torch.float16

# Exactly representable in fp16 and far outside randn's range, so a row the
# kernel or the padding path must not touch is recognisable byte for byte.
SENTINEL = -7.0

# The one-sided kernel is a single dense matmul: it accumulates in fp32 and
# rounds to fp16 exactly ONCE, against twice for the two-sided split in cube/
# and log2(N) times for the vector butterfly. It therefore does not reproduce
# any staged fp16 reference bit for bit and is compared against an exact float64
# transform instead. Measured worst case over the grid below is 2.43e-4 relative
# RMS (at batch=1, N=64, where a single sample makes the RMS noisy) and 1.88e-3
# for the worst element, both relative to the output RMS; the tolerances below
# keep 4-5x of margin and match the two-sided suite's.
#
# Head to head at batch=128 the one-sided error is 0.71x the two-sided kernel's
# (2.10e-4 against 2.96e-4 at N=256, 2.07e-4 against 2.93e-4 at N=512), so
# test_more_accurate_than_two_sided passes with room, not marginally.
REL_RMS_TOL = 1e-3
MAX_ELEMENT_TOL = 1e-2

# test_more_accurate_than_two_sided compares two measured error figures, so it
# allows a few percent of slack rather than asserting a strict ordering on
# numbers that move with the input.
ACCURACY_SLACK = 1.05


def fwht_f64(x):
    """Exact natural-order (Sylvester) Walsh-Hadamard transform in float64.

    Everything runs on the host: the NPU has no float64 and silently falls
    back to float32 for it.
    """
    a = x.cpu().double().clone()
    b, n = a.shape
    h = 1
    while h < n:
        a = a.reshape(b, n // (2 * h), 2, h)
        u = a[:, :, 0, :].clone()
        v = a[:, :, 1, :].clone()
        a = torch.stack((u + v, u - v), dim=2).reshape(b, n)
        h *= 2
    return a


def samples_per_tile(n: int) -> int:
    """S for this N, as the wrapper mirrors FAST_HADAMARD_ONESIDED_CASES.

    Not `.get(n, 1)` like the schedule's own lookup: a size missing from the
    table would quietly turn every boundary below into batch=1, so fail loudly
    instead.
    """
    assert n in _SAMPLES_PER_TILE, (
        f"N={n} is in the test grid but not in the kernel's tile table "
        f"{sorted(_SAMPLES_PER_TILE)}"
    )
    return _SAMPLES_PER_TILE[n]


def relative_errors(actual, exact):
    """(relative RMS, worst element) of `actual`, both scaled by the exact RMS."""
    rms = exact.pow(2).mean().sqrt()
    diff = actual - exact
    return (
        (diff.pow(2).mean().sqrt() / rms).item(),
        (diff.abs().max() / rms).item(),
    )


def run_onesided_hadamard(onesided_hadamard_kernel, x, **kwargs):
    y = torch.full_like(x, SENTINEL)
    onesided_hadamard_kernel(x, y, **kwargs)
    torch.npu.synchronize()
    return y


def assert_matches_exact(y, x_cpu, context):
    actual = y.cpu().double()
    assert torch.isfinite(actual).all(), f"non-finite output for {context}"
    if actual.numel() == 0:
        return

    exact = fwht_f64(x_cpu)
    rel_rms, max_element = relative_errors(actual, exact)
    assert rel_rms <= REL_RMS_TOL, (
        f"relative RMS {rel_rms:.3e} exceeds {REL_RMS_TOL:.1e} for {context}"
    )
    assert max_element <= MAX_ELEMENT_TOL, (
        f"worst element {max_element:.3e} exceeds {MAX_ELEMENT_TOL:.1e} "
        f"for {context}"
    )


def check_shape(onesided_hadamard_kernel, npu_device, batch, n, seed=0):
    torch.manual_seed(seed)
    x_cpu = torch.randn(batch, n, dtype=DTYPE)
    y = run_onesided_hadamard(onesided_hadamard_kernel, x_cpu.to(npu_device))
    assert_matches_exact(y, x_cpu, f"seed={seed}, batch={batch}, N={n}")


@pytest.mark.parametrize("seed", TEST_SEEDS)
@pytest.mark.parametrize("batch", TEST_BATCHES)
@pytest.mark.parametrize("n", TEST_HIDDEN_DIMS)
def test_onesided_hadamard_correctness(
    onesided_hadamard_kernel, npu_device, seed, batch, n
):
    check_shape(onesided_hadamard_kernel, npu_device, batch, n, seed=seed)


@pytest.mark.parametrize("n", TEST_HIDDEN_DIMS)
def test_sample_tile_boundaries(onesided_hadamard_kernel, npu_device, n):
    """The kernel halves its sample tile on the way down to 16, so straddle
    every level it can stop at for this N."""
    s = samples_per_tile(n)
    batches = sorted(
        {
            b
            for step in (s, s // 2, s // 4, 16)
            for b in (step - 1, step, step + 1, 2 * step, 2 * step + 1)
            if b >= 1
        }
    )
    for batch in batches:
        check_shape(onesided_hadamard_kernel, npu_device, batch, n)


@pytest.mark.parametrize("n", TEST_HIDDEN_DIMS)
def test_backed_up_tail(onesided_hadamard_kernel, npu_device, n):
    """batch % 16 != 0: the last 1..15 samples are covered by backing one more
    16-sample tile up to the end, so some samples are recomputed and stored
    twice. Both stores must write the same bytes."""
    s = samples_per_tile(n)
    batches = sorted(
        {
            b
            for b in (17, 23, 31, 33, 47, s - 1, s + 1, s + 15, 2 * s + 3, 999)
            if b >= MIN_ONESIDED_BATCH and b % 16 != 0
        }
    )
    assert batches, f"no tail batches for N={n}"
    for batch in batches:
        check_shape(onesided_hadamard_kernel, npu_device, batch, n)


@pytest.mark.parametrize("n", TEST_HIDDEN_DIMS)
def test_batch_partition_boundaries(onesided_hadamard_kernel, npu_device, n):
    """The launch splits whole tiles over the cores, so cross the point where
    cores start running out of work (and the point where there is none)."""
    s = samples_per_tile(n)
    cores = onesided_hadamard_kernel.block_dim
    batches = sorted(
        {0, 1, s, s * (cores - 1), s * cores, s * cores + 1, s * cores + 17}
    )
    for batch in batches:
        check_shape(onesided_hadamard_kernel, npu_device, batch, n)


@pytest.mark.parametrize("n", TEST_HIDDEN_DIMS)
def test_short_batch_padding(onesided_hadamard_kernel, npu_device, n):
    """batch 1..15 is below the kernel's smallest tile, so the wrapper runs a
    cached 16-row scratch instead. The result must still be right, and the rows
    past `batch` in the caller's buffer must be left alone."""
    rows = MIN_ONESIDED_BATCH + 4
    torch.manual_seed(11)
    x_cpu = torch.randn(rows, n, dtype=DTYPE)
    x = x_cpu.to(npu_device)

    for batch in SHORT_BATCHES:
        y = torch.full((rows, n), SENTINEL, device=npu_device, dtype=DTYPE)
        x_in, y_out = x[:batch], y[:batch]
        assert x_in.is_contiguous() and y_out.is_contiguous()

        onesided_hadamard_kernel(x_in, y_out)
        torch.npu.synchronize()

        assert_matches_exact(y_out, x_cpu[:batch], f"padded batch={batch}, N={n}")
        untouched = y[batch:].cpu()
        assert torch.equal(
            untouched, torch.full_like(untouched, SENTINEL)
        ), f"padding path wrote past batch={batch} for N={n}"


@pytest.mark.parametrize("n", [64, 512])
def test_short_batch_padding_scratch_is_not_stale(
    onesided_hadamard_kernel, npu_device, n
):
    """The scratch is cached across calls, so rows [batch, 16) hold the previous
    call's data. Walk the batch down so every call inherits a longer stale tail
    than it writes."""
    for batch in (15, 1, 9, 2, 14, 3):
        check_shape(onesided_hadamard_kernel, npu_device, batch, n, seed=batch)


@pytest.mark.parametrize("n", [64, 512])
def test_input_is_not_modified(onesided_hadamard_kernel, npu_device, n):
    # 9 goes through the padded scratch, 100 straight to the kernel.
    for batch in (9, 100):
        torch.manual_seed(0)
        x_cpu = torch.randn(batch, n, dtype=DTYPE)
        x = x_cpu.to(npu_device)

        run_onesided_hadamard(onesided_hadamard_kernel, x)

        assert torch.equal(x.cpu(), x_cpu), (
            f"kernel modified its input for batch={batch}, N={n}"
        )


@pytest.mark.parametrize("n", [256, 512])
def test_more_accurate_than_two_sided(request, onesided_hadamard_kernel, npu_device, n):
    """Rounding once instead of twice is the point of the one-sided path.

    The two-sided kernel needs N = R * C with both factors 16-aligned, so this
    only has a comparison arm at N >= 256.
    """
    try:
        cube_hadamard_kernel = request.getfixturevalue("cube_hadamard_kernel")
    except Exception as exc:  # pragma: no cover - depends on the environment
        pytest.skip(f"two-sided kernel unavailable: {exc}")

    torch.manual_seed(0)
    x_cpu = torch.randn(128, n, dtype=DTYPE)
    x = x_cpu.to(npu_device)
    exact = fwht_f64(x_cpu)

    onesided_rel_rms, _ = relative_errors(
        run_onesided_hadamard(onesided_hadamard_kernel, x).cpu().double(), exact
    )

    cube_y = torch.empty_like(x)
    cube_hadamard_kernel(x, cube_y)
    torch.npu.synchronize()
    cube_rel_rms, _ = relative_errors(cube_y.cpu().double(), exact)

    assert onesided_rel_rms <= cube_rel_rms * ACCURACY_SLACK, (
        f"one-sided relative RMS {onesided_rel_rms:.3e} is worse than the "
        f"two-sided kernel's {cube_rel_rms:.3e} for N={n}"
    )


def test_hadamard_full_is_cached_and_orthogonal(npu_device):
    """H is handed to the kernel as the right operand with no transpose, which
    only gives the natural-order transform because H is symmetric."""
    h = hadamard_full(256, torch.device(npu_device))
    assert hadamard_full(256, torch.device(npu_device)) is h, "H is not cached"
    assert h.dtype == DTYPE and h.is_contiguous()

    h64 = hadamard_matrix(64, "cpu", dtype=torch.float64)
    assert torch.equal(h64, h64.T), "H must be symmetric"
    assert torch.equal(h64 @ h64, 64 * torch.eye(64, dtype=torch.float64))


# --- argument rejection ---------------------------------------------------


@pytest.mark.parametrize("n", [MIN_ONESIDED_N // 2, MAX_ONESIDED_N * 2])
def test_out_of_range_n_rejected(npu_device, n):
    x = torch.randn(32, n, device=npu_device, dtype=DTYPE)
    y = torch.empty_like(x)

    with pytest.raises(ValueError, match="n must be in"):
        validate_onesided_hadamard_args(x, y)


def test_non_power_of_two_n_rejected(npu_device):
    x = torch.randn(32, 192, device=npu_device, dtype=DTYPE)
    y = torch.empty_like(x)

    with pytest.raises(ValueError, match="power of two"):
        validate_onesided_hadamard_args(x, y)


def test_wrong_dtype_rejected(npu_device):
    x = torch.randn(32, 256, device=npu_device, dtype=torch.float32)
    y = torch.empty_like(x)

    with pytest.raises(TypeError, match="torch.float16"):
        validate_onesided_hadamard_args(x, y)


def test_noncontiguous_input_rejected(npu_device):
    x = torch.randn(256, 32, device=npu_device, dtype=DTYPE).transpose(0, 1)
    y = torch.empty(32, 256, device=npu_device, dtype=DTYPE)
    assert x.shape == y.shape
    assert not x.is_contiguous()

    with pytest.raises(ValueError, match="contiguous"):
        validate_onesided_hadamard_args(x, y)


def test_shape_mismatch_rejected(npu_device):
    x = torch.randn(32, 256, device=npu_device, dtype=DTYPE)
    y = torch.empty(33, 256, device=npu_device, dtype=DTYPE)

    with pytest.raises(ValueError, match="matching shapes"):
        validate_onesided_hadamard_args(x, y)


# --- block_dim schedule ---------------------------------------------------
#
# block_dim is a per-launch argument and the best value moves with the shape:
# too many cores leaves each one a tile run too short to fill its pipes. The
# regret for a constant 24 is smaller here than for the two-sided kernel (1.28x
# worst against 1.91x) because this tile body touches each pipe once and so
# reaches steady state after very few tiles -- but it must not change the
# result, which is what these pin down.


def test_block_dim_schedule_in_range():
    for n in TEST_HIDDEN_DIMS:
        for batch in (1, 7, 64, 256, 1024, 8192, 65536):
            bd = onesided_block_dim_schedule(batch, n, MAX_BLOCK_DIM)
            assert 1 <= bd <= MAX_BLOCK_DIM, (batch, n, bd)


def test_block_dim_schedule_grows_with_batch():
    """More tiles justify more cores, so the schedule must be monotone."""
    for n in TEST_HIDDEN_DIMS:
        seq = [
            onesided_block_dim_schedule(b, n, MAX_BLOCK_DIM)
            for b in (1, 64, 256, 1024, 4096, 8192, 65536)
        ]
        assert seq == sorted(seq), (n, seq)


def test_block_dim_schedule_respects_cap():
    assert onesided_block_dim_schedule(1 << 20, 256, 4) == 4
    assert onesided_block_dim_schedule(1 << 20, 256, 1) == 1


@pytest.mark.parametrize("n", TEST_HIDDEN_DIMS)
def test_block_dim_does_not_change_result(onesided_hadamard_kernel, npu_device, n):
    """The gate that makes the schedule free of correctness risk: every
    block_dim, including the scheduled one, must give bit-identical output.

    block_dim moves both the per-core range boundaries and which tile size each
    sample lands in, and the backed-up tail makes neighbouring ranges overlap,
    so this is the test that the overlap really is byte-identical.
    """
    torch.manual_seed(7)
    batch = 300  # 300 % 16 == 12, so no tile size divides it on any core
    assert batch % 16 != 0 and batch % samples_per_tile(n) != 0
    x = torch.randn(batch, n, device=npu_device, dtype=DTYPE)

    reference = torch.empty_like(x)
    onesided_hadamard_kernel(x, reference, block_dim=MAX_BLOCK_DIM)
    torch.npu.synchronize()

    scheduled = onesided_block_dim_schedule(batch, n, MAX_BLOCK_DIM)
    for block_dim in sorted({1, 2, 3, 5, 8, 13, MAX_BLOCK_DIM, scheduled}):
        y = torch.full_like(x, float("nan"))
        onesided_hadamard_kernel(x, y, block_dim=block_dim)
        torch.npu.synchronize()
        assert torch.equal(y, reference), f"block_dim={block_dim} changed the result"
