## Fast Hadamard

### Layout

- `standard/`: the vector-unit butterfly -- kernel, host wrapper, benchmark,
  plot, and test files
- `cube/`: both cube-unit (matmul) kernels, each with the same five files:
  - `fast_hadamard_twosided.cpp`: `Y = H_R * X * H_C`, the balanced Kronecker
    split, for `N` in 256..32768
  - `fast_hadamard_onesided.cpp`: `Y = X * H_N` as a single right
    multiplication, with `H_N = H_B (x) H_M` applied as fixed-size matmuls
    accumulated in L0C, for `N` in 64..512
  - plus `devtime_bench.py` / `devtime_run.py`, the msprof device-time harness
    both kernels are measured with (the event-timed benchmarks cannot see them;
    see "Measured device time")
- `auto/`: **the entry point to use.** One `fn(x, y)` that branches on `N` to
  whichever of the above is fastest, covering `N` in 64..32768
- `fuse_int8_quant/`: int8 quantize and fused Hadamard+int8 quant files
- `fuse_int4_quant/`: int4 quantize and fused Hadamard+int4 quant files
- shared helpers stay at the top level:
  - `bench_common.py`
  - `plot_common.py`
  - `jit_util_common.py`
  - `conftest.py`

### Usage

```bash
export PTO_LIB_PATH=${ASCEND_TOOLKIT_HOME}
cd examples/jit_cpp/fast_hadamard

# available generic entry points
python benchmark.py --list
python plot.py --list
python test.py --list

# benchmark and write CSV files
python benchmark.py --target standard --npu 0
# two-sided (Kronecker split) Hadamard, also timed against the vector kernel
python benchmark.py --target twosided --npu 0
# one-sided (blocked right-multiply) Hadamard over N in {64, 128, 256, 512}, timed
# against the vector kernel and, where N >= 256, the two-sided kernel
python benchmark.py --target onesided --npu 0
python benchmark.py --target quantize_int8 --npu 0
python benchmark.py --target quantize_int4 --npu 0
# run both standalone quantize suites for later int8-vs-int4 plotting
python benchmark.py --target quantize_compare_int8_int4 --npu 0
# compares fused PTO, separate PTO, and torch+torch_npu unfused
python benchmark.py --target hadamard_quant_int8 --npu 0
python benchmark.py --target hadamard_quant_int4 --npu 0
# run both fused Hadamard+quant suites for later int8-vs-int4 plotting
python benchmark.py --target hadamard_quant_compare_int8_int4 --npu 0
# or run all benchmark suites with one command
python benchmark.py --target all --npu 0

# cube vs vector over the whole grid, from msprof DEVICE time rather than the
# event timer -- the only comparison these kernels are visible in. Needs msprof
# on PATH; it is not part of --target all for that reason.
python cube/bench_cube_vs_vector.py

# generate plots from CSV files
python plot.py --target standard
# two-sided-vs-vector duration, bandwidth, and speedup plots
python plot.py --target twosided
# one-sided-vs-vector plots, plus one-sided-vs-two-sided for the N it shares
python plot.py --target onesided
python plot.py --target quantize_int8
python plot.py --target quantize_int4
# generate top-level int8-vs-int4 quantize comparison plots
python plot.py --target quantize_compare_int8_int4
python plot.py --target hadamard_quant_int8
python plot.py --target hadamard_quant_int4
# generate top-level int8-vs-int4 fused Hadamard+quant comparison plots
python plot.py --target hadamard_quant_compare_int8_int4
# or run all plot suites with one command
python plot.py --target all

# cube-vs-vector speedup heatmaps, device time and bandwidth, from the CSV
# cube/bench_cube_vs_vector.py writes
python plot_hadamard_cube_compare.py

# correctness tests
python test.py --target standard -v --npu "npu:0"
python test.py --target twosided -v --npu "npu:0"
python test.py --target onesided -v --npu "npu:0"
python test.py --target auto -v --npu "npu:0"
python test.py --target quantize_int8 -v --npu "npu:0"
python test.py --target quantize_int4 -v --npu "npu:0"
python test.py --target hadamard_quant_int8 -v --npu "npu:0"
python test.py --target hadamard_quant_int4 -v --npu "npu:0"
# or run all test suites with one command
python test.py --target all -v --npu "npu:0"
```

### Output

Benchmark scripts save CSV files into the subdirectories that own the real
implementations:

- `standard/outputs/csv/fht_pto_bd*.csv` from `benchmark.py --target standard`
- `cube/outputs/csv/fht_twosided_compare_bd*.csv` from `benchmark.py --target twosided`
- `cube/outputs/csv/fht_onesided_compare_bd*.csv` from `benchmark.py --target onesided`
- `fuse_int8_quant/outputs/csv/quantize_compare_bd*.csv` from `benchmark.py --target quantize_int8`
- `fuse_int8_quant/outputs/csv/fht_quant_compare_bd*.csv` from `benchmark.py --target hadamard_quant_int8`
- `fuse_int4_quant/outputs/csv/quantize_compare_bd*.csv` from `benchmark.py --target quantize_int4`
- `fuse_int4_quant/outputs/csv/fht_quant_compare_bd*.csv` from `benchmark.py --target hadamard_quant_int4`

Plot scripts read CSV files from the same subdirectories and save figures to:

- `standard/outputs/plots/`
- `cube/outputs/plots/`
- `fuse_int8_quant/outputs/plots/`
- `fuse_int4_quant/outputs/plots/`
- `outputs/plots/` for the top-level int8-vs-int4 comparison plot targets

JIT-compiled shared libraries are saved to:

- `standard/outputs/so/`
- `cube/outputs/so/`
- `fuse_int8_quant/outputs/so/`
- `fuse_int4_quant/outputs/so/`

In `benchmark.py --target hadamard_quant_int8`, the unfused reference is
`hadamard_torch_stagewise(...) + torch_npu.npu_quantize(...)`, reported as
`torch_unfused_*` in the CSV and plots.

In the int4 targets, outputs use packed-byte storage with shape `[batch, N // 2]`.
The unfused int4 reference path is `hadamard_torch_stagewise(...)` followed by a
torch packed-int4 reference, also reported as `torch_unfused_*` in the fused CSV.

In `benchmark.py --target twosided`, the kernel applies the balanced
Kronecker split `H_n = H_R (x) H_C` as `Y = H_R * X * H_C`, i.e. two matmuls
on the cube unit per tile of samples instead of `log2(N)` vector butterfly
stages. It is out-of-place (`fn(x, y)`) and supports `N` in `[256, 32768]`,
so its hidden-dimension grid starts at 256 instead of 128. The same shapes
are also timed with the `standard/` vector kernel, reported as `vector_*` in
the CSV and plots alongside the `twosided_speedup_vs_vector` column.

The two-sided arm launches with the wrapper's per-shape `block_dim` schedule, so
the `bd` in the CSV filename is the cap (`cube_core_num`) and the
`twosided_block_dim` column records what each shape actually ran at.

Both arms resolve the stream per call: `torch.npu.current_stream()` drains the
torch_npu async task queue, which is what orders a raw ctypes launch after the
torch ops that fill its inputs. That drain costs ~27 us per call, more than the
device time of every shape in the grid, so these CSV durations are launch-bound
for both kernels and the `twosided_speedup_vs_vector` column mostly compares
wrapper overhead. Hoisting the stream pointer out of the loop does not fix
this -- it drops the drain, and the kernel can then run ahead of the torch ops
setting up its inputs and of the events meant to bracket it.

Device time needs a profiler. `cube/devtime_bench.py` runs a list of shapes
under `msprof` and reads the per-launch duration off the device task record:

```bash
cat > /tmp/cfg.json <<'EOF'
[{"tag":"ts", "kind":"twosided","batch":8192,"n":4096,"bd":24,"reps":10,"warmup":3},
 {"tag":"os", "kind":"onesided","batch":8192,"n":512, "bd":24,"reps":10,"warmup":3},
 {"tag":"std","kind":"std",     "batch":8192,"n":4096,"bd":24,"reps":10,"warmup":3}]
EOF
python cube/devtime_bench.py /tmp/cfg.json
```

### The combined entry point (`auto/`)

`auto/jit_util_hadamard_auto.py` exposes one out-of-place `fn(x, y)` that picks
the kernel from `N` and, at one size, the batch:

| `N` | kernel | why |
|---|---|---|
| 64, 128 | one-sided | no `R * C` with both factors 16-aligned exists |
| 256, batch < 32768 | two-sided | `N*(R+C)` MACs instead of `N*N` |
| 256, batch >= 32768 | one-sided | both are at the memory roofline; its stores win |
| 512 .. 32768 | two-sided | the split's margin grows with `N` |

**Where both kernels run, the split wins.** The one-sided product's MACs grow
4x per doubling of `N` against 2x for its bytes, so it is left holding only the
sizes the split cannot reach -- `N < 256`, where `N = R * C` with both factors
16-aligned has no solution -- plus one corner of `N=256`. There both kernels are
bandwidth-bound rather than MAC-bound, which is the only regime where the
one-sided kernel's cheaper store pattern can show, and past batch 16384 it does.
Each back end is compiled the first time a size needs it, and
`pick_implementation(n, batch)` answers the routing question without compiling
anything.

**The vector kernel is not one of the branches.** Over the required grid it
loses 99 of 104 cells, geomean 1.61x; the five it wins are all at batch <= 8
and `N` <= 512, where a cube tile's fixed cost has nothing to amortise against
-- see "Cube versus vector over the whole grid" below.
It stays in `standard/` as the baseline, and as the only kernel that runs
`N < 64`.

#### Required sizes, and what covers them

The grid this example holds itself to is `bench_common.BENCH_HIDDEN_DIMS`:

```
128, 256, 512, 1024, 2048, 4096, 8192, 16384
```

which is also `standard/test_hadamard.py`'s own grid; the fused quantize suites
use the subset `{128, 1024, 16384}`. **All eight are covered**, and the
dispatcher reaches two sizes beyond them -- `N=64`, which only the one-sided
kernel takes, and `N=32768`, which only the two-sided one does (the vector
kernel caps at 16384, where one row fills its 32 KB UB tile). `N=32768` is the
ceiling for all of them: at 65536 the split's accumulator would be 256 KB
against a 128 KB L0C.

`auto/test_hadamard_auto.py` runs the required grid plus both extremes, 4
batches each, against an exact fp64 transform, and asserts the dispatcher's
output is **bit-identical** to calling the chosen back end directly -- the
dispatcher must add nothing but the choice.

**Batch is unrestricted everywhere.** The two-sided kernel splits by whole
tiles, then `runTiles<R, C, S>` recurses into `runTiles<R, C, S / 2>` for the
remainder, so any count decomposes into tiles of S, S/2, ... 1 samples. The
one-sided kernel needs at least 16 samples -- its smallest tile is 16 rows and
its tail trick backs a tile up to `end` -- so its host pads shorter batches
through a cached scratch pair and discards the extra rows. Either way the
caller sees no constraint: batches 0, 1, 2, 3, 5, 7, 15, 16, 17, 31, 33, 63,
65, 127, 129, 1000 and 1023 all pass at N = 128, 512, 1024, 8192 and 32768,
`batch = 0` included as a no-op.

#### Where the restrictions live

In each kernel's own host wrapper, which is the only place that can report
them. **None of the three kernels can diagnose a bad argument**: the vector one
guards only `n == 0 || n > ELEMENTS_PER_TILE` and that returns silently with the
input untouched, while a non-power-of-two `n` or a `log2_n` that disagrees with
it falls through to the generic body and quietly computes the wrong transform --
`log2_n` is passed in, never derived, and the specialised dispatch treats a
mismatch as a reason to fall through rather than to fail.

both cube kernels validated in their wrappers already. The vector
kernel's checks used to sit in `standard/test_hadamard.py`, where nothing in the
shipped path called them; they now live in `standard/jit_util_hadamard.py`
beside `MAX_HADAMARD_N`, are applied by the wrapper on every call, and
`batch`/`n`/`log2_n` are derived from `x` when omitted. Two tests go through the
wrapper rather than the validator, so the checks cannot quietly stop being
wired -- which is exactly the state they were moved out of.

#### Against the vector kernel

msprof `op_summary`, median of 15 launches after 4 warmups, 910B2, device time.
Each kernel runs at its own best `block_dim` (the two cube kernels use their
shipped schedules; the vector kernel was swept and 24 is its best at every `N`,
since `blockDim=24` already gives it all 48 AIV cores). One input buffer for
every kernel, so all three see the same footprint. Above 1.0 means the combined
entry point is faster:

| batch | N=128 | N=256 | N=512 | N=1024 | N=2048 | N=4096 | N=8192 | N=16384 |
|------:|------:|------:|------:|------:|------:|------:|------:|------:|
|     1 | 1.59x | 1.61x | 1.17x | 1.82x | 1.55x | 1.41x | 1.65x | 2.11x |
|    16 | 1.81x | 1.86x | 1.42x | 1.66x | 1.69x | 1.87x | 1.97x | 1.84x |
|    64 | 1.62x | 1.63x | 1.27x | 1.55x | 1.73x | 1.89x | 2.06x | 2.27x |
|   256 | 1.41x | 1.47x | 1.24x | 1.62x | 1.80x | 2.21x | 2.81x | 2.55x |
|  1024 | 1.62x | 1.42x | 1.26x | 1.81x | 2.47x | 2.98x | 3.58x | 2.71x |

**Geometric mean 1.79x over the 40 cells, faster in every one**, from 1.17x to
3.58x. Per `N` the geomeans are 1.60 / 1.59 / 1.27 / 1.69 / 1.82 / 2.01 / 2.32 /
2.28x -- the gap widens with `N`, because the vector kernel pays `log2(N)`
passes over UB while the split pays two matmuls. At batch 1024 that is
120.5 -> 44.4 us at N=16384 and 57.6 -> 16.1 us at N=8192.

`N=512` is the weakest column at 1.27x, which is the crossover sitting there:
the one-sided kernel is already arithmetic-bound and the split's turnaround is
still not amortised. The single cell where the size rule is not optimal is
`N=512` with batch < 16, where the split read 1.96 us against the one-sided
kernel's padded 2.18 us -- 0.22 us at a 2 us shape, inside what a per-library
offset produces, so it is not worth a batch condition.

**This table predates the bit-plan tile and the batch-aware N=512 routing**, so
that weakest column has since improved: at batch 256 - 1024 the entry point then
returned the bit-plan two-sided tile and read 1.28x - 1.45x rather than 1.24x -
1.26x. The bit-plan has since been retired and the routing re-measured --
see *Re-routing one-sided vs two-sided* for what ships now. Every other column reproduces: re-measured at batch 1024, N=16384 is
120.4 -> 44.3 us against the 120.5 -> 44.4 recorded here, and N=8192 is
57.6 -> 16.0 against 57.6 -> 16.1.

Two measurement notes. The vector kernel is **in place**, so relaunching it on
one buffer makes its values diverge (`H*H = n*I`, and fp16 saturates to inf
within a few reps); a control at N=4096 with a fresh buffer per launch read
1.005x, so inf costs nothing and the reuse is fair. Cycling a 19-buffer pool
instead *did* cost 1.028x at N=16384 -- but that is the 608 MB pool spilling L2,
not the values, which is why the comparison uses one buffer for all three
kernels rather than a pool.

### Measured device time

#### The full grid: which kernel wins where

Every (N, batch) in the supported range, each kernel at the `block_dim` its own
wrapper ships, msprof `op_summary`, median of 20 launches after 5 warmups.
Microseconds; **bold** is the winner. "two-sided" is the wrapper as shipped, so
at N=512 it is the bit-plan tile.

| N | batch | vector | one-sided | two-sided | fastest | margin |
|---:|------:|-------:|----------:|----------:|:--------|-------:|
| 64 | 16 | 2.96 | 1.41 | n/a | **one-sided** | 2.10x |
| 64 | 64 | 3.07 | 1.44 | n/a | **one-sided** | 2.13x |
| 64 | 256 | 3.21 | 1.87 | n/a | **one-sided** | 1.72x |
| 64 | 1024 | 3.90 | 2.01 | n/a | **one-sided** | 1.94x |
| 64 | 4096 | 6.78 | 2.50 | n/a | **one-sided** | 2.71x |
| 64 | 16384 | 18.11 | 3.30 | n/a | **one-sided** | 5.49x |
| 128 | 16 | 3.01 | 1.66 | n/a | **one-sided** | 1.81x |
| 128 | 64 | 3.02 | 1.90 | n/a | **one-sided** | 1.59x |
| 128 | 256 | 3.36 | 2.35 | n/a | **one-sided** | 1.43x |
| 128 | 1024 | 4.30 | 2.62 | n/a | **one-sided** | 1.64x |
| 128 | 4096 | 7.90 | 3.59 | n/a | **one-sided** | 2.20x |
| 128 | 16384 | 22.34 | 5.00 | n/a | **one-sided** | 4.47x |
| 256 | 16 | 3.03 | 1.67 | 2.33 | **one-sided** | 1.40x |
| 256 | 64 | 3.22 | 2.00 | 2.46 | **one-sided** | 1.23x |
| 256 | 256 | 3.28 | 2.25 | 3.79 | **one-sided** | 1.46x |
| 256 | 1024 | 4.17 | 2.90 | 6.20 | **one-sided** | 1.44x |
| 256 | 4096 | 7.87 | 4.30 | 11.08 | **one-sided** | 1.83x |
| 256 | 16384 | 22.67 | 8.91 | 20.66 | **one-sided** | 2.32x |
| 512 | 16 | 3.15 | 2.19 | 2.26 | **one-sided** | 1.03x |
| 512 | 64 | 3.20 | 2.35 | 2.54 | **one-sided** | 1.08x |
| 512 | 256 | 3.70 | 2.94 | 2.88 | **two-sided** | 1.02x |
| 512 | 1024 | 5.56 | 4.33 | 3.84 | **two-sided** | 1.13x |
| 512 | 4096 | 12.45 | 8.86 | 6.30 | **two-sided** | 1.41x |
| 512 | 16384 | 40.00 | 27.08 | 15.92 | **two-sided** | 1.70x |
| 1024 | 16 | 3.25 | n/a | 2.00 | **two-sided** | 1.63x |
| 1024 | 64 | 3.56 | n/a | 2.22 | **two-sided** | 1.60x |
| 1024 | 256 | 4.47 | n/a | 2.81 | **two-sided** | 1.59x |
| 1024 | 1024 | 8.43 | n/a | 4.72 | **two-sided** | 1.79x |
| 1024 | 4096 | 23.71 | n/a | 10.86 | **two-sided** | 2.18x |
| 1024 | 16384 | 84.54 | n/a | 35.38 | **two-sided** | 2.39x |
| 2048 | 16 | 3.63 | n/a | 2.16 | **two-sided** | 1.68x |
| 2048 | 64 | 4.15 | n/a | 2.49 | **two-sided** | 1.67x |
| 2048 | 256 | 6.28 | n/a | 3.46 | **two-sided** | 1.81x |
| 2048 | 1024 | 14.76 | n/a | 6.04 | **two-sided** | 2.44x |
| 2048 | 4096 | 48.19 | n/a | 15.96 | **two-sided** | 3.02x |
| 2048 | 16384 | 186.20 | n/a | 56.26 | **two-sided** | 3.31x |
| 4096 | 16 | 4.25 | n/a | 2.39 | **two-sided** | 1.78x |
| 4096 | 64 | 5.42 | n/a | 2.85 | **two-sided** | 1.90x |
| 4096 | 256 | 10.18 | n/a | 4.52 | **two-sided** | 2.25x |
| 4096 | 1024 | 28.47 | n/a | 9.36 | **two-sided** | 3.04x |
| 4096 | 4096 | 101.34 | n/a | 29.16 | **two-sided** | 3.48x |
| 4096 | 16384 | 397.29 | n/a | 215.43 | **two-sided** | 1.84x |
| 8192 | 16 | 5.62 | n/a | 2.82 | **two-sided** | 1.99x |
| 8192 | 64 | 8.10 | n/a | 3.80 | **two-sided** | 2.13x |
| 8192 | 256 | 18.02 | n/a | 6.34 | **two-sided** | 2.84x |
| 8192 | 1024 | 57.55 | n/a | 16.04 | **two-sided** | 3.59x |
| 8192 | 4096 | 219.51 | n/a | 54.90 | **two-sided** | 4.00x |
| 8192 | 16384 | 852.17 | n/a | 456.98 | **two-sided** | 1.86x |
| 16384 | 16 | 8.41 | n/a | 4.50 | **two-sided** | 1.87x |
| 16384 | 64 | 13.94 | n/a | 6.00 | **two-sided** | 2.32x |
| 16384 | 256 | 35.30 | n/a | 13.74 | **two-sided** | 2.57x |
| 16384 | 1024 | 120.40 | n/a | 44.28 | **two-sided** | 2.72x |
| 16384 | 4096 | 464.79 | n/a | 193.44 | **two-sided** | 2.40x |
| 16384 | 16384 | 1827.18 | n/a | 920.43 | **two-sided** | 1.99x |
| 32768 | 16 | n/a | n/a | 5.46 | **two-sided** | -- |
| 32768 | 64 | n/a | n/a | 11.82 | **two-sided** | -- |
| 32768 | 256 | n/a | n/a | 36.00 | **two-sided** | -- |
| 32768 | 1024 | n/a | n/a | 132.96 | **two-sided** | -- |
| 32768 | 4096 | n/a | n/a | 525.44 | **two-sided** | -- |
| 32768 | 16384 | n/a | n/a | 2086.01 | **two-sided** | -- |

Reading it:

- **`N <= 256`: the one-sided kernel wins everywhere**, by 1.23x - 2.58x over
  the two-sided split and 1.43x - 5.49x over the vector butterfly. Its tile
  touches each pipe once; at these sizes the extra `sqrt(N)/2` MACs are cheaper
  than the two-sided tile's L1 turnaround.
- **`N = 512` is the one crossover, and batch decides it.** One-sided wins at
  batch 16 and 64 (1.03x, 1.08x); the two-sided bit-plan tile wins from batch
  256 up (1.02x, 1.13x, 1.41x, 1.70x). **Superseded**: the split now wins every
  batch at N=512 by 1.25x - 1.82x, and the one crossover has moved to N=256
  pointing the other way -- see *Re-routing one-sided vs two-sided*.
- **`N >= 1024`: the two-sided split wins everywhere**, 1.59x - 4.00x over the
  vector kernel. The one-sided product's `N*N` MACs grow 4x per doubling against
  2x for its bytes, which is why its table stops at 512.
- **`auto` picks the faster of the two cube kernels at all 60 shapes**, the
  N=512 boundary included.
- **The vector kernel wins 0 of the 54 cells it can run.** It does beat the
  *two-sided* kernel in three of them -- N=256 at batch 256, 1024 and 4096, by
  1.16x, 1.49x and 1.41x, the weak case analysed below -- but the one-sided
  kernel is faster than both there, so it never takes a cell outright. It
  remains the only kernel for `N < 64`.
- Overall: **one-sided takes 20 of the 60 cells, two-sided 40.**

#### One host-side cost the device-time grid hid: `batch < 16` (SUPERSEDED)

**This no longer describes the code.** It is kept because the host-vs-device
trap it records is worth knowing, but the padding it blames is gone: the tile is
16 rows while the TRANSFERS around it are not, so the kernel masks a short batch
instead of the wrapper padding it, and `grep pad cube/jit_util_hadamard_onesided.py`
now finds only the comment saying so. The wall-clock figures below were measured
against the padding wrapper and have NOT been re-measured since it was removed.

As recorded then: the wrapper padded a shorter batch into a cached 16-row
scratch, which cost two extra torch copies around the launch, each forcing
another `torch_npu` task-queue drain. Wall-clock went from ~63 us at
`batch >= 16` to **243 - 279 us below it**, against ~57 - 59 us for the vector
kernel at the same shapes, while device time was unchanged -- so none of it
appeared anywhere in the grid above. It mattered because `auto` routes every
`N <= 512` to this kernel, so at `batch < 16` the dispatcher returned the kernel
that was ~4.3x slower in wall-clock while being 1.4 - 2.1x faster on device.

The lesson that outlived the code: **a device-time grid cannot see a host-side
regression, and this kernel's launch overhead dwarfs its device time at small
batch.** Re-measure wall-clock, not just msprof, before calling a short-batch
shape fixed. The device side of that regime did separately get 1.28x - 1.36x
faster -- see **The per-core remainder** below.

Bandwidth shows where the two-sided margin comes from and where it goes: the
cube path peaks at **2.30 - 2.45 TB/s** while the working set (`2*N*batch*2`
bytes) stays under ~134 MB and settles at **1.0 - 1.2 TB/s** past that, as L2
stops absorbing the traffic. The vector kernel never exceeds ~800 GB/s at any
footprint, so the ratio compresses from ~4x to ~1.9x at the largest shapes
without either kernel changing behaviour.

#### The two-sided-vs-vector ratio, at a wider batch range

msprof `op_summary`, median of 20 launches over 3 interleaved passes, 910B2,
with the shipped `block_dim` schedule. The ratio is `vector_us / twosided_us`,
so **above 1.0 means the two-sided kernel is faster** and below 1.0 means the
vector kernel in `standard/` wins. The `twosided_speedup_vs_vector` column in
the CSV uses the same convention.

| batch | N=256 | N=1024 | N=4096 |
|------:|------:|-------:|-------:|
|    64 | 1.26x | 1.64x  | 1.67x  |
|   256 | 0.87x | 1.43x  | 1.79x  |
|  1024 | 0.66x | 1.49x  | 2.01x  |
|  4096 | 0.72x | 1.61x  | 2.08x  |
|  8192 | 0.84x | 1.64x  | 2.09x  |
| 65536 | 1.95x | 1.48x  | 1.74x  |

At batch 8192 that is 15.1 / 26.7 / 94.7 us for the cube kernel against
12.7 / 43.9 / 198.3 us for the vector one; peak is 1614 GB/s at N=256
batch 65536. Geometric mean over the 18 shapes above is 1.42x.

`N=256` is the weak case, and `batch` 256 to 8192 is where it loses. The
balanced split makes both Hadamard factors 16x16, so a tile is only 32 samples
and a given batch yields the fewest tiles of any size -- which turned out to be
exactly the problem. See **Why N=256 is the weak case** below. The alternative
for small `N` is the one-sided `Y = X * H_N` product, which trades 8x the FLOPs
for a tile body that touches each pipe once; that lives in the one-sided kernel rather
than here, because it is no longer the two-sided split this kernel is about, and
it is 1.23x - 2.58x faster than this kernel at `N <= 256`, and faster at N=512
too until the bit-plan tile turns that size around at batch 256.

### Why N=256 is the weak case

The cube kernel's *slope* is the better one at every size: fitted across batch
at N=256 it is 0.629 ns/sample against the vector kernel's 1.196, i.e. 1.90x
leaner. What it carried was a ~13 us intercept. Four candidate causes were
measured and all four refuted -- block dispatch is 0.90 us at `block_dim=24`,
the per-core Hadamard-factor prologue is free (0.90 us, identical to an empty
kernel), deleting both `TMATMUL`s saves 2-13%, and reps=20 and
reps=2000 give identical times, so it is not a clock ramp either.

> **The third of those four was wrong, and it was the answer.** The prologue's
> ND operand load *was* the intercept: see *The operands were loaded as a
> narrow-tile gather* below, where reading the factors as NZ instead is
> 1.34x - 1.65x at N=256 and turns this section's core-count U-curve into
> monotone scaling to 24 cores. The probe that cleared it timed a
> prologue-only kernel against an empty one, and at `block_dim=24` all 24
> cores run their gathers *in parallel, behind block dispatch*, so in isolation
> it hides inside the 0.90 us floor. In the real kernel it sits in front of
> MTE1 on every core instead. **Pricing a fixed cost in isolation measures its
> latency, not what it blocks** -- the probe has to delete it from the whole
> kernel, which is what finally found it. The rest of this section's analysis
> stands; its conclusion about the cause does not.

It is **pipeline depth per core**, and two sweeps isolate it. Holding the work
fixed at 48 tiles and changing only the spread:

| cores | 2 | 4 | 8 | 12 | 24 |
|---|---:|---:|---:|---:|---:|
| us | 13.15 | 9.06 | **8.68** | 10.12 | 16.26 |

8 cores beat 24 on identical work by 1.87x. Pinning tiles-per-core at 2 and
varying the core count gives `t = 1.99 us + 0.679 us * cores` -- wall time grows
*linearly* with core count although every core does identical independent work,
and the aggregate rate stays at 94 GB/s whether 1 core or 24 are running.
Pinning the core count at 4 and varying the run length gives
`t = 4.01 us + 0.365 us * tiles_per_core`, i.e. 359 GB/s. Both fit every
measured point to within 0.03 us. A tile costs **340 ns on a shallow core and
91 ns on a deep one**. Depth buys bandwidth; width does not.

Since every tile is the same 16 KB at every size (`S * N = 8192` throughout the
case table), the tile count scales with `batch * N`, so N=256 is simply the size
that runs out of tiles first -- at batch 1024 it has 32 tiles against 128 for
N=1024 and 512 for N=4096, which at `block_dim=24` is 2 tiles per core, the
worst possible depth.

Two fixes follow, and they are both in the shipped kernel:

- **The two-phase tile body** splits the tile at its L1 turnaround into two
  phases and issues `A(i+1)` before `B(i)`. The straight-line body walks
  MTE1 -> M -> FIX -> MTE1 -> M -> FIX, and because each pipe is in-order its
  *second* MTE1 use cannot start until its own first matmul and L0C->L1 move
  retire -- which also pins the next tile's first MTE1 behind it, so only the
  GM->L1 load ever overlapped. Interleaving makes the MTE1 order
  `A0 A1 B0 A2 B1 ...`. Needs two of every intermediate plus a 4-deep X ring.
- **A `block_dim` schedule** in the wrapper instead of the constant 24:
  `bd = clamp(round(sqrt(k * tiles)), 1, 24)`, with `k` fitted per `N` over the
  whole case table -- 40 measured shapes, no extrapolation. It lands within
  1.025x of the best measured `block_dim` (mean 1.001x). The last column is what
  the constant 24 cost at that size, i.e. what the schedule buys:

  |     N |     k | shapes | worst regret | constant-24 worst |
  |------:|------:|-------:|-------------:|------------------:|
  |   256 |  0.49 |      6 |       1.025x |            1.834x |
  |   512 |  1.00 |      6 |       1.005x |            1.908x |
  |  1024 |  8.00 |      6 |       1.000x |            1.207x |
  |  2048 | 16.00 |      6 |       1.000x |            1.085x |
  |  4096 |  8.00 |      6 |       1.000x |            1.093x |
  |  8192 |  8.00 |      5 |       1.000x |            1.000x |
  | 16384 |  8.00 |      5 |       1.000x |            1.000x |
  | 32768 | 16.54 |      6 |       1.029x |            1.074x |

  `k` is fitted per `N` rather than from a rule because nothing about `N`
  predicts it: **N=512 wants `k = 1` where N=1024 wants 8**, and an earlier
  version of this schedule assumed 8 for it and was 1.51x off at batch 1024.
  N=512 is the only size whose tile is 16 `TEXTRACT`s plus 16 `TSTORE`s -- the
  most per-tile work in the table -- which is the likely reason, but it was found
  by sweeping every size, not by that argument. If the case table gains a size,
  sweep `block_dim` for it and add the entry rather than trusting the default.

Measured as a 2x2 factorial at N=256, device time in us (`sched` = scheduled
`block_dim`, `dbuf` = the two-phase tile):

| batch | neither | sched only | dbuf only | both | vs vector, both |
|------:|--------:|-----------:|----------:|-----:|----------------:|
|    64 |    2.84 |       2.84 |      2.90 | 2.52 | 1.26x |
|   256 |    6.38 |       3.99 |      6.41 | 3.66 | 0.87x |
|  1024 |   11.58 |       7.25 |     11.26 | 6.14 | 0.66x |
|  4096 |   16.89 |      12.88 |     16.06 |10.86 | 0.72x |
|  8192 |   20.28 |      17.83 |     18.68 |14.66 | 0.84x |
| 65536 |   53.93 |      53.93 |     41.28 |41.28 | 1.95x |

The two are complementary rather than redundant -- the schedule pays at small
batch, the deep pipeline at large batch -- and superadditive where they meet
(at batch 1024, 1.60x and 1.03x alone, 1.89x together), because the deep
pipeline only has something to overlap once a core has a run of tiles to give
it. Together, measured against the kernel as it stood before both changes (built
from git HEAD) over all 7 sizes x 5
batches: **1.00x to 1.96x, geometric mean 1.12x, no regression**. The largest
single wins are N=256 batch 1024 (1.96x) and N=512 batch 256 (1.91x).

A caution on reading that table: at the largest shapes this harness cannot
resolve differences below about 10%. A null control -- the same `.so` under two
names, same `block_dim` -- measured 208.08 vs 228.28 us at N=8192 batch 8192,
a 0.91x "regression" from zero code difference, and the sign of that shape's
difference flips between runs. The spread within one pass is tight (<1%), so it
is a per-run, per-library offset rather than jitter; tensor placement is the
likely cause. Anything inside that band at `block_dim=24` and `N > 256`, where
both builds compile the same shallow path, is noise -- the shallow buffer
offsets are mechanically identical and the shallow event ids unchanged.

#### The NZ view is not a reinterpret trick, it is a symmetry of the transform

Both GM transfers of a 16-column tile go through a `Layout::NZ` view
which is worth 4.5-8.7x at N=256: the
default ND2NZ path writes L1 in C0-sized **32-byte** pieces, and the NZ view
makes the whole tile one contiguous burst. The obvious reading is that this is a
*byte reinterpret* and so needs `C == 16` -- one block-column, where ND and NZ
bytes coincide. That reading is too weak, and it had the kernel leaving the win
on the table at every other size.

NZ position is a **permutation of the bit fields** of ND position:

    ND:  [ row>>4 | row&15 | col>>4 | col&15 ]
    NZ:  [ col>>4 | row>>4 | row&15 | col&15 ]

and a Sylvester Hadamard is *invariant* under any bit permutation of its index,
because `H[i][j] = (-1)^popcount(i & j)` and `popcount` does not care which bit
positions the set bits occupy: `popcount(sigma(i) & sigma(j)) = popcount(i & j)`.
So feeding the cube the ND bytes read as NZ computes the transform of a permuted
input, which *is* the permuted transform -- and reading the accumulator back out
through the same NZ view undoes the permutation exactly. The two legs must flip
**together**; either alone emits a permuted result, at `rel_rms` sqrt(2).

Checked exactly in integer arithmetic for (R,C) = (16,16), (16,32), (32,16),
(32,32), (16,64), (32,64), (64,64) and (64,128), then on hardware over the
validator's 288 cases.

**The one restriction is the sample axis.** The permutation lifts `col>>4` above
the row index, and for a tile of `S` stacked samples that is above the *sample*
index too -- but the kernel's operator is `I_S (x) H_N`, which is only invariant
under permutations that leave the sample bits fixed. So the tile must hold one
sample. That is `S == 1`, i.e. N >= 8192. (N=32768 is excluded for a second,
unrelated reason: its blocked body stores row SUB-BLOCKS of the output, and
`TStoreAccNz2nz` derives the destination block-column stride from the shape
rather than from `gStride1`, so it can only write a whole NZ tile.)

**Measured: 1.017x geometric mean, and that is the honest number.** Paired A/B,
`block_dim` 12 and 24, batch 16..4096, with N=32768 as a built-in null control
(blocked, so the switch cannot reach it) reading 0.997-1.005x across five cells:

| N | b=16 | b=64 | b=256 | b=1024 | b=4096 |
|---:|---:|---:|---:|---:|---:|
| 8192 (bd 24) | 1.006x | 0.992x | 1.017x | 1.018x | 1.154x |
| 8192 (bd 12) | 1.028x | 1.009x | 1.018x | 1.018x | 1.019x |
| 16384 (bd 24) | 1.022x | 1.023x | 1.007x | 1.002x | 0.968x |
| 16384 (bd 12) | 1.018x | 1.004x | 1.007x | 1.013x | 1.009x |

18 of 20 cells sit at or above 1.0 and the modal value is 1.017-1.018x, which is
clear of the null band; the two outliers are both >60 us shapes, where this
harness is known to read 0.86-1.14x from zero code difference.

**Why so much smaller than the 4.5-8.7x at N=256?** Because the prize scales
with how *narrow* the tile is, and it is the sizes that cannot use the trick
that would want it most. At `C == 16` the ND descriptor row is 32 bytes; at
`C == 128` it is 256 bytes, and these sizes are L2-resident anyway -- the GB/s
column reads 1500-2100 GB/s against a ~920 GB/s streaming ceiling, so the GM leg
is simply not the constraint. A noop-matmul floor probe says the same thing
from the other side: dropping both matmuls saves only 1-5% at N <= 8192, so
95-99% of the time is movement -- but it is *on-chip* movement (L1->L0 `TMOV`,
L0C->L1 `TEXTRACT`) and pipe serialization, not GM descriptors.

**How much is left on the table, and where.** A probe that drops the
`S == 1` guard, which computes wrong answers but moves exactly the bytes a
restructure would. Measured on the movement floor (matmuls dropped, so the
matmuls cannot mask it -- the most favourable framing available):

| N | b=1024 | b=4096 |
|---:|---:|---:|
| 512 | 1.012x | 1.034x |
| 1024 | **1.092x** | **1.146x** |
| 2048 | 1.014x | 1.012x |
| 4096 | 1.007x | 1.006x |

So the entire remaining prize is ~1.1x at **N=1024**, and essentially nothing
elsewhere. N=512 has the narrowest tile in this group (`C == 32`, 64-byte
descriptor rows) and still gains only 1.01-1.03x, because it is bound by its 16
`TEXTRACT`s plus 16 `TSTORE`s rather than by the load -- the same thing that
makes its `block_dim` constant `k = 1` where every neighbour wants 8.

**Why reaching it is a restructure and not a switch.** There is a layout tension
with no free side. The *load* wants the horizontal concatenation `(R x S*C)`,
which is one contiguous NZ descriptor because sample `s` lands exactly in
block-columns `[s*C/16, (s+1)*C/16)`. The *store* needs the concatenated
accumulator too, because that is the only view in which sample `s` is a
contiguous byte range of L0C and can be re-`TASSIGN`ed as its own `(R x C)` Acc
tile -- a row sub-block of the vertical stack cannot be stored at all, for the
same `TStoreAccNz2nz` reason that excludes N=32768. But the concatenated view
comes from the *left* multiply and the stacked view from the *right* one, so
only one of the two legs can have it: putting `H_R` first (verified correct in
exact arithmetic for S = 2, 4, 8 and 16) buys the contiguous load and loses the
store. Breaking the tension needs the load to scatter `S` per-sample NZ images
into the strided row blocks of a stacked L1 tile, which is a 5D-descriptor
question that has not been tried.

So: kept, because it is correct, costs nothing and regresses nothing, but it is
worth ~1.7% rather than the large win the descriptor-width argument suggests,
and the only size with a further ~1.1x in it is N=1024.

#### The bit-plan tile: conversion-free transfers at any S, because H is a constant (2.47x at N=512)

The `S == 1` restriction above is not a property of the layouts. It is a
property of insisting the operand be a *Hadamard matrix*. Drop that and the
restriction goes away.

`H_N` is a tensor power of `H_2`, so the transform applies `H_2` to each index
bit **independently**. A matmul against a compile-time operand applies whatever
we like to one FIELD of the byte index, and a contiguous NZ load of an `(A x B)`
view groups the GM byte index `p` as

    p = [ col_hi | row_hi | row_lo | col_lo ]
          t-1..a+4  a+3..8   7..4     3..0

so the left operand owns `p[4, a+4)` and the right operand owns
`p[0,4) u p[a+4, t)` -- together they partition `p`. Correctness is then pure
bit accounting:

> **every transform bit gets `H_2` exactly once; every sample bit gets `I_2`.**

The operand for a field is the tensor product, in that field's bit order, of
`H_2` for a transform bit and `I_2` for a sample bit. Sample bits sit at the top
of `p`, so they land in `col_hi` and are simply skipped. At N=512 with a T=8192
tile the plan is view `(64 x 128)` with

    OpA = I_2 (x) H_32   (64 x 64,   the row field p[9..4],  p9 is a sample bit)
    OpB = I_8 (x) H_16   (128 x 128, the col field p[12..10] u p[3..0])

Verified in exact integer arithmetic for every entry in the case table, then on
hardware. **This needs no new kernel code**: view `(64 x 128)` with `S == 1` is
byte for byte the configuration the kernel already runs at N=8192, so the
bit-plan is the existing, already-validated body driven with `T`-element rows
and these operands substituted for `H_64` / `H_128`. It lives in
`bitplan_operands` in `cube/jit_util_hadamard_twosided.py`; a batch that does not
fill whole tiles sends its remainder to the classic path in a second launch.

**What it buys, and why only at N=512.** The tile drops from `2S+5` descriptors
to **7 at any size** -- one contiguous NZ load, one `TEXTRACT` with no reshape,
one contiguous NZ store -- which is what the `C == 16` body already enjoys at
N=256. Measured against the classic path, each at its own best `block_dim`:

| N=512 | classic | bit-plan | |
|---|---:|---:|---:|
| batch 256 | 4.940 us | 2.850 us | **1.73x** |
| batch 1024 | 8.640 us | 3.870 us | **2.23x** |
| batch 4096 | 15.730 us | 6.380 us | **2.47x** |
| batch 16384 | 30.291 us | 16.140 us | **1.88x** |

The price is MACs. Both operands span the whole index, so the cost is
`A + B = 2*sqrt(T)` per value instead of `2*sqrt(N)`, i.e. `sqrt(S)` more --
6x at N=256 falling to 1x at N >= 8192. That only pays where the per-tile
descriptor count dominates, and N=512 is the only size where it does: it is the
one tile that is 16 `TEXTRACT`s plus 16 `TSTORE`s plus a 64-byte-row gather,
with matmuls at 0.7% of its time. Everywhere else the trade is a wash or a loss,
measured 0.94-1.11x at N=1024 through 4096, so the table has one entry.

This was priced *before* being built, by noticing that the N=512 plan compiles
to the existing N=8192 configuration: comparing the two at matched values and
matched tile counts predicted 1.76/2.23/2.50/1.91x against the 1.73/2.23/2.47/
1.88x finally measured. A free forecast from a measurement already in hand.

##### The whole (N, T) grid, and why T == N is a trap

The tile size `T` is a free parameter, and the MAC cost `2*sqrt(T)` falls as `T`
does -- at `T == N` it is `2*sqrt(N)`, i.e. **conversion-free transfers at zero
MAC penalty**. That looks like the obvious choice and it is the worst one.
A one-sample-per-tile build is exactly that configuration (correct answers, one
sample per tile), and the full grid, device time at each point's own best
`block_dim`, measured against the shipped converting-load kernel:

| N | batch | T=N | 2N | 4N | 8N | 16N | 32N |
|---|---|---|---|---|---|---|---|
| 512 | 1024 | 0.44x | 1.38x | 1.89x | **2.26x** | 2.19x | 1.76x |
| 512 | 4096 | 0.39x | 0.91x | 1.47x | 2.05x | **2.42x** | 1.74x |
| 1024 | 1024 | 0.49x | 0.74x | 0.95x | **0.98x** | 0.79x | -- |
| 1024 | 4096 | 0.34x | 0.58x | 0.88x | **1.12x** | 0.79x | -- |
| 2048 | 1024 | 0.58x | 0.82x | **0.96x** | 0.68x | -- | -- |
| 2048 | 4096 | 0.45x | 0.72x | **1.01x** | 0.66x | -- | -- |
| 4096 | 1024 | 0.77x | **0.98x** | 0.69x | -- | -- | -- |
| 4096 | 4096 | 0.69x | **1.00x** | 0.66x | -- | -- | -- |

**`T == N` loses 1.3x to 3x** despite being MAC-free and conversion-free: a
one-sample tile is 7 descriptors per *sample* where the shipped tile is `2S+5`
per *S* samples, and per-tile overhead swamps everything else. The curve is
unimodal in `T` with its peak well inside the range, so this is a parameter to
sweep, not to reason about -- "no MAC penalty" was the wrong thing to optimise.

It also settles the shipped choice: T=8192 at N=512 is 2.19x/2.42x against a
best-possible 2.26x/2.42x, so one constant is within 1.03x of the per-batch
optimum and no batch-dependent `T` is worth carrying. And it confirms the middle
sizes are a wash at **every** `T`, not just at the one point measured first --
N=1024 peaks at 1.12x only at batch 4096, N=2048 at 1.01x, N=4096 at 1.00x.

##### Where conversion-free transfers stand, per kernel and per size

Complete, after sweeping the grid above:

| kernel / size | GM legs today | conversion-free possible? | measured |
|---|---|---|---|
| two-sided N=256 | **NZ both** (byte identity, `C == 16`) | already is | shipped |
| two-sided N=512 | ND->NZ converting | **yes**, bit-plan T=8192 | 2.19-2.42x, since superseded by the `C == 16` split |
| two-sided N=1024 | ND->NZ converting | yes, best T=8192 | 0.98-1.12x, declined |
| two-sided N=2048 | ND->NZ converting | yes, best T=8192 | 0.96-1.01x, declined |
| two-sided N=4096 | ND->NZ converting | yes, best T=8192 | 0.98-1.00x, declined |
| two-sided N=8192 | **NZ both** (`S == 1`) | already is | 1.017x, shipped |
| two-sided N=16384 | **NZ both** (`S == 1`) | already is | 1.017x, shipped |
| two-sided N=32768 | ND->NZ + ND store | **no** | -- |
| one-sided, all N | ND->NZ converting | **no** | -- |

N=32768 has no conversion-free view at all: `t = 15` needs `8 <= a <= 11` for
the field widths to work, while an fp16 operand must satisfy `A = 2^a <= 128`,
i.e. `a <= 7`. No overlap, so there is no two-matmul plan -- independently of
the blocked-store limitation noted above.

The one-sided kernel's `no` is a counting argument, not a layout accident: every
one of its operands is a RIGHT operand, so it can cover only one of the two
fields the layout offers, and the other keeps live transform bits
(`p[4, a+4)` under NZ, `p[0,4)` under ND). Covering both means a left *and* a
right multiply, which is the two-sided kernel. This is not free: a noop-matmul
split shows the one-sided kernel is **51-91% movement** (matmuls are 9.0% at
N=64 rising to 48.8% at N=512 batch 4096), so it spends most of its time on
movement it cannot reduce this way. It keeps N=64-256 regardless, because the
two-sided kernel already has ideal transfers at N=256 and still loses there.

##### `TMOV` will not fractalise an ND tile (so the conversion cannot move to MTE1)

The tempting third option is to leave the conversion out of the GM leg
altogether: load ND->ND (contiguous) and let the `L1 -> L0` `TMOV` fractalise on
the MTE1 leg instead of MTE2. `TLOAD` does support ND->ND into a `Mat` tile, and
a `TMOV` from an ND (`SLayout::NoneBox`) source compiles.

It does not convert. Measured with an identity multiply so the output is exactly
what the cube saw, against a converting-load control that returns the identity:
**`TMOV` from an ND L1 tile reads the bytes as fractals and TRANSPOSES them.**
That matches the dispatch in `tmov_common.hpp`, which branches on
`DstTileData::SFractal == SrcTileData::SFractal` and takes the transposing path
when they differ. So the ND route is not a way to avoid the permutation -- it is
a *different* permutation, `p = [ r_hi | c_hi | c_lo | r_lo ]`, which is still a
bit-field permutation and so still fits the bit accounting, with the left and
right operand roles swapped. Same framework, no new capability, and the
one-sided impossibility holds under both routes.

**It moved a routing decision.** N=512 was the one size where the one-sided
kernel beat the two-sided one, which is why `auto` branched there. With the
bit-plan the two-sided kernel wins from batch 256 up -- 1.01x at 256 to 1.69x at
16384 -- while one-sided keeps a flat 1.04-1.06x below it. So N=512 is now the
only size whose dispatch reads batch as well as N; see `BITPLAN_BATCH_MIN`.

#### The phase split applies at every size, not just N=256

The split was written for the `C == 16` body and gated on `C == 16`, which reads
like a dependency on the free NZ reinterpret that case enjoys. **It is not.** The
reinterpret is what makes the `C == 16` *turnaround* cheap -- two TMOVs instead
of `S` TEXTRACTs plus `S` TSTOREs. The *split* is about pipe ORDER, and needs
nothing but a cut at the turnaround and two of every intermediate. The general
body already pays its reshape either way, so extending the split adds no work.

It is also the case that should gain MORE: it visits each pipe twice exactly as
the `C == 16` body does, with far more queued behind the second visit. Measured
paired against the switch off, with N=256 (deep in both builds) and N=16384
(fits in neither) as built-in null controls, both reading 1.00x:

| N | b=1024 | b=8192 |
|---:|---:|---:|
| 512 | 1.15x | 1.17x |
| 1024 | 1.21x | 1.42x |
| 2048 | 1.33x | 1.55x |
| 4096 | 1.50x | **1.70x** |
| 8192 | 1.59x | see below |

The gain holds until the working set outgrows L2 (~192 MB), after which the
kernel is HBM-bound and pipe order cannot help -- neutral there, not harmful.
Swept at N=8192 against a byte-identical copy of the same `.so`:

| batch | footprint | deep/flat | null/deep |
|------:|----------:|----------:|----------:|
| 1024 | 34 MB | **1.59x** | 1.00x |
| 2048 | 67 MB | **1.67x** | 1.00x |
| 4096 | 134 MB | **1.70x** | 1.00x |
| 8192 | 268 MB | 0.95x | **0.86x** |
| 16384 | 537 MB | 1.04x | 1.01x |

The 0.95x is not a regression: the null control reads 0.86x from *zero* code
difference at that shape. Always run one before believing a large-shape delta.

**Before/after over the whole range** -- previously shipped (split at N=256
only, shallow-fitted schedule) against now (split wherever it fits, refitted
schedule), 32 shapes, 3 interleaved passes: **0.99x to 1.71x, geometric mean
1.16x, no regression beyond 2%**. By `N`: 1.00x, 1.20x, 1.16x, 1.21x, 1.38x,
1.25x, 1.01x for 256 .. 16384.

The two unchanged sizes are unchanged for different reasons. N=256 already had
the split. **N=16384 is locked out by capacity**: its tile is one whole sample,
32 KB, so two slots are 100% of L0A and 100% of L0B on their own, whatever else
is evicted. Blocking each stage internally does shrink those slots by `C/M` at
no MAC cost, and it does let N=16384 reach the deep path -- and it is a
**1.28x regression** there, because blocking costs more than the interleave
buys. That was built, measured, and then deleted; see **Blocking each stage
internally** below.

All three were gated behind compile-time switches while they were being
A/B-ed; the switches were removed for release once each had shipped (see
"Switches removed for release" below).

One caution learned here: the deep path needed extra `set_flag`/`wait_flag`
event ids, and renumbering the *existing* ids -- in a path that was otherwise
untouched -- cost a reproducible 0.92x at N=1024 batch 65536. The two drivers
now keep separate id names over a shared pool so the shallow path's flags are
exactly what they were. Measure renumbering as its own change.

#### Blocking each stage internally (`PTO_TS_BLOCK`): costs 1.5x, buys N=32768

The two-sided kernel is two one-sided products, and each of those can be split
again: `H_C = H_{C,acc} (x) H_{C,mat}`, applied as `kC1` column blocks of the
data against a resident `+-H_M`, with the sum over blocks taken in the L0C
accumulator. `H_{C,acc}` is never built -- its entries are `(-1)^popcount(i&j)`,
so the only operands that ever occur are `+H_M` and `-H_M`. The MAC count is
unchanged (`kC1^2` matmuls of `rows x M` by `M x M` is the same product), there
is still exactly one fp32 rounding, and results are **bit-identical**: all three
of dense, blocked-deep and blocked-shallow report the same `rel_rms` to three
figures at every size. This is the same identity the one-sided kernel uses, and
there it was worth 1.3-2.6x.

**Here it costs 1.5x and is off by default.** Blocking shrinks the per-matmul
slot, which is the point, but the blocked body walks `MTE1 -> M -> MTE1 -> M`
and so uses each pipe `kC1` times per phase -- re-serializing consecutive tiles,
which is exactly what the deep path's two-phase interleave exists to prevent.
Measured with everything else held fixed (batch 1024, msprof device time):

| arm | N=8192 | N=16384 | what differs |
|---|---:|---:|---|
| dense, deep path | 16.16 us | -- | -- |
| blocked, deep path | 28.76 us | -- | **1.78x** |
| dense, shallow path | 26.16 us | 44.72 us | -- |
| blocked, shallow path | 38.86 us | 67.88 us | **1.49x / 1.52x** |

Null controls for those: the two arms at N=4096 (where `M=64` equals both
factors, so neither blocks) read 0.996-1.008x, and at N=32768 (where both arms
block) 0.995-1.000x. So the effect is 50-80x the harness's resolution here.

Consequently `kBlockIt` is gated on whether the **dense shallow** path fits --
the fallback every size has -- and not on whether the deep path fits. Gating it
on the deep path instead makes N=16384 blocked-deep at 56.86 us against
44.30 us dense-shallow, a 1.28x regression that holds at all 24 `block_dim`
values (1.28-1.33x). With the shipped gate, every size that worked before
compiles to a **byte-identical** `.so`.

`kDbuf` therefore carries a `!kBlockIt` term: **blocking implies the shallow
path, always.** That is not just a policy, it is what makes the deep path's
capacity expressions revert to the dense ones (`kMa == R`, `kMb == C`, so the
operands are `H` whole and the slices are full tiles) -- and it made the blocked
*deep* phase bodies unreachable, so they were deleted along with the four event
ids and the accumulator-aliasing edge that only they used. The kernel lost 166
lines (1354 -> 1188) at 0.995-1.018x on five sizes x two batches, i.e. no
measurable change. A build that forces blocking on everywhere was used for the
A/B; it now routes to the blocked *shallow* body, which is the arm the 1.49x
and 1.52x rows above were measured with.

**What it does buy is a size that could not run at all.** N=32768 splits as
`128 x 256`, and `H_C` alone is 128 KB against a 64 KB L0B, so there is no dense
form of it -- blocking is not a speedup there, it is the difference between
running and not compiling. With `M=64` the operands are 16 KB, and with the two
accumulators aliased (they are live in disjoint phases) L0C lands at exactly
128 KB. 288 validation cases pass at `rel_rms` 2.94e-04, the same fp16 floor as
every other size, and a repeat-until-failure probe reported 0 bad in 40 repeats
at three shapes. At batch 1024 `block_dim` 24 it runs **133.1 us, 1009 GB/s**.

Two things were worth tuning on it. **Block-level double buffering of the right
multiply** gives MTE1 a second L0A slice so it can stage block `i+1` while the
cube is still on block `i`; that recovers 1.07-1.09x (133.06 vs 145.18 us at
batch 1024, nulls 0.999x and 1.010x). The left multiply cannot have it -- its
slice is `kMa * S * C`, twice as wide, and two of those plus the operand
overflow L0B. And the **block size was swept rather than reasoned**: `kMa = 64`
beats 32 by 1.03-1.04x, while `kMb = 32` is fastest at all three batches tested
but only by 1.004-1.017x, inside the null band, so the simpler `64/64` default
ships.

#### The defect this found: one event id must not carry two channels out of one pipe

The first blocked build passed all 252 validation cases and then failed 4 times
in 30 repeats of a single shape -- a race. It is worth recording because the
credit arithmetic that hides it looks correct.

The blocked loop needs two edges out of `PIPE_M`: a per-block write-after-read
edge `(M -> MTE1)` so the next `TMOV` does not overwrite the slice the cube is
still reading, and the end-of-phase handoff `(M -> FIX)`. Both were given the
phase's own event id. Credits balance **per channel**, and the PTO library
budgets them per `EventIdCounter<SrcPipe, DstPipe>`, so that reads as safe --
but if the hardware flag is keyed on `(src pipe, id)` rather than the ordered
pair, the `FIX` wait can take a *mid-loop* credit and run `TEXTRACT`/`TSTORE`
before the last matmul lands. Giving the write-after-read edge its own ids fixed
it: 0 bad in 60 repeats per config against a 4/30 control, probability ~e^-17
under the control rate.

Three things made it findable, and all three generalise:

- **The error magnitude named the stage.** Dropping one of `k` accumulation
  terms in an orthogonal transform leaves relative RMS `sqrt(1/k)`. Every
  failure read 0.7071, i.e. exactly one of two terms missing -- which says a
  store ran ahead of a matmul, not that data was corrupt. Report `rel_rms`
  rather than a pass/fail, and the stage names itself.
- **The reproducer had no shared buffers.** It failed on a core with two tiles,
  whose parities use disjoint slots, so nothing cross-tile could be at fault and
  the search collapsed to one tile's own flags.
- **Three extra edges were built and measured before being believed.** Forcing
  the `FIX -> M` accumulator edge, and draining `M` at the end of each phase,
  left the rate at 4/30 unchanged -- which is what ruled out the missing-edge
  explanations and pointed at the ids. Two further probe bits deadlocked on an
  unprimed credit and were discarded as mis-wired rather than scored.

The rest of the kernel was audited against this: the shallow path, the unblocked
deep path and `loadHadamardFactors` each use at most one `src = PIPE_M` channel
per id, which is why they always validated.

### Switches removed for release

Every measurement in this file was taken with a compile-time switch holding the
two arms apart. Once an arm had shipped, keeping the other one cost clarity and
bought nothing, so the switches are gone and only the two genuine tuning
constants remain -- `PTO_TS_BLOCK` / `PTO_OS_BLOCK` (the M of the blocked
accumulation) and the per-N samples-per-tile overrides. Removed:

| removed | was | why it is gone |
|---|---|---|
| `PTO_TS_NZLOAD`, `PTO_TS_NZSTORE`, `PTO_TS_NZGM` | three switches over the NZ GM legs | folded into one `Layout::kNzGm`; the legs must agree, so one predicate is also safer than three |
| `PTO_TS_DBUF`, `PTO_TS_DBUF_GEN` | the two-phase tile body, and its extension past `C == 16` | both always on; `kDbuf` is now decided by capacity alone |
| `PTO_TS_BLOCK_ALWAYS` | force blocking where it is not needed | blocking costs 1.5-1.9x; only ever on for the A/B |
| `PTO_TS_ACCALIAS` | accumulator aliasing | the off arm does not compile at N=32768, so it was never a choice |
| `PTO_TS_NOMM`, `PTO_TS_STOP`, `PTO_TS_NZUNSAFE`, `PTO_OS_NOMM` | movement-floor and truncation probes | **all compute wrong answers**; not something to ship a switch for |
| `PTO_TS_FORCE_S1` | one sample per tile | a probe for the tile-size sweep, now recorded above |
| `PTO_OS_BLOCKED` | blocked accumulation on/off | always on; the dense form it falls back to was 0.6-0.8x of the vector kernel |
| `PTO_OS_BIG`, `PTO_OS_S1024`, `PTO_OS_S2048` | one-sided N=1024/2048 | correct but 1.7-4.8x slower than the two-sided kernel |
| `PTO_OS_TILE_SPLIT` | split per-core work by samples instead of tiles | the sample split walks the halving chain on most cores -- **that reason has since expired**, there is no chain any more; see *The per-core remainder* |
| `PTO_TS_BITPLAN` (host) | bit-plan on/off | **removed**; the `C == 16` split gives the same transfers for free, see *Re-routing one-sided vs two-sided* |

Two whole code paths went with them, both **provably** dead rather than merely
slow, and the proof is that deleting each one left a **byte-identical `.so`**:

- the two-sided kernel's blocked DEEP bodies, unreachable because `kDbuf`
  requires `!kBlockIt`;
- the one-sided kernel's CHUNKED body and its H ring, unreachable because every
  N in the table is either resident (N <= 128) or blocked (N = 256, 512) -- the
  chunked form needs `!kResident && !kBlocked`, i.e. `N > 128` and `N <= 64`.

With only two bodies left, the one-sided `Layout`'s chunk abstraction
(`kKc`, `kChunks`, `kHInL0B`, the `kL1HChunk` / `kL0bChunk` pair) collapsed into
a resident/blocked branch and a single `kHPair` -- also byte-identical.

Byte-identical codegen is a stronger no-regression result than any timing, and
it is the only evidence worth having for a change that is meant to be neutral.
The A/B arms themselves are in git history.

The same pass moved the kernels' measurement narrative out of their comments and
into this file: both `.cpp` files now state the constraint each constant encodes
and nothing else, and the numbers behind them live here. That too was checked by
compiling before and after from one identical path and diffing the `.so`.

### The one-sided kernel (`cube/fast_hadamard_onesided.cpp`, N = 64..512)

`fast_hadamard_twosided.cpp` applies the balanced Kronecker split and pays two
matmuls of about sqrt(N) on a side. The one-sided kernel instead applies `H_N`
from the right only, which costs `N*N` MACs per sample -- sqrt(N)/2 times as
many, 8x at N=256 -- and buys a tile body that touches each pipe exactly ONCE:

```
GM --MTE2--> L1 --MTE1--> L0A --M--> L0C --FIX--> GM
```

which is precisely what the two-sided tile cannot do. Its tile has to turn the
intermediate around through L1 between the two multiplies, so it visits MTE1 and
M twice and consecutive tiles serialize behind the second visit -- the effect
measured above, where a tile costs 340 ns on a core with a short run against
91 ns on a long one. On the cube a fp16 matmul is cheap beside the data
movement, so trading MACs for a straight pipe is the right side of that trade.

#### `H_N` is never materialised: the butterfly, done in the accumulator

Applying `H_N` as one dense `N x N` right operand is what used to cap this
kernel at 512, because all of H has to be read once per tile. Instead the kernel
takes the vector butterfly's idea -- a Kronecker factorization applied in stages
-- and restricts the matmul to a fixed efficient size. With `N = B*M` and
`H_N = H_B (x) H_M`, blocking the product by M columns gives

```
Y[:, jM:(j+1)M] = sum_i  X[:, iM:(i+1)M] * (H_B[i,j] * H_M)
```

and `H_B[i,j]` is `+-1`, so **the only right operands that ever occur are `+H_M`
and `-H_M`**. Both sit in L0B for the whole kernel at 16 KB total, loaded once
per core, and `TMATMUL_ACC` sums the B partial products of each output block
*in the L0C accumulator* -- which is the cube's version of what the vector core
does across its `log2(N)` stages, at zero traffic. `H_B` is never built or read:
`H_B[i,j] = (-1)^popcount(i & j)`, computed from the loop indices.

The arithmetic is identical to the dense form (`B*B` matmuls of `S x M` by
`M x M` is the same `S*N*N` MACs), the rounding is identical (fp32 accumulation,
one fp16 store), and `B == 1` degenerates exactly to the dense single matmul --
which is why `N <= 128` is untouched by this.

Measured, paired over 3 interleaved passes, each kernel with its own `block_dim`
schedule. Ratios above 1.0 mean the one-sided kernel is faster:

| N | batch | one-sided us | GB/s | vector | vs vector | two-sided | vs two-sided |
|---:|------:|-------------:|-----:|-------:|----------:|----------:|-------------:|
|  64 |    64 |   1.49 |   11 |   3.06 |  2.05x | n/a | -- |
|  64 |  1024 |   2.04 |  129 |   3.99 |  1.96x | n/a | -- |
|  64 |  8192 |   2.74 |  765 |  10.60 |  3.87x | n/a | -- |
|  64 | 65536 |   5.52 | 3039 |  63.12 | **11.44x** | n/a | -- |
| 128 |    64 |   1.92 |   17 |   3.11 |  1.62x | n/a | -- |
| 128 |  1024 |   2.62 |  200 |   4.19 |  1.60x | n/a | -- |
| 128 |  8192 |   3.87 | 1084 |  12.65 |  3.27x | n/a | -- |
| 128 | 65536 |  10.20 | 3290 |  79.26 |  7.77x | n/a | -- |
| 256 |    64 |   1.97 |   33 |   3.10 |  1.57x |  2.47 | 1.25x |
| 256 |  1024 |   2.89 |  363 |   4.14 |  1.43x |  5.80 | **2.01x** |
| 256 |  8192 |   5.74 | 1461 |  12.68 |  2.21x | 13.85 | **2.41x** |
| 256 | 65536 |  27.08 | 2478 |  81.29 |  3.00x | 39.21 | 1.45x |
| 512 |    64 |   2.38 |   55 |   3.26 |  1.37x |  3.62 | 1.52x |
| 512 |  1024 |   4.32 |  485 |   5.52 |  1.28x | 10.54 | **2.44x** |
| 512 |  8192 |  15.13 | 1108 |  21.62 |  1.43x | 27.72 | 1.83x |
| 512 | 65536 |  99.88 | 1343 | 149.34 |  1.50x | 111.33 | 1.11x |

**Faster than the vector kernel at every one of the 16 shapes** (geometric mean
2.31x, up to 11.44x) and faster than the two-sided kernel at every shape they
share (geomean 1.69x). N=512 is the size that changed: it was 0.58x - 0.78x of
the vector kernel and 0.54x of the two-sided one, and is now 1.28x - 1.50x and
1.11x - 2.44x. The kernel also covers N=64 and N=128, which the two-sided split
cannot reach at all (it needs `N = R * C` with both factors 16-aligned).

**The two-sided column here is the CLASSIC path**, which is what that kernel ran
when this table was measured, and what it ships again today. At N=512 it then
shipped the bit-plan tile instead,
and that reverses the last two rows: from batch 256 up the two-sided kernel is
1.02x - 1.70x FASTER, which is why `auto` routes N=512 by batch. N <= 256 is
unaffected -- the bit-plan has no entry there -- and the one-sided kernel still
wins every shape at those sizes. See the full grid under **Measured device
time**.

Against the previous dense-H version of this same kernel, paired with the tile
geometry and `block_dim` retune the change licensed: **1.33x - 1.56x at N=256,
1.64x - 2.55x at N=512**, 0.98x - 1.02x at `N <= 128` (no regression beyond 2%
at any shape; a null control -- the same `.so` under two names -- read 1.00x).

The GB/s column exceeds HBM at the small sizes because the working set is then
L2-resident (N=64 batch 65536 is 16.8 MB). Both arms read the same data.

It is also the most accurate of the three: fp32 accumulation rounded to fp16
exactly ONCE, measured 2.16e-4 worst relative RMS against an exact fp64
transform, against ~2.9e-4 for the two-sided split (two roundings) and 6.3e-4 to
8.4e-4 for the vector butterfly (`log2(N)` roundings). Blocking does not cost
accuracy: products of fp16 values accumulate exactly in fp32 here, so the
blocked and dense forms agree to every digit measured.

#### What capped it before, and what caps it now

The dense form reads all of H once per tile, so H's L1->L0B traffic per sample is
`N*N*2 / S` bytes against `4*N` bytes of HBM traffic for X and Y. With `S*N`
fixed by L0C capacity that ratio grows as `N^2`:

| N | dense H | where it had to live | H traffic vs data | dense result |
|---:|-------:|---|---|---|
| <= 128 | <= 32 KB | **L0B, whole kernel** | zero | 1.6x - 11.4x vs vector |
| 256 | 128 KB | L1 (too big for L0B's 64 KB) | ~1:1 | 1.0x - 2.1x vs vector |
| 512 | 512 KB | nowhere -- it IS all of L1, so GM per tile | ~4:1 | 0.6x - 0.8x vs vector |

The blocked form deletes that whole column: `+-H_M` is 16 KB at every N. What
binds instead is arithmetic. A noop-matmul probe (matmuls
deleted, every transfer and handshake kept) says so directly -- dropping the
matmuls is **3.96x** faster at N=512 batch 65536 and 2.34x at N=256, where
before the change the same probe read 0% and 7-8%. The kernel moved from
movement-bound to compute-bound, which is the intended direction: the bound it
now sits against is `N*N` MACs per sample, not a buffer size.

**That makes 512 a compute crossover rather than a capacity limit.**
The blocked form also computes N=1024 and 2048 correctly, which
correctly and the dense form could never attempt (H would be 2 MB and 8 MB).
They are off by default because the MACs grow 4x per doubling while the bytes
only double, and measurement agrees:

| N | batch | one-sided us | two-sided us | vector us | verdict |
|---:|------:|-------------:|-------------:|----------:|---|
| 1024 |  1024 |   9.56 |  5.71 |  13.54 | two-sided 1.7x |
| 1024 |  8192 |  52.77 | 26.53 |  84.35 | two-sided 2.0x |
| 2048 |  1024 |  31.50 |  8.04 |  26.13 | two-sided 3.9x; loses to vector too |
| 2048 |  8192 | 214.90 | 45.94 | 181.96 | two-sided 4.7x; loses to vector too |

So the guidance is now simple, which it was not before:

| N | use |
|---:|---|
| 64 .. 512 | one-sided |
| >= 1024 | two-sided |

#### Two tuning decisions that inverted, and are worth the warning

Both had been measured, were correct, and stopped being correct when the
bottleneck moved. Neither would have been caught by re-running the old A/B.

**Tile height.** The tall tile at N=512 (`PTO_OS_S512 = 64`) was justified
*solely* by halving H's per-tile traffic, and was worth 1.57x - 1.90x on that
basis. With H traffic at zero the justification is gone and the smaller tile's
deeper buffering wins instead: `S512` goes 64 -> **16** (1.24x at batch 65536)
and `S256` goes 64 -> **32**. `S256 = 16` is a cliff, not a continuation
(40.8 us against 27.1 at batch 65536).

**Block size M.** The argument for `M = 128` is that it is the largest that fits
L0B and that `B*B` matmuls shrink as `B = N/M` falls, so it issues the fewest.
That argument is wrong:

| N | batch | M=32 | M=64 | M=128 |
|---:|------:|-----:|-----:|------:|
| 512 | 65536 | 300.5 | **99.9** | 123.8 |
| 256 | 65536 |  40.8 | **27.2** |  27.8 |

`M = 64` is 3.01x over M=32 and 1.24x over M=128. Fewer and larger matmuls is
the wrong objective; what matters is that the `S x M` L0A slice stays small
enough to double-buffer deeply.

The `block_dim` schedule was refitted for the same reason -- `k` went
(8, 16, 16, 16) to (7.51, 7.23, 7.51, 22.57), and has since been refitted again
per tile tier (see the resweep below). **Sweep every integer `block_dim`,
not a subset.** A first fit over `{1,2,3,4,6,8,12,16,20,24}` picked constants
that emit `bd = 15` and `23`, values it had never measured, and scored them
against their nearest neighbour. It reported 1.013x - 1.031x worst regret; the
real cost at N=64 batch 8192 was 1.12x, because 32 tiles divide evenly over 16
cores and badly over 15. Refitted over all 24 values: worst regret 1.000x -
1.041x, against 1.18x - 1.21x for a constant 24.

#### The tile size and the core count are one knob, not two (1.033x overall)

A full resweep of the tile table after the geometry changed -- every power of
two from 16 up to the L1/L0A/L0C ceiling, crossed with `PTO_OS_BLOCK` in
{32, 64, 128}, each measured at **every reachable core count**. 32 variant
builds, ~2000 device-time points.

The resweep confirmed three of the four sizes unchanged (`S128 = 128`,
`S256 = 32`, `S512 = 16`, all at `M = 64`) and reproduced the numbers recorded
above to within 0.4% -- 40.65 us against 40.8 for `S256 = 16` at batch 65536,
26.93 against 27.1 for `S256 = 32`, 99.75 against ~99.9 for `S512 = 16`. A
resweep that cannot reproduce the fit it is replacing is measuring something
else.

**Sweep the tile size and the core count together, because at small batch they
are the same variable.** The tile height caps usable cores at
`ceil(batch / S)`, and below the crossover the measured optimum sits exactly on
that cap:

| N | batch | S | max cores | best cores | us |
|---:|------:|--:|----------:|-----------:|---:|
| 512 | 64 | 16 | 4 | 4 | **2.20** |
| 512 | 64 | 32 | 2 | 2 | 2.82 |
| 512 | 64 | 64 | 1 | 1 | 4.24 |

1.93x across that column, and none of it is tile efficiency -- it is three
quarters of the part sitting idle because the tile was too tall to divide. At
bulk the cap is slack, all 24 cores run whatever `S` is, and the ranking
inverts onto pipeline depth. Sweeping `block_dim` at a fixed `S`, or `S` at a
fixed `block_dim`, sees one regime and misses the other.

The clean way to sweep it is to parameterise by `(S, tiles_per_core)` rather
than `(S, block_dim)`: two `block_dim` values that produce the same split are
the same measurement, so they collapse, and every point is already an
active-core count.

**So the resident sizes carry two tiles and pick between them.** The optimum
crosses over sharply and monotonically, at both N:

| batch | N=64 S=128 | N=64 S=256 | | batch | N=128 S=64 | N=128 S=128 |
|------:|-----------:|-----------:|-|------:|-----------:|------------:|
|   192 |  **1.430** |      1.680 | |   128 |  **1.790** |       2.060 |
|   256 |  **1.500** |      1.720 | |   256 |  **1.880** |       2.060 |
|  1024 |  **1.720** |      1.840 | |  1024 |  **2.230** |       2.400 |
|  4096 |  **2.220** |      2.280 | |  4096 |      3.000 |       3.020 |
| 16384 |      3.180 |  **3.120** | | 16384 |      4.840 |   **4.610** |
| 65536 |      5.980 |  **5.320** | | 65536 |     11.800 |  **10.020** |
|262144 |     17.431 | **14.180** | |262144 |     39.511 |  **31.601** |

Either size alone loses at one end: 1.175x / 1.151x below the crossover for the
tall tile, 1.229x / 1.250x above it for the short one. The threshold is stated
in tiles, not samples -- take the tall tile once the batch yields
`PTO_OS_TILES_TO_PIPELINE = 32` of them, enough that 24 cores each get one and
most get two. That lands on both measured crossovers (32 x 256 = 8192,
32 x 128 = 4096) without a per-N magic number. The blocked sizes have no
crossover at all -- their short tile wins at every batch -- so they repeat one
value and `if constexpr` compiles the pick away, costing them no instantiation
and no branch. Cost of the two extra rings: `.so` 36488 -> 42760 bytes.

`k` was refitted per (N, tile), each tier scored only over the batches it
serves and against the clamped core count the launch really uses:
`_TILE_RUN_K` (8, 16, 16, 16) -> (7.51, 7.23, 7.51, 22.57) ->
**(7.51, 7.51, 15.02, 23.02)**, plus `_TILE_RUN_K_SHALLOW`
(5.52, 7.51, 15.02, 23.02). Worst-case regret 1.000x - 1.024x.

#### Launch only the cores that have work (1.0055x, +0.32% per idle core)

`tiles_per_core = ceil(tiles / block_num)`, so every core from
`ceil(tiles / tiles_per_core)` on finds `begin >= batch` and returns. The
schedule's `round(sqrt(k * tiles))` lands on such a value often -- at
`block_dim = 24` with 20 tiles, four cores are launched to do nothing.

Re-deriving `block_dim` from the active count is **exactly** split-preserving:
over all (tiles <= 5000) x (block_dim <= 24), `ceil(tiles / ceil(tiles / bd))`
leaves `tiles_per_core` unchanged in every case, so it cannot move a byte of
output -- it only stops paying for the empty blocks. One expression in the
host, and the three schedule tests (range, monotonicity, cap) still hold.

Measured on 64 pairs that differ **only** in the launched block count -- same
`.so`, same tile, same split -- it is worth **1.0055x geomean, +0.32% per idle
core, worst 0.991x** (one profiler quantum).

**The saving is on-device, not on the host.** `rtKernelLaunch` costs the host
the same whatever `blockDim` is; what the idle cores cost is block dispatch,
which is why msprof device time can see it. Worth taking, but it is not a
remedy for the ~46-59 us host launch.

It also composes with the tile size rather than adding to it: a taller tile
means fewer tiles, so more cores would have idled, so the clamp is worth more
exactly where the two-tier pick chooses the tall tile.

#### What the two changes are worth together (1.0328x over 100 shapes)

Old table and schedule against new, each at the `block_dim` its own host
picks -- the only honest comparison, since the tile size sets the tile count,
which sets what the schedule asks for, which the clamp then reduces.

| N | geomean | best | worst |
|---:|--------:|-----:|------:|
| 64 | 1.0444x | 1.135x | 0.953x |
| 128 | 1.0520x | 1.126x | 0.974x |
| 256 | 1.0109x | 1.071x | 0.994x |
| 512 | 1.0243x | 1.190x | 0.997x |
| **all** | **1.0328x** | **1.190x** | **0.953x** |

The gain sits where the shipped tile was too coarse to spread over cores --
batch 160 to 4096, 1.08x to 1.15x. The apparent sub-tile regressions do not
survive more reps: at 120 reps x 4 passes the whole band below 256 is
**1.0546x** geomean with a worst of 0.984x -- one 0.02 us quantum, not
reproducible. N=64 batch 16 read 0.953x at 40 reps and 0.984x at 120.
**Three passes is enough to rank a 1.1x difference and not enough to sign a
0.02 us one.**

#### The launch floor is the host, and it is 46 us against 1-8 us of kernel

Priced by `METH-360`'s protocol (host clock, synchronise *outside* the window,
`b = (t1-t0)/I` at `I = 400`), with device events for the wall column so a
captured graph is charged for its execution and not just its submission:

| N | batch | wrapper (host) | hoisted args (host) | wrapper (wall) | hoisted (wall) | captured graph (wall) |
|---:|------:|---------------:|--------------------:|---------------:|---------------:|----------------------:|
| 64 | 256 | 45.63 | 3.60 | 45.02 | 3.75 | **1.13** |
| 64 | 4096 | 45.58 | 3.61 | 45.51 | 3.77 | **1.80** |
| 512 | 1024 | 46.11 | 3.61 | 46.34 | 3.89 | **3.81** |
| 512 | 65536 | 46.21 | 3.57 | 99.67 | 99.54 | 99.53 |

Read the last row first: once the kernel costs ~100 us the three rungs are
indistinguishable, because the launch is hidden behind the device. Everywhere
else the wrapper's host cost **is** the runtime -- at N=64 batch 4096 the device
does 2.25 us of work behind 45.6 us of host, so the device-time wins recorded
above are worth ~0.15x of their size to a caller going through Python.
**Device time is still the right metric for judging the kernel** -- it is the
only one that can see a tile change at all -- but it is not the runtime, and
the gap is 20x.

Where the 46 us goes, same protocol, one component per row:

| step | us/call |
|---|---:|
| **full wrapper** | **47.26** |
| `resolve_stream_ptr` (task-queue drain) | **18.76** |
| `validate_onesided_hadamard_args` | 2.76 |
| `onesided_block_dim_schedule` | 2.00 |
| `torch_to_ctypes` x3 | 1.50 |
| `hadamard_full` (cache hit) | 0.51 |
| the launch itself, args hoisted | 3.77 |

The single largest item is the stream-pointer resolution, and it is **not**
removable as the wrapper is written: it is what drains torch_npu's task queue
so a raw ctypes launch is ordered behind the tensor setup. Dropping it
reintroduces the launch race.

**What was taken:** `onesided_block_dim_schedule` is now `lru_cache`d -- it is a
pure function of `(batch, n, max_block_dim)` sitting on the per-launch path.
Worth **1.104x / 1.106x** on the whole host path (46.34 -> 41.97 us,
46.49 -> 42.02), measured as a within-process paired A/B over six interleaved
alternations against the undecorated function. More than the 1.8 us the
component row suggests, because the memoised path also skips
`onesided_samples_per_tile`, the table lookups and the call frames.

**What was not taken, and what it is worth.** Both change the wrapper's
contract, so they are recorded rather than applied:

- **Hoist the arguments** (`METH-007` rung 1): 46 -> 3.6 us, **12.7x**. Needs a
  caller that reuses one tensor set, and a stream whose ordering is guaranteed
  some other way than a per-call drain.
- **Capture and replay** (`METH-007` rung 3): 1.13 - 3.81 us wall, i.e. close to
  device-bound. Needs a fixed launch sequence.

A note on the kernel side: this build is already `AI_CORE`, not `MIX_AIC` --
`--cce-aicore-arch=dav-c220-cube` is named in `compile_cpp`, and msprof confirms
`Task Type AI_CORE` with `Mix Block Num 0`. `STRUCT-001` records that an
unmarked cube-only kernel built with the *generic* arch pays +2.7 us at block
dim 1 rising to +4.4 us at 24; that cost is not being paid here, and marking the
`__global__` `__cube__` would be free insurance against the flag ever changing
rather than a saving available now.

#### A capacity cliff that does not announce itself: `kL0aSlots == 1`

The tile-size search ran into a configuration that compiles, runs, and then
hangs the device hard enough to kill the msprof session around it. When `S` is
tall enough that one L0A slice exceeds half the buffer, `kL0aSlots` quietly
falls to 1 and `l0aX(Par)` returns the **same** address for both parities --
while the ring's write-after-read edges still alternate `EV_A0` / `EV_A1` by
tile parity. Two event ids then guard one buffer, and the next tile's `TMOV`
overruns the `TMATMUL` still reading it.

Exactly two reachable configurations hit it, `N=64, S=512` and `N=128, S=256`,
both at the capacity ceiling and neither a contender. A
`static_assert(kL0aSlots == 2u)` turns a silent race into a compile error;
byte-identical `.so`, so it costs nothing. This is the mirror of the event-id
defect recorded below -- there one id carried two channels, here two ids
carried one buffer. Both are the same question asked backwards: **does every
(id, buffer) pair correspond one-to-one?**

#### The per-core remainder: one unit of work, not a chain of them (1.28x - 1.56x)

`runTiles` used to consume a core's span with a recursive power-of-two
**halving** chain: run `span/S` whole tiles, tail-recurse on `S/2`, down to 16,
and hand the last 1..15 rows to `partialTile`. It was correct, and in bulk it
cost **nothing** -- which is why it survived so long, and why a bulk A/B could
never have found the problem:

- Work is split by **tiles**, so every core except the last-with-work has a span
  that is a whole multiple of `S`. Those cores see `span == 0` at every lower
  level, compute `n == 0`, and never descend. The chain runs on **at most one
  core per launch**.
- That one core is the **truncated** one, so it is also the *under*-loaded one,
  and its serialization hides in its own slack. At N=64 the theoretical worst
  case is invisible: batch 65535, where the tail core walks `128+64+32+16` plus
  a 15-row partial, measures **5.46 us against 5.48** for batch 65536, which
  walks nothing.
- Depth also bounds it by size: at N=512 (`S = 16`) there is **no chain at all**,
  and at N=256 (`S = 32`) there is at most one extra level. Only N=64 and N=128
  have one worth walking.

What it cost was the regime where that slack is gone -- `tiles_per_core` of 1 or
2, where the descending core **is** the critical path. There every level is a
full prime/ring/drain for a single tile with no predecessor to overlap it, and a
255-sample span at N=64 walks five of them back to back.

The replacement makes a core's span exactly **one** unit of work:

- `span >= S`: `m = ceil(span/S)` tiles through the existing 4-deep ring, every
  tile at its natural position and every tile carrying its own row count,
  `min(S, end - start)`. A full tile is just the case where that min picks `S`,
  so there is no special last tile and the remainder is an ordinary ring tile —
  overlapped like any other, and a **full-size `S` tile with a short transfer**
  rather than a small tile of its own.
- `span < S`: **one** standalone masked tile, at the smallest instantiated tile
  size that covers the span.

There is now only **one** body. `runRing` handles any span >= 1, because its
last tile is masked, so `runTiles` decides nothing but the tile SIZE and the
hand-written `partialTile` that used to duplicate the transfers is gone.

That merge depended on the lazy priming below, and the order is the lesson: it
measured **1.05x - 1.16x slower** while the ring still primed its credits, and
became free the moment they went away. A sharing refactor that benchmarks badly
may only be blocked by a fixed cost somewhere else — fix that, then re-measure
rather than recording the refactor as rejected.

Four shapes of `runTiles` were measured against each other. All are parity in
bulk; they differ only on short and unaligned spans:

| `runTiles` | `.so` | where it loses |
|---|---:|---|
| no size pick: one ring at the top-level S for every span | 32936 | **1.09x - 1.22x** below ~100 samples |
| full tiles in the ring, remainder split off at its own size | 39208 | **1.14x - 1.22x** on mixed spans at N=64 |
| size pick folded in, every level entering `runRing` | 50440 | nothing, but +38% code |
| **size pick folded in, recursed levels run one tile** | **36520** | **nothing** |

The shipped form is two functions: `runTiles` chooses between a ring and a
short span, and `runShortSpan` picks the tile size for the latter. An earlier
revision folded both into one function behind a `bool Top` template parameter;
splitting them is **byte-identical codegen**.

Two of those are instructive.

**Dropping the size pick** is the smallest and simplest code and it loses,
because **masking is free on the transfers but not on the tile**: the cube runs
all `S` rows whatever the row count says, so a 1-row span at N=128 still pushes
128 rows through MTE1 and the cube.

**Splitting the remainder off** so it runs at its own smaller size — 7 wasted
rows instead of 135 — also loses, and badly. The reason is that there was
nothing to win in the first place. Forcing one core to take a whole batch
(`block_dim = 1`) at N=64 prices the near-empty last tile directly:

| batch | tiles | device us |
|------:|-------|----------:|
| 256 | 1 | 1.580 |
| 257 | 2, the second holding **one row** | **1.580** |
| 272 | 2, the second holding 16 rows | 1.580 |
| 320 | 2 | 1.640 |
| 511 | 2 | 1.821 |

A second 256-row tile carrying a single row is **free**: its 255 wasted cube
rows hide entirely behind the first tile's movement, and cost only appears once
the tile carries real rows. At N=128 the step is not zero (+0.140 us for the
second tile at one row), but the handoff below cancels it. The remainder stops overlapping,
and since the ring no longer drains itself it needs an explicit three-edge
handoff (`FIX->M`, `M->MTE1`, `MTE1->MTE2`) before the tail tile may touch L0C,
L0A and L1. That is a full pipeline flush in front of a tile that then overlaps
nothing. The wasted rows are hidden — the kernel is movement-bound at that size
— and the flush is not. Skipping the handoff is not an option: it faults the
cube outright with *multi-bit ECC error when FIXP read L0C*, M overwriting the
accumulator FIX is still storing from.

The shipped form keeps the remainder in the ring and exploits one fact to stay
small: a **recursed** level was entered because `span <= S`, and stayed because
`span > S/2`, so it is always **exactly one tile**. It calls a one-tile body
instead of the ring, and `runRing` — its loop and all four slot/parity
combinations — is instantiated only once, at the top. That is where the 38%
goes.

Both ends of the earlier *arm* split were measured too, and each loses at the
other's end of the range:

| shape of `runTiles` | mixed span (whole tiles + remainder) | span < one tile |
|---|---|---|
| ring over whole tiles, remainder standalone | **1.045x / 1.071x slower** at N=64 | parity |
| every span through the ring | parity or faster | **1.05x - 1.11x slower** |
| **the split above** | **parity or faster** | **parity** |

0 regressions over 29 shapes against the first shipped form of this change, and
0.974x - 0.99x on the N=64 mixed cases. Earlier revisions of this kernel instead
backed the last tile up to `end` and recomputed the overlap; masking it is the
same speed, duplicates no rows, and drops the address-rewriting helper the
back-up needed.

msprof device time, median of 3 interleaved passes of 30 reps, scheduled
`block_dim`, old -> new:

| N | exposed case | before | after | | aligned / bulk control | before | after |
|---:|---|---:|---:|---|---|---:|---:|
| 64 | batch 255 | 2.62 | **1.68** (1.56x) | | batch 256 | 1.78 | 1.78 |
| 64 | batch 511 | 2.66 | **1.76** (1.51x) | | batch 65536 | 5.40 | 5.40 |
| 128 | batch 127 | 2.68 | **1.94** (1.38x) | | batch 128 | 2.18 | 2.18 |
| 256 | batch 17 | 2.14 | **1.56** (1.37x) | | batch 65536 | 27.02 | 27.02 |
| 512 | batch 15 | 2.64 | **1.94** (1.36x) | | batch 16 | 2.10 | 2.12 |
| 512 | batch 31 | 2.66 | **2.08** (1.28x) | | batch 65536 | 99.80 | 99.76 |

Over a 115-case sweep: 53 cases at >= 1.05x, 60 inside 0.98 - 1.02x, **none
below**. Run-to-run spread is 0.02 - 0.06 us, and the profiler quantises to
~0.02 us, so every "parity" cell above is parity to the resolution available.
The `.so` shrinks 22% (49352 -> 38568 bytes): one ring form instead of five
recursive instantiations per N.

**Only about a third of the win is the chain.** Three costs were measured apart,
and the other two are independent of how the span was divided:

1. The chain itself (N=64/128 only) -- 1.26x - 1.56x.
2. **`partialTile`'s blocked body was serialized**: it synced `M -> MTE1` after
   every one of the `B` blocks. One tile is the whole of a short-span core's
   work, so that *is* the critical path. Using the ring's own `blockStep`
   instead is **1.35x at N=512 and 1.14x at N=256** -- at N=512, where no chain
   ever existed. This was the cheapest single fix on the list.
3. The remainder being a drained standalone unit when the core also has whole
   tiles; folding it into the ring is worth 1.20x - 1.34x. **This is the one
   that decides the shape of `runTiles`** -- it is why the remainder is a
   masked ring tile and not a tile of its own.

**Halving back down is not the cheap alternative it looks like.** A 16-row ring
tile costs ~0.11 us at the margin, so the 15 small tiles covering a 255-sample
span lose to four serialized big ones. A two-level form (`S` then 16) measured
**0.78x - 0.86x** -- slower than the chain it was meant to fix -- and so did a
form that covered the remainder with extra 16-row tiles (0.86x - 0.91x). A
short span wants ONE big masked tile.

**Masking is free on the transfers but not on the tile**: the cube runs all `S`
rows whatever the row count says. A 1..15 row span in an `S = 256` tile wastes
240 rows of TMOV and MACs -- 0.19 us at N=64 and 0.32 us at N=128, 15-20% of a
short-span core, which is why the masked tile picks its level instead of always
using the top one. Masking at the top level alone **regresses** batch 1..15 to
0.83x - 0.85x.

Two consequences for things recorded elsewhere in this file. The sample split
(`PTO_OS_TILE_SPLIT`, in **Switches removed for release**) was rejected because
it "walks the halving chain on most cores" -- there is no chain now, so that
rejection no longer holds and the split is safe to revisit. A single-core probe
puts a masked tile at `1.42 us + 1.26 ns/row`, and the new N=64 batch-255 figure
sits exactly on that line, i.e. it is already at the single-core movement floor;
spreading those 255 rows over the 3 cores the schedule launches could save at
most ~0.20 us, and only if bandwidth scaled with cores, which **Why N=256 is the
weak case** shows it does not in the shallow regime. Worth 1.1x at best.

#### Credits are primed lazily, not up front (1.02x - 1.13x)

The ring used to open by asserting that all four X slots, both accumulators and
both L0A slots were free, and to close by draining whatever nobody consumed --
**16 flag ops a core pays whether it runs 1 tile or 200**. A credit means "this
resource is free again", which is only worth saying when someone had it.

So nothing is primed. Each tile waits for a slot only if an earlier tile used
it, and signals it free only if a later tile will reuse it:

```
tile j's load waits   iff  j >= 4          (it reuses tile j-4's slot)
tile i's body signals iff  i + 4 <= last   (tile i+4 exists to reuse it)
```

The pairing is exact -- tile `j`'s wait matches tile `j-4`'s set, one for one --
and tile `j`'s load is issued one iteration *after* tile `j-4` was consumed, so
the credit is always produced before it is awaited. The same argument covers the
accumulator at depth `kAccSlots` and resident L0A at depth 2. The blocked body
is the exception and keeps its primed pair: its L0A credits rotate between the
`B` blocks *within* a tile, so the run's first two blocks have no predecessor.

The saving is a constant, so the win is set by how little work it is spread
over -- which is what the measurement shows, and the monotone decay with tiles
per core is the evidence that the mechanism is the one claimed:

| tiles per core | shapes | speedup |
|---|---|---|
| 1 | N=64 b256, N=128 b128, N=256 b32, N=512 b16, ... | **1.06x - 1.13x** |
| 2 - 3 | N=128 b2048, N=256 b1024, N=512 b1024 | 1.05x - 1.10x |
| 7 - 22 | N=256 b5000, N=512 b5000, N=64 b65536 | 1.02x - 1.04x |
| 86 - 171 | N=256 / N=512 b65536 | 1.00x |

18 of 21 shapes beyond 2%, none slower; `.so` 38120 -> 39272 bytes. Because this
is a sync-protocol change -- the class of defect recorded under *one event id
must not carry two channels out of one pipe*, which was intermittent and
shape-dependent -- it was gated on **0 failures in 15 full correctness sweeps**
(84 shapes x 4 `block_dim` each, bit-exactness plus fp64 residual), not on one
green run.

Note the generalisation, because it is not specific to this ring: a double
buffer's prime/drain is a *fixed* cost, so it is invisible in the bulk A/B that
usually justifies the buffering and it dominates exactly where runs are short.
The same reasoning is why a lone tile does not go through the ring at all.

#### Auditing the flags: balance, liveness, and the one thing neither catches

Three properties, each needing its own check, all run over the same 520
control-flow paths (`runRing` trip counts 1..64 x resident/blocked x
`kBlk` in {2,4,8} x `kAccSlots` in {1,2}, plus the single-tile path):

| property | question | result |
|---|---|---|
| balance | per `(src, dst, id)`, does `#set == #wait`? | yes, on every path |
| liveness | does any wait precede the credit that satisfies it? | no |
| aliasing | can a wait pop a credit posted by a *different* channel? | no, but see below |

Balance is the rule every flag must obey, and the lazy-credit form above is
what makes it provable: the producer guard and the consumer guard are the same
proposition written at both sites, so they pair off by construction rather than
by counting. Liveness is separate -- balanced counts do not make a schedule
live -- and holds because each credit is produced an iteration before it is
awaited.

Aliasing is the one a balance check cannot see, because `Sync<A,B>()` is a set
and a wait on one line and so is self-balancing on every path. The hardware
keys a credit on the **source pipe and id**, not on the destination, so two
channels out of one pipe sharing an id let a waiter take the other's credit and
run ahead -- the defect recorded above under *one event id must not carry two
channels out of one pipe*. Grouping this kernel's ids by source pipe:

| source | id 0-3 | id 4,5 | id 6 | id 7 |
|---|---|---|---|---|
| MTE2 | -> MTE1 (X data ready) | | | |
| MTE1 | -> MTE2 (L1 slot free) | | | -> M (TMOV to TMATMUL) |
| M | -> FIX (acc ready) | -> MTE1 (L0A write-after-read) | | |
| FIX | | | -> M (acc slot free) | |

`Sync<MTE1,M>` used to spend the default id 0, which `(MTE1 -> MTE2)` also
carries. That was **not** a live bug: the slot-free `set_flag` is guarded by
`i + 4 <= last` at tile `i`, the prefetch wait that consumes it by
`(i + 4 <= last)` at tile `i+1`, and the prefetch runs before the tile body --
so the credit is always taken one instruction group before any `Sync` can
reach it, and the aliasing simulation finds zero steals. But that safety rests
on an invariant nothing states or enforces: change the prefetch distance, move
the `set_flag`, or add one `Sync` and it becomes the recorded defect, which was
intermittent (4 failures in 30 repeats of one shape) and passed 252 validation
cases first.

So `Sync<MTE1,M>` was given `EVENT_ID7`. Against a byte-identical null arm in
the same profiling session the change is inside the resolution floor -- null
geomean 0.996 with per-cell spread 0.973-1.024, measured 1.004 with spread
0.987-1.031, and the sign flips between sessions -- while `.aicore_binary`
drops 384 bytes and `.text` is unchanged. Gated like the other sync-protocol
change on 0 failures in 15 full sweeps.

`Sync` also lost its `id = 0` default, which is what made the collision easy to
write in the first place: the id decides which channel the call lands on, so a
default silently puts every `Sync` on the one id most likely to be in use
elsewhere, and the call site stops saying which channel it is. With the id
required, the table above can be read straight off the source. The `.so` is
byte-identical before and after that change, so it is purely a source-level
guard.

#### Three gotchas worth recording

**The straight pipe needs WAR flags the turnaround body got for free** -- the
flip side of the whole design.
The straight-pipe body has no edge holding MTE1 back, so MTE1 runs ahead by
slots and tile `i+2`'s `TMOV` lands on the L0A slot tile `i`'s `TMATMUL` is still
reading. The two-sided kernel gets that WAR protection for free from its own L1
turnaround; here it has to be an explicit `M -> MTE1` flag per slot. Without it
the kernel faults, at a shape that moves between runs. A second instance of the
same class: staging an H chunk with the generic `Sync<PIPE_MTE2, PIPE_MTE1>()`
helper, whose id 0 collides with `EV_X0` on that very pipe pair -- only the
streamed-H case runs that Sync inside the tile loop, so it corrupted the X ring's
credits at N=512 and nowhere else.

**A broken knob is not a rejected technique.** The first `M = 64` A/B computed
garbage (relative RMS 1.41, i.e. uncorrelated), which reads like "small blocks
don't work". It was a host/device disagreement: `PTO_OS_BLOCK` sets M in the
kernel while `_BLOCK` in the wrapper decides which `+-H_M` to build, and only
one of the two had been changed, so the kernel was multiplying by a wrong-sized
operand. With both set, M=64 validates -- and then wins by 1.24x. The two are
now coupled through one env override precisely so the A/B cannot drift again.

**Evaluation order can race the operand buffer.** `hadamard_full` enqueues torch
ops on a cache miss, and the raw ctypes launch is only ordered after them because
`resolve_stream_ptr` drains torch_npu's async task queue first. Python evaluates
call arguments left to right, so building H *inline after* the stream pointer
enqueues those ops after the drain, and the kernel can read H before it is
written: `Y = X * 0 = 0`, on the first call for each `n` only. Latent before this
change and near-certain after it, because a `cat` of three pieces is a wider
window than one `.to(device)`. Fixed by hoisting the lookup above the call, with
a cache-clearing regression test -- the correctness sweep cannot catch it,
because by then every `n` is warm.

The two tuning constants are documented at the top of the kernel;
`cube/devtime_bench.py` builds one combination and
`cube/test_hadamard_onesided.py` checks a prebuilt variant against the exact fp64
reference over 168 cases. Validate before timing: a variant that computes the
wrong answer is a broken lever, not a slow technique.

### Back-porting the one-sided kernel's changes to the two-sided one

Every change the one-sided kernel took after the `d2d9a59` restructure, checked
against `cube/fast_hadamard_twosided.cpp` and **measured rather than argued**.
Four of the six ported, one does not apply, and one is recorded with its
numbers and left on the shelf.

| one-sided change | ports? | measured on the two-sided kernel |
|---|---|---|
| `lru_cache` on the `block_dim` schedule | **yes** | **1.0847x** host path (1.117x where it is called twice) |
| launch only the cores that have work | **yes** | **1.0162x** device, **+0.50% per idle core** |
| credits primed lazily | **yes**, after the chain | **1.0289x** device, geomean over 40 shapes |
| per-core remainder as one masked ring tile | **yes** | with the above, **1.0972x** / best **1.6339x** / worst **0.9994x** |
| tile size and core count are one knob | applies | 1.05x - 1.28x, **batch <= 64 only**; not taken, see below |
| `static_assert(kL0aSlots == 2)` | **no** | the defect cannot occur here |

Combined, the two kernel-side changes are **1.0972x geomean over 176 shapes,
96 of them at >= 1.05x, none below 0.98x**, and the `.so` shrinks 17%
(75944 -> 63336 bytes). The host-side changes are a further 1.085x on the
launch path and 1.016x on device.

A function-by-function audit and a tile resweep then followed, and the two
subsections below them -- the ND operand gather and the three-height tiling with
its tiles-per-core schedule -- are worth more than everything above. **Over the
whole of it: 1.2043x geomean over 134 shapes, best 2.9096x, worst 0.9638x**
against a null arm of 1.0001 (band 0.9863 - 1.0379), 63 shapes at >= 1.05x and
2 below 0.98x (both small-batch, at N=1024 and N=2048, marginally outside the
null). Per N: **N=256 1.7772x, N=512 1.8360x**, everything else 1.02x - 1.07x.

The rig's floor first: 19 shapes, two byte-identical `.so` under different
names, interleaved in one profiling session -- geomean **0.9997**, band
**0.9933 - 1.0050**. Everything below is read against that.

#### The two host-side changes port unchanged

`twosided_block_dim_schedule` is a pure function of `(batch, n, max_block_dim)`
on the per-launch path, so it memoises exactly as the one-sided one does.
Priced by `METH-360` (host clock, synchronise outside the window, `b =
(t1-t0)/I` at `I = 400`), as a within-process paired A/B over six interleaved
alternations:

| N | batch | plain | memoised | ratio | schedule calls per launch |
|---:|------:|------:|---------:|------:|---:|
| 256 | 1024 | 49.63 | 46.35 | 1.0706 | 1 |
| 512 | 1024 | 54.46 | 48.74 | **1.1173** | **2** |
| 1024 | 1024 | 51.32 | 47.40 | 1.0827 | 1 |
| 4096 | 256 | 51.17 | 47.16 | 1.0849 | 1 |
| 16384 | 64 | 50.38 | 46.75 | 1.0776 | 1 |
| 32768 | 32 | 49.78 | 46.29 | 1.0754 | 1 |

Geomean **1.0847x**. N=512 is the outlier and it is the mechanism confirming
itself: that is the one size whose launch splits into a bit-plan bulk plus a
remainder, so the wrapper calls the schedule **twice**, and it gains most.

The `block_dim` clamp is the same one expression, and it is worth **more** here
than in the one-sided kernel because this schedule's `k` is larger relative to
the tile count. It fires on **118 of 272** shapes, shedding up to **8** idle
cores (N=32768 batch 22 asks for 19 and eleven have work). It is exactly
split-preserving for this kernel's split too: over all
(tiles <= 6000) x (block_dim <= 24), `ceil(tiles / ceil(tiles / bd))` leaves
`tiles_per_core` unchanged in every case, and every active core's `[begin, end)`
with it.

50 pairs that differ **only** in the launched block count -- same `.so`, same
tile, same split:

| idle cores | pairs | geomean |
|---:|---:|---:|
| 1 | 7 | 1.0036 |
| 2 | 14 | 1.0071 |
| 3 | 7 | 1.0204 |
| 4 | 14 | 1.0200 |
| 5 | 4 | 1.0406 |
| 7 - 8 | 3 | 1.0271 |

**1.0162x geomean, +0.503% per idle core**, best 1.0797x, worst 0.9806x,
against 10 null pairs at 0.9966x (0.9837 - 1.0090). The monotone trend in idle
count is the evidence that the mechanism is block dispatch and not noise.

One thing the port surfaced: `tiles == 0` makes the clamp divide by zero, which
`test_batch_partition_boundaries` catches because it covers `batch = 0`. The
one-sided kernel carries the same guard, so this is its guard, not a new one.

#### Lazy credits port, but the drain was load-bearing and cost 1.25x to replace

The deep ring primes four `(MTE1 -> MTE2)` slot credits and drains four, and
`runTwosided` primes and drains `EV_ACC` -- which the deep path, six of the
eight sizes, never uses at all. Ten flag ops a core pays whether it runs one
tile or two hundred, the same fixed cost the one-sided ring shed.

Removing them the obvious way **fails 157 of 864 correctness cases, every one of
them at an unaligned batch**. The four drains were not bookkeeping: they were
the handoff between levels of the power-of-two halving chain, the one place
where something *follows* a ring and reuses its L1 slots. Priced on aligned
batches, where no lower level runs and the defect cannot fire, the credits alone
are worth **1.0271x** over 40 shapes, decaying monotonically with how much work
the constant is spread over:

| tiles per core | speedup |
|---|---|
| 1 | 1.03x - 1.07x |
| 2 | 1.02x - 1.06x |
| 8 | 1.01x - 1.03x |
| 86 - 2048 | 1.000x |

So the drain has to be replaced by an explicit handoff, and **how many edges it
needs is a measurement, not a derivation**. The one-sided README names three
(`FIX -> M`, `M -> MTE1`, `MTE1 -> MTE2`); only the last is load-bearing here,
and the other two are pure cost:

| handoff | geomean (40 shapes) | worst |
|---|---:|---:|
| `MTE1 -> MTE2` only | **1.0289x** | **0.9821x** |
| `+ M -> MTE1` | 1.0185x | 0.9521x |
| `+ FIX -> M` as well | 0.9723x | **0.7929x** |

Three barriers in front of a tile that then overlaps nothing turn a 1.03x win
into a 1.1x - 1.25x loss on exactly the spans they run on. Take the cheapest
handoff that validates, and price each edge rather than copying the set.

#### The halving chain, replaced by one masked ring tile (1.0972x, best 1.6339x)

`runTiles` consumed a core's span with the same power-of-two halving chain the
one-sided kernel deleted -- 43 samples as 32 + 8 + 2 + 1, each level priming and
draining its own slot flags. It is exposed here for the same reason and to the
same degree. Measuring an unaligned batch against the aligned one just below it,
so the unaligned case does strictly **less** work:

| N | S | batch | aligned us | unaligned batch | us | cost |
|---:|--:|------:|-----------:|----------------:|---:|-----:|
| 256 | 32 | 32 | 2.340 | 31 | 3.180 | **1.36x** |
| 256 | 32 | 256 | 7.160 | 255 | 8.000 | 1.12x |
| 512 | 16 | 16 | 2.700 | 15 | 3.320 | **1.23x** |
| 1024 | 8 | 8 | 1.880 | 7 | 2.190 | 1.17x |
| 2048 | 4 | 4 | 1.900 | 3 | 2.000 | 1.05x |
| 4096 | 2 | 2 | 2.020 | 1 | 1.760 | none |

A two-sided tile is S samples of R matrix rows each, so a row count is always a
multiple of R and stays 16-aligned whatever the sample count is -- which makes
`MatTileV` / `SetValidRow` and a `DYNAMIC`-row `GlobalTensor` enough to carry
any remainder in a full-size tile. `runTiles` becomes one ring over
`ceil(span/S)` tiles with the last one masked, and the whole chain, its level
handoffs and its four recursive instantiations per N go away.

Two things had to be got right, and each was found by measuring, not by reading.

**A masked tile must keep its LAYOUT.** Falling back to `GmTileDyn` (ND) for the
short tile -- which looks free at N=256, where `C == 16` makes the NZ image the
row-major bytes -- costs **0.49x - 0.78x there and nothing anywhere else**,
because a 16-column ND transfer is a 32-byte-per-row gather rather than one
burst. N=256 is the only size in the table with `C == 16`, which is exactly why
the regression was confined to it. Keeping the burst with a `DYNAMIC`-row NZ
tensor is worth **1.1129x geomean, best 3.2222x**. The two layouts index their
shape differently and the difference is easy to get silently wrong:
`TileShape2D<..., ND>` is `Shape<1,1,1,rows,cols>`, so the row count is
**DIM_3** and counts rows; `TileShape2D<..., NZ>` is
`Shape<1, cols/C0, rows/16, 16, C0>`, so it is **DIM_2** and counts **fractals**.

**A short span must pick its tile size.** Masking at the top `S` alone regresses
`batch = 1` to **0.8655x - 0.9783x** at every size with `S > 1`: the cube runs
all S samples whatever the row count says, and a lone tile has no predecessor to
hide the waste behind. Inside the ring the same waste **is** hidden, which is
why only the short-span path picks a size and the ring's last tile does not.
The size pick is worth **1.0156x geomean, best 1.3008x**, and it is what takes
the worst case from 0.8655x to 0.9994x.

Combined, against the shipped kernel, msprof device time, median of 3
interleaved passes of 40 reps at each shape's own scheduled `block_dim`:

| N | S | shapes | geomean | best | worst | sub-tile band | aligned / bulk |
|---:|--:|---:|---:|---:|---:|---:|---:|
| 256 | 32 | 47 | **1.1983** | 1.6339 | 1.0090 | 1.2310 | 1.1375 |
| 512 | 16 | 32 | 1.0998 | 1.3822 | 0.9994 | 1.1255 | 1.0776 |
| 1024 | 8 | 23 | 1.0690 | 1.2584 | 1.0017 | 1.1116 | 1.0509 |
| 2048 | 4 | 20 | 1.0699 | 1.2757 | 1.0000 | 1.1029 | 1.0642 |
| 4096 | 2 | 18 | 1.0456 | 1.1406 | 1.0055 | 1.0976 | 1.0426 |
| 8192 | 1 | 12 | 1.0456 | 1.0962 | 1.0007 | -- | 1.0456 |
| 16384 | 1 | 12 | 1.0299 | 1.0642 | 1.0025 | -- | 1.0299 |
| 32768 | 1 | 12 | 1.0172 | 1.0478 | 0.9995 | -- | 1.0172 |
| **all** | | **176** | **1.0972** | **1.6339** | **0.9994** | | |

**96 shapes at >= 1.05x, 80 inside 0.98 - 1.05x, none below 0.98x.** The gain is
monotone in S, which is the signature of the chain: S is what set its depth. The
three `S == 1` sizes have no chain at all, so their 1.017x - 1.046x is the lazy
credits on their own.

Because this is a sync-protocol change -- the class of defect that is
intermittent and shape-dependent -- it was gated on more than one green run:
**18600 shape-checks over 15 sweeps** of 1240 shapes (8 N x 31 batches x 5
`block_dim`), each sweep required to be **bit-identical** to the first and the
first checked against an exact fp64 transform. 0 non-deterministic, 0 out of
tolerance, plus an 864-case fp64 sweep and the repo's 265 tests.

#### The operands were loaded as a narrow-tile gather, and every core repaid it (1.1819x overall, 1.84x at N=256)

`loadHadamardFactors` read its `(W x W)` factors through `GmTile`, i.e. ND. A
factor is small, so that is a `W`-row, `2W`-byte-per-row ND2NZ gather -- at
N=256, **16 separate 32-byte reads** of a 512-byte buffer. It is the narrow-tile
defect this file already records for the X tile at `C == 16` (4.5x - 8.7x),
applied to the OPERAND instead of the data, with one extra consequence: the
factors are loaded **once per core**, before MTE1 may touch that core's first
tile, so the cost is paid again for every core in the launch.

The fix is four `GmTile` -> `GmNz` substitutions plus a host `_to_nz` that
stores each operand block pre-fractalised. Measured in isolation at N=256, each
shape at its shipped `block_dim`, against a byte-identical null arm at 0.9997
(band 0.9907 - 1.0048):

| batch | 32 | 256 | 1024 | 4096 | 8192 | 65536 |
|---|---:|---:|---:|---:|---:|---:|
| ND -> NZ | 1.335 | 1.403 | 1.542 | 1.651 | 1.624 | 1.455 |

**It is also the whole of "why N=256 is the weak case".** That section attributes
the size's refusal to scale to pipeline depth, and the core-count experiment it
reports -- 48 tiles, every reachable core count -- says otherwise once the
gather is gone:

| cores | 1 | 2 | 4 | 6 | 8 | 12 | 16 | 24 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ND operands | 16.02 | 9.42 | 7.04 | **7.02** | 7.56 | 9.33 | 11.40 | 15.72 |
| NZ operands | 15.43 | 8.48 | 5.00 | 3.93 | 3.40 | 2.97 | 2.80 | **2.73** |

The U-curve is not a property of the kernel. It is 24 copies of a 0.5 - 0.7 us
gather, and with the burst the size scales **monotonically to 24 cores**:
2.571x at each arm's own best count, 5.76x at `bd = 24`.

**So the schedule was encoding the defect.** `_TILE_RUN_K` was 0.49 at N=256 and
0.6 at N=512 against >= 8 at every other size -- the two sizes whose factors are
16 and 32 wide. Those constants existed only to stop adding cores, because each
extra core repaid the gather. Keeping them on the fixed kernel costs **1.6210x /
1.7398x** worst-case regret; refitting lands both on the same value as everything
else (7.51 / 7.23, worst regret 1.0106 / 1.0000). *A stale schedule was the
larger half of this change*, and the resweep below then replaced the fit's
functional form outright.

End-to-end, old kernel at its own schedule against new kernel at its refit one,
126 shapes, 3 interleaved passes:

| N | shapes | geomean | best | worst |
|---:|---:|---:|---:|---:|
| 256 | 16 | **1.8382** | 2.8885 | 1.3380 |
| 512 | 16 | **1.8024** | 2.6604 | 1.1794 |
| 1024 .. 32768 | 94 | 1.0111 - 1.0372 | 1.0619 | 0.9986 |
| **all** | **126** | **1.1819** | **2.8885** | **0.9986** |

42 shapes at >= 1.05x, **none below 0.98x**, null arm 0.9993.

**Correctness is a host/kernel contract, and it hides its first failure.** A
16-wide factor's NZ image IS its row-major bytes, so the kernel-only change
validates at N=256 (108/108) and the host side looks unnecessary. The first size
to break is N=512, and only on its `H_C`: without `_to_nz` it reads
`rel_rms` **1.366**, i.e. uncorrelated. With both halves, 864/864 fp64 cases pass
at worst `rel_rms` 3.264e-04, identical to the baseline, and the repo suite
passes including the bit-plan path -- `_bit_operand` feeds the very same loads,
so it needed the same treatment.

The two layouts also index their shape differently, which is easy to get
silently wrong: `TileShape2D<..., ND>` is `Shape<1,1,1,rows,cols>`, so the row
count is **DIM_3** and counts rows, while `TileShape2D<..., NZ>` is
`Shape<1, cols/C0, rows/16, 16, C0>`, so it is **DIM_2** and counts **fractals**.

A note for the sibling kernel: `fast_hadamard_onesided.cpp` loads its operands
as ND too, but its narrowest is `kM = 64` (a 128-byte row) against 16 here, and
the measurements above put a 64-wide factor at 1.00x - 1.05x. So the same defect
exists there and is worth little -- the severity is set by the row length, not by
the pattern.

#### Taken on a resweep: three tile heights and a tiles-per-core schedule (1.0343x over 135 shapes)

An earlier version of this section recorded the crossover as measured,
applicable and **not** taken, because N=256 appeared to cross at 2 tall tiles
while N=1024 crossed at 32, so no single threshold constant could serve both.
That sweep ran before `loadHadamardFactors` read its operands as `Layout::NZ`.
On the fixed kernel the crossovers line up, and the whole thing ships. **A
tuning conclusion is only as current as the kernel it was measured on.**

Resweep: every candidate `S` from the table down to 1 -- six builds, each
halving the whole case table -- crossed with **every reachable active core
count**, over batch 1..256 densely plus 1024 and 8192. 5730 timed
configurations: 1910 (S, cores, batch) points x 3 interleaved passes, medians.
Each `S` at its own best count, `us@cores`:

| batch | N=256 S=32 | S=16 | S=8 | S=4 | N=1024 S=8 | S=4 | S=2 | S=1 |
|------:|-----------:|-----:|----:|----:|-----------:|----:|----:|----:|
| 8 | 1.360@1 | 1.320@1 | 1.340@1 | 1.330@2 | 1.740@1 | 1.500@2 | 1.480@2 | **1.460@4** |
| 16 | 1.420@1 | 1.410@1 | 1.350@1 | **1.340@2** | 1.760@2 | 1.600@4 | **1.580@4** | 1.660@8 |
| 32 | 1.640@1 | 1.460@2 | **1.380@2** | 1.440@4 | 1.880@4 | **1.760@4** | 1.780@8 | 1.960@16 |
| 64 | 1.640@2 | 1.520@2 | **1.480@4** | 1.660@8 | 2.060@8 | **1.960@8** | 2.080@16 | 2.410@16 |
| 128 | 1.720@4 | **1.640@4** | 1.690@8 | 1.970@16 | 2.320@8 | **2.280@16** | 2.590@16 | 2.980@22 |
| 256 | 1.920@4 | **1.821@8** | 2.000@16 | 2.380@16 | **2.630@16** | 2.820@22 | 3.240@22 | 3.890@24 |
| 1024 | **2.430@16** | 2.730@24 | 3.130@24 | 3.960@24 | **4.460@24** | 5.070@24 | 6.670@24 | 9.300@24 |
| 8192 | **5.500@24** | 7.140@24 | 10.340@24 | 16.790@24 | **18.601@24** | 25.041@24 | 38.481@24 | 62.041@24 |

Three readings, and the third is the one that was missing before:

**The shipped `S` is right at bulk and wrong below it, by up to 1.19x.** At
batch >= 1024 it wins at every N and every core count; the whole opportunity is
batch <= 512 and it is worth 1.05x - 1.19x there. (Pre-NZ-fix the same table
read 1.284x at N=256 batch 16; the operand fix took most of that back, so the
numbers here are smaller than the ones they replace -- a fix in one place
shrinks the case for a change in another.)

**The S-vs-S/2 crossover IS at one tile count: 16 tall tiles.** Measured
intervals, in tall tiles: N=256 [8, 32], N=512 [16, 64], N=1024 [16, 24],
N=2048 [12, 16], N=4096 [8, 12]. 16 is in all five. The earlier "2 versus 32"
reading came from comparing different pairs of sizes on the pre-fix kernel.

**The cascade is exactly halving, and three heights is the depth.** The
per-batch optimum over every `S` is `S`, `S/2`, `S/4` across the whole range
those two thresholds select them in, so the tiers need no table of their own.
Scored against the per-batch oracle over 90 (N, batch) points:

| tile heights | thresholds | regret geomean | regret worst | gain vs 1 tier |
|---|---|---:|---:|---:|
| 1 (shipped before) | -- | 1.0553 | 1.1918 | 1.0000 |
| 2: S, S/2 | 16 tall | 1.0155 | 1.0859 | 1.0393 |
| **3: S, S/2, S/4** | **16 tall, 6 mid** | **1.0092** | **1.0846** | **1.0457** |
| 4: + S/8 | 16, then 6 per level | 1.0031 | 1.0625 | 1.0521 |
| 5: + S/16 | 16, then 6 per level | 1.0052 | 1.0625 | 1.0499 |
| 7: full cascade | 16, then 6 per level | 1.0055 | 1.0606 | 1.0495 |

**The depth is a measurement too, and it has an interior optimum**: a FIFTH
height is worse than four (1.0052 against 1.0031) -- past S/8 the one lower
threshold that fits every level starts picking tiles that are too short. Four
is the regret optimum and **three is what ships**, for three reasons: the
fourth height's whole gain lands on batch <= 8 (0.04 - 0.09 us on a 1.3 - 1.5
us kernel, in the widest part of the null band), it is slightly WORSE where the
third height pays most (N=256 batch 24-32: 1.157x / 1.139x against 1.182x /
1.188x), and it costs another ~17 KB of instantiation for +0.6% geomean. If
batch 1-8 at N=512 or N=1024 ever becomes the shape that matters, the fourth
tier is one `else if` and `PTO_TS_TILES_MID` already has the right value for
it.

The second threshold is **not** the same constant as the first (it is ~5-7 tiles
of the mid tile at every N, against 16 of the tall one), so two constants are
the honest minimum: `PTO_TS_TILES_TALL = 16` and `PTO_TS_TILES_MID = 6`. No
per-N number anywhere -- the case table is unchanged and the tiers are `S / 2`
and `S / 4` of it, with `if constexpr` collapsing the pick where they repeat
(N=4096's third tier, and all of N >= 8192, cost nothing). Cost of the nine
extra rings: `.so` **62760 -> 109416 bytes**; two tiers would have been 91912.

#### The schedule is tiles per core, and it was never a square root (up to 1.49x)

Refitting `_TILE_RUN_K` per tier exposed the form as wrong rather than the
constants as stale. `bd = sqrt(k * tiles)` means **tiles per core grows with
the batch**; the measured optimum holds it at 2 and lets the 24-core cap raise
it. At N=4096 S=1 the best block_dim over tiles 4, 6, 7, 8, 12, 16, 24, 31 is
2, 3, 4, 4, 6, 8, 12, 16 -- `ceil(tiles / 2)`, exactly, at every point.

Swept at **every** integer block_dim in [1, 24] over 139 shapes (91 across the
three tiers of N=256..4096, 48 at N=8192..32768), scored on the clamped count
the launch really uses and against the best **reachable** split -- a block_dim
of any other shape is one `ceil(tiles / ceil(tiles / bd))` can never produce,
so scoring against it invents regret no `k` could remove:

| law | regret worst | regret geomean |
|---|---:|---:|
| `sqrt(k * tiles)`, best k per tier | 1.4879 | 1.0251 |
| one tile per core always | 1.6895 | 1.1197 |
| two tiles per core always | 1.0680 | 1.0015 |
| **one each up to `_SOLO_TILES[n]`, two each after** | **1.0139** | **1.0005** |

so `_TILE_RUN_K` is gone and the only free value left is where one-per-core
turns into two-per-core -- a tile count, per N:

| N | `_SOLO_TILES` | why |
|---|---:|---|
| 256..4096 | 1 | they carry three tile HEIGHTS instead, so a small batch arrives as many short tiles, never as a few tall ones, and pairing them always wins |
| 8192 | 6 | one height, and it is the full 8192 elements: up to six of them, a core wants its own |
| 16384, 32768 | 24 | dense-shallow and blocked-shallow, no deep ring, so pairing a core's tiles buys no overlap and they want every core they can get |

**The two shallow sizes were the largest single loss the sweep found**: their
old `k` asks for `sqrt(8 * tiles)` and `sqrt(16.54 * tiles)` cores where they
want all of them, which costs **1.21x at N=16384 batch 12-16 and 1.49x at
N=32768 batch 24**. Neither has a tile tier, so that is pure schedule -- and it
sat under a fit that had been scored only over batches 64 and up.

#### What the two changes are worth together (1.0343x over 135 shapes)

Old kernel and old schedule against new, each arm at the block_dim its **own**
host picks -- the only honest comparison, since the tile height sets the tile
count, which sets what the schedule asks for. 135 shapes (every N x batch
1..65536), **arm order rotated per pass** so no arm keeps the same position,
6 passes, medians.

| N | geomean | best | worst |
|---:|--------:|-----:|------:|
| 256 | 1.0436x | 1.145x | 0.985x |
| 512 | 1.0447x | 1.176x | 0.992x |
| 1024 | 1.0355x | 1.176x | 0.986x |
| 2048 | 1.0342x | 1.148x | 0.986x |
| 4096 | 1.0271x | 1.104x | 0.996x |
| 8192 | 1.0042x | 1.053x | 0.994x |
| 16384 | 1.0508x | 1.220x | 0.997x |
| 32768 | 1.0314x | 1.493x | 0.996x |
| **all** | **1.0343x** | **1.493x** | **0.985x** |

By batch band: 1-8 **1.0232x**, 12-64 **1.0697x**, 96-512 **1.0190x**,
>= 1024 **1.0002x** -- nothing is given up at bulk, which is what the tall tier
is for. Null arm, same session, same shape set, two byte-identical `.so` under
different names: geomean **0.9999x**, band **0.9875x - 1.0113x**. Every
negative cell in the table above is inside that band.

**N=512 is the exception, and it is a host-side one**: the wrapper sends batch
>= 16 at N=512 through the bit-plan tile (`T = 8192`, i.e. the N=8192 case),
so the 1.0447x measured here is the classic path, which production reaches only
below batch 16 and for the <= 15-sample remainder of a bit-plan launch. The
tiers are still carried there, because the bit-plan is a host decision and the
kernel must be right without it.

**The arm order mattered more than the change at two shapes.** In the first,
unrotated run (base, null, tier every pass) N=1024 batch 65536 and N=4096 batch
16384 -- the two 64M-element shapes -- read **1.175x** for the tier arm, which
would have been the best numbers in the table. The null read 1.029x and 1.033x
at exactly those two shapes, its largest deviations anywhere, and the arms'
block_dim and tile height are identical there. Rotating the order moved them to
0.992x and 0.990x with a per-pass spread of 35 us. A fixed arm order plus a
shape big enough to move the clock is a 1.17x result waiting to be believed.

#### Can `S` go UP? No -- every row is exactly on the L0C ceiling

The tall tile is capped, and the comment that says `S * N <= 16384` is the
SHALLOW bound. The deep path keeps two tiles in flight, each with a stacked and
a concatenated fp32 accumulator, so `Layout::kDbuf` requires
`4 * kAccBytes <= L0C_BYTES`, i.e.

    16 * S * N <= 131072   ->   S * N <= 8192

and **every row of the case table sits exactly at `S * N = 8192`** (32x256,
16x512, 8x1024, 4x2048, 2x4096, 1x8192). L0C is 100% spent; L1 uses at most
139264 of 524288 and L0A/L0B have >= 32 KB free except at N=8192, where L0B is
also exactly full. Doubling any `S` needs 262144 bytes of L0C.

Dropping to the shallow path to get the room back is not an option and not a
guess: `runTiles` has `static_assert(S == 1)` on that path (no masked last
tile), and the configuration a 16384-element tile would land in is the one
N=16384 already runs -- which the bit-plan `(N, T)` grid above measured
directly, at N=512: `T = 8192` (deep) scores 2.19x / 2.42x where `T = 16384`
(shallow, one tile twice as tall) scores 1.76x / 1.74x. **The deep ring is
worth more than the taller tile, by 1.24x - 1.39x.**

One route is identified and unmeasured: aliasing the stacked and concatenated
accumulators the way the blocked path does (`kAlias`) halves the requirement to
`8 * S * N <= 131072`, i.e. `S * N <= 16384`, because phase A's result leaves
L0C for the L1 turnaround before phase B's matmul writes. It would serialise
FIX against M inside a tile and lean on the ring's other in-flight tile to
cover it, which is new sync on the path that already produced the one-id
two-channel defect -- so it is a project, not a tweak. Note that blocking's
measured 1.5x - 1.8x loss does **not** condemn it: that number is aliasing plus
an M-sized matmul, and only the second half is the expensive part.

#### How far the operand fix extends: nothing left below N=2048, 1.14x still there at N=8192/16384

Asked whether the N=256/512 operand win extends upward. The question has an
exact answer, because the operand load can simply be **deleted**: strip the four
`TLOAD`s from `loadHadamardFactors` (keeping the `TMOV`s, the `Sync`s and all
compute, so only the GM leg goes) and the ratio is a hard ceiling on everything
operand-transfer-related. A wrong-answer probe -- device time here is
data-independent -- built from one path with a null arm, 4 rotated passes, 60
reps:

|       N | b1 | b8 | b32 | b128 | b1024 | b16384 | worst-case us |
|---|---:|---:|---:|---:|---:|---:|---:|
|   256 | 1.0156 | 1.0000 | 1.0147 | 1.0000 | 1.0042 | 0.9966 | +0.02 |
|   512 | 1.0308 | 1.0036 | 1.0128 | 1.0000 | 1.0062 | 1.0008 | +0.04 |
|  1024 | 1.0153 | 1.0137 | 1.0114 | 1.0000 | 1.0045 | 1.0000 | +0.02 |
|  2048 | 1.0466 | 1.0800 | 1.0265 | 1.0178 | 1.0078 | 1.0023 | +0.13 |
|  4096 | 1.0700 | 1.0655 | 1.0270 | 1.0219 | 1.0066 | 0.9981 | +0.11 |
|  8192 | **1.1350** | 1.1143 | 1.0920 | 1.0565 | 1.0162 | 1.0004 | +0.25 |
| 16384 | **1.1411** | 1.1277 | 1.0952 | 1.0488 | 1.0103 | 0.9987 | +0.45 |
| 32768 | 1.0140 | 1.0097 | 1.0100 | 1.0047 | 1.0017 | 1.0002 | +0.50 |

null band 0.9898 - 1.0078, geomean 1.0002.

Two readings, and they point opposite ways.

**At N=256..1024 the fix is complete.** Deleting the operands outright is now
worth 1.00x - 1.03x, i.e. 0.02 us. There is no residue to chase: the `GmNz`
change captured essentially the whole prologue. That also confirms the
diagnosis -- what was expensive was never the bytes (512 of them) but the
32-byte descriptor rows, and severity scales as 1/row-length, which is why the
same change read 1.00x - 1.05x from N=1024 up where the rows are already
64 - 512 bytes.

**At N=8192 and N=16384 a different defect is still there**, and it was never
measured before because the narrow-row story did not predict it. 1.14x at
batch 1, still 1.05x at batch 128, 0.25 - 0.45 us against a 2 - 3 us kernel. It
is not a layout problem -- those operands are already one contiguous NZ burst
with 256-byte rows. It is **volume**: 40 KB per core at N=8192 (64x64 plus
128x128) and 64 KB at N=16384 (two 128x128), against 1 KB at N=256, sitting on
MTE2 in front of the first data tile. 64 KB at a plausible single-core MTE2
rate is ~0.5 us, which is the order measured, so the arithmetic corroborates.
It decays with batch exactly as a fixed per-core cost should.

Layout cannot shrink it, and the fix is not a layout change at all.

#### Taken: when R == C the kernel was reading the same matrix from GM twice (1.0493x at N=16384)

`R == C` makes `H_R` and `H_C` **the same matrix** -- both are the Sylvester
`H` of that size -- and the case table has four such sizes: N=256, 1024, 4096
and **16384**, the worst cell above. The kernel was loading that matrix from GM
once for L0A and again for L0B. H is symmetric, so the left operand needs no
transpose, and both `TMOV`s read the same NZ bytes: **one L1 copy can feed
both**. `Layout::kShareHf = (R == C) && !kBlkA && !kBlkB` skips the second
`TLOAD` and points phase B's L0B move at `kL1Hl`.

`kL1Hc` is left reserved so `kL1X` and the ring are untouched, which keeps the
change confined to the prologue -- and makes the four `R != C` sizes an
**in-binary control**, since `kShareHf` is false there and their device code is
unchanged. Measured (pre / share, 5 rotated passes x 60 reps):

|       N | b1 | b2 | b4 | b8 | b16 | b32 | b64 | b128 | b512 | b4096 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|   256 | 1.0000 | 1.0000 | 1.0156 | 1.0152 | 1.0149 | 1.0147 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
|  1024 | 1.0071 | 1.0143 | 1.0141 | 1.0137 | 1.0000 | 0.9941 | 1.0000 | 1.0000 | 1.0000 | 0.9915 |
|  4096 | 1.0000 | 1.0247 | 1.0241 | 1.0224 | 1.0204 | 1.0088 | 1.0078 | 1.0061 | 1.0136 | 1.0014 |
| **16384** | **1.0441** | 1.0435 | **1.0493** | 1.0392 | 1.0355 | 1.0273 | 1.0250 | 1.0189 | 1.0051 | 1.0011 |

`R == C` geomean 1.0127 over 40 cells, worst 0.9915; **control (R != C)
geomean 1.0001**, band 0.9863 - 1.0154; null arm 0.9846 - 1.0086. 864/864 fp64
cases pass, worst rel_rms 3.264e-04 (at N=512, which does not use the path --
the pre-existing fp16 floor).

The absolute column is what identifies the mechanism: at N=16384 it is
**+0.12 us at every batch from 1 to 4096**, dead flat. That is one 32 KB GM
read per core, removed -- exactly a fixed per-core cost, not a scaling effect.

The kernel now *depends* on the two operand buffers holding identical bytes,
and nothing in `_factor_buffer` enforces it: give either side its own
convention and the kernel silently multiplies by the wrong matrix on the side
it no longer reads. `test_shared_factor_sizes_have_identical_operand_buffers`
parses the case table for `R == C` rows and asserts `torch.equal` on the pair.

#### Still open: N=8192, and why blocking is the only remaining lever

N=8192 is untouched -- R=64, C=128, so there is nothing to share -- and keeps
its 1.135x ceiling. The only thing that can shrink those bytes is the
compile-time operand freedom itself: `H_128 = H_2 (x) H_64`, so the blocked
form applies `+-H_64` twice on half the operand bytes. That is the
`kBlkA`/`kBlkB` path the kernel already runs at N=32768 -- which reads 1.014x
in the table above, the lowest, consistent with its smaller blocked operands.
Blocking's measured cost here is 1.5x - 1.8x against a <=1.135x prize, so
expect it to lose; but see the caveat above, that number is aliasing plus an
M-sized matmul and only the second half is expensive. H_64 is also literally
the top-left block of H_128, but in NZ that sub-block is 4 strided 2 KB chunks
rather than one burst, and it is only 8 KB of the 40 KB -- scaling from the
N=16384 result, worth ~0.03 us, i.e. under the null band. Not worth the risk.

The honest ordering, though: after the fix above, N=16384 has ~1.09x of this
class left at batch 1 and ~1.00x in bulk, while the N=32768 column-block
serialization below is worth ~1.78x wherever it applies. That is where the
large-N work is.

#### The split is a free variable, and C == 16 buys conversion-free GM legs (1.1793x)

`hadamard_split` returned the **balanced** split, and the kernel's case table
mirrored it. Balanced minimises MACs -- the two-sided cost is `N * (R + C)` per
sample, minimal at `R == C == sqrt(N)` -- but it is not the only objective.
`Layout::kNzGm` is `(C == 16) || (S == 1 && !kBlockIt)`, and the `C == 16` arm is
there because **a 16-column tile is a single NZ block-column, so its NZ image IS
the row-major bytes**: the ND2NZ permutation is the identity and BOTH GM legs go
conversion-free at any S. The balanced split throws that away for every size
except N=256.

The two objectives can be priced against each other:

|    N | balanced | C == 16 form | R + C | MAC cost |
|---|---|---|---:|---:|
|  512 | 16 x 32 | **32 x 16** | 48 -> 48 | **neutral** |
| 1024 | 32 x 32 | **64 x 16** | 64 -> 80 | +25% |
| 2048 | 32 x 64 | 128 x 16 | 96 -> 144 | +50% |
| 4096 | 64 x 64 | 256 x 16 | 128 -> 272 | +113%, and R=256 forces blocking |

N=512 is **free**: `16 + 32 == 32 + 16`. And the +25% at N=1024 is nearly free
for a different reason -- the audit put the cube at 1-7% of device time below
N=16384, so MACs are the one resource there is slack in, while the transfers
are not. Measured (balanced / C==16, 5 rotated passes x 60 reps, null band
0.9902 - 1.0143):

|    N | b1 | b8 | b32 | b128 | b512 | b2048 | b8192 | b32768 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|  512 | 1.0000 | 1.0286 | 1.0194 | 1.0211 | 1.0645 | 1.0646 | 1.1406 | **1.1551** |
| 1024 | 0.9772 | 1.0137 | 1.0595 | 1.0446 | 1.0031 | 1.0868 | 1.1400 | **1.1734** |

The win **grows with batch**, which is the signature of a per-byte transfer
saving rather than a fixed cost, and it is paid for at batch 1-2 at N=1024
(0.972x - 0.975x, +0.04 us) where there is no volume to amortise the extra MACs.

**N=2048 was argued down and then measured, and the measurement agreed -- but
only just.** +50% MACs was *not* obviously too much, since +25% had just proved
free, so both orientations were built and validated (108/108 each):

| batch | 32x64 balanced | 128x16 | 16x128 |
|---|---:|---:|---:|
|     1 | **1.46** | 1.56 | 1.64 |
|   128 | **2.56** | 2.78 | 2.78 |
|  2048 | **9.28** | 9.54 | 9.53 |
|  8192 | **29.26** | 29.31 | 29.67 |
| 32768 | 189.64 | **189.64** | 192.19 |

Balanced wins: 128x16 reads 0.9588 geomean (worst 0.9211) and 16x128 0.9370.
Note the trend is still there -- 128x16 climbs monotonically to break-even by
batch 32768 -- so the MAC penalty did not stop the transfer saving, it outran
it. The crossover is therefore **sharp, between N=1024 (+25%, 1.05x) and
N=2048 (+50%, 0.96x)**, and it is a ratio of two growth rates, not a threshold
anyone could have named. N=4096 is worse still: `16x256` needs S=1, which
collapses the tile height, and measured **0.52 geomean** (worst 0.27);
`256x16` will not compile, since R=256 forces blocking.

Both orientations cost the same MACs -- `R + C` is symmetric under the swap --
so at equal MACs the `C == 16` orientation is the only one that also gets the
layout, which is why `128x16` beats `16x128` everywhere.

**A latent fault this uncovered.** `X(1024, 64, 16, 8)` compiled and **faulted
the aicore** with an MTE range error. The `C == 16` tile bodies reinterpret the
loaded `(S*R x 16)` tile as `(R x S*R)` -- "read wide" -- and that is
byte-compatible only when `S*R*16 == S*R*R`, i.e. **R == 16**. They were
written for N=256, the only `C == 16` row at the time, and no assert captured
the assumption. The bodies now gate on `R == C == 16` and everything else takes
the general body, which handles any `(R, C)`. **That predicate change is
byte-identical on the shipped table** (N=256 is the only row with `C == 16` and
it has `R == 16`), so it is a pure safety fix -- proved by `cmp` on the `.so`
from one build path, not by a timing.

#### The bit-plan floor had to move: its baseline got 1.84x faster underneath it

`_BITPLAN_TILE = {512: 8192}` routed **every** batch >= 16 at N=512 through the
bit-plan path. That was fitted when the classic path still did the 32-byte-per-
row operand gather. Since the `GmNz` fix (1.84x at N=512) and now the `32 x 16`
split, the classic path is much faster and the comparison inverts. Three ways
to do N=512, device us per launch:

| batch | classic (16x32) | bit-plan (shipped) | split (32x16) | split / bit-plan |
|---|---:|---:|---:|---:|
|    16 | 1.48 | 2.04 | 1.46 | **1.3973** |
|    32 | 1.58 | 2.08 | 1.56 | 1.3333 |
|   128 | 1.96 | 2.34 | 1.90 | 1.2316 |
|   512 | 2.66 | 2.86 | 2.50 | 1.1440 |
|  2048 | 4.43 | 4.48 | 4.18 | 1.0719 |
|  8192 | 10.74 | 9.42 | 9.60 | 0.9811 |
| 32768 | 36.10 | 28.92 | 31.27 | 0.9249 |

Bit-plan was **1.23x - 1.40x slower than the path it replaced** from batch 16 to
2048, and only pays from ~8192 up. `_BITPLAN_MIN_BATCH = {512: 8192}` gated it
there. It kept a large-batch win (1.08x over the split at 32768), so at this
point it was gated rather than deleted. **That win has since gone too** -- the
split caught it at every batch and the path is now removed; see *Re-routing
one-sided vs two-sided*, which is the second instance of the caution below.

Two cautions this leaves on record. First, **an optimisation's verdict expires
when its baseline changes**; bit-plan's 2.19x - 2.42x was true against the
kernel of the time. Anything fitted as "X beats Y" needs re-running after Y
improves. Second, raising the floor moved the bit-plan path **out of test
coverage** -- batches 22 and 65 used to exercise it and then took the classic
path -- so `test_bitplan_floor_boundaries` straddled the floor and crossed a
whole-tile boundary to keep both the bulk launch and the bulk-plus-remainder
split exercised.

#### The free reinterpret is worth 1.28x -- NOT the order, which is what it was first credited to

With the `(N/16, 16)` split, which of `H_16` and `H_{N/16}` goes first? It
depends on the body, and **the two bodies disagree**:

| body | phase A | phase B | applied first |
|---|---|---|---|
| general | `X . H_C` | `H_R . T` | **H_C** (the 16-wide one) |
| `C == 16` ("cat") | `H_R . X` | `T . H_C` | **H_R** |

So the order is a free variable too, and N=256 isolates it exactly: at
`R == C == 16` both bodies are valid with the same split, the same MACs and the
same layout, so forcing N=256 through the general body changes **nothing but
the order**. Measured:

| batch | cat (H_R first) | general (H_C first) | ratio |
|---|---:|---:|---:|
|     1 | 1.28 | 1.28 | 1.0000 |
|   128 | 1.63 | 1.72 | 0.9477 |
|  2048 | 3.02 | 3.34 | 0.9039 |
|  8192 | 5.48 | 6.48 | 0.8457 |
| 32768 | **15.10** | 19.16 | **0.7881** |

**H_R-first measured 1.083x geomean and 1.269x at batch 32768**, and the gap
grew with batch. The reason is FIX descriptors: the general body turns the
vertical stack of S samples into a horizontal concatenation with **S separate
`TEXTRACT`s** -- S FIX ops per tile against the cat body's one `TMOV` -- on the
pipe that is already the bottleneck.

> **This experiment was mis-attributed, and the follow-up below corrects it.**
> Forcing N=256 through the general body changes the order AND the descriptor
> count, because the general body paid its S `TEXTRACT`s *unconditionally* --
> it never took the `C == 16` reinterpret even where it was available. So what
> was isolated was the BODY, not the order, and the mechanism named (FIX
> descriptors) is in fact the whole of it. Give the single body the reinterpret
> and the order is free: measured 1.0009 geomean over 54 cells. The 1.28x is
> real and still the reason the `C == 16` sizes are fast -- it is just a
> descriptor-count win, not an ordering win. **A probe that swaps one
> `if constexpr` has not isolated one variable if that branch differs in two
> ways.**

**The better order was unreachable because of one symbol.** The cat body set
`kCat = S * R` and viewed the loaded tile as `MatTile<R, kCat>`. The tile holds
`S*R*C` elements, so that view only matches when `R == C` -- true of N=256, the
only `C == 16` row at the time. The intended width is `S * C`:

> With `C == 16` each sample's `(R x 16)` block is ONE NZ block-column of
> `R/16` fractals, so the vertical stack of S samples **is** their horizontal
> concatenation, for **any** R. The read-wide reinterpret was never specific to
> `R == 16`; only the arithmetic expressing it was.

`kCat = S * C` and the gate opened from `R == C == 16` back to `C != 16`.
**The `S*C` rewrite is byte-identical with the gate still closed** -- N=256 is
its only user and `S*C == S*R` there -- so the rewrite is provably neutral and
only the gate is a behaviour change. Measured, same split, H_R-first vs
H_C-first:

|    N | b8 | b128 | b2048 | b8192 | b32768 | geomean |
|---|---:|---:|---:|---:|---:|---:|
|  256 | 1.0148 | 1.0000 | 1.0035 | 1.0000 | 1.0013 | 1.0022 |
|  512 | 1.0362 | 1.0335 | 1.0505 | 1.0853 | **1.1060** | 1.0425 |
| 1024 | 0.9733 | 1.0091 | 1.0460 | 1.0541 | 1.0637 | 1.0130 |

N=256 holding at 1.0022 is the consistency check the byte-identity predicts.
N=1024 pays 0.958x - 0.977x at batch 2-32, the same small-batch corner the
split itself pays, and the two do not compound: the final sweep reads 0.972x
there, not 0.93x.

**Net, commit e5e0386 -> here, 70 cells, 4 rotated passes x 50 reps:**

|       N | geomean | best | worst |
|---|---:|---:|---:|
|     256 | 1.0047 | 1.0159 | 0.9963 |
|     512 | **1.0959** | **1.2830** | 0.9923 |
|    1024 | **1.0722** | 1.2450 | 0.9722 |
|    2048 | 1.0005 | 1.0066 | 0.9898 |
|    4096 | 1.0113 | 1.0277 | 1.0001 |
|    8192 | 1.0026 | 1.0071 | 0.9991 |
|   16384 | 1.0211 | 1.0475 | 0.9998 |
|   32768 | 0.9986 | 1.0022 | 0.9955 |
| **ALL** | **1.0257** | **1.2830** | 0.9722 |

null geomean 1.0007, band 0.9926 - 1.0183; 12 cells >= 1.05x, 2 below 0.98x
(N=1024 batch 1-2). 864/864 fp64 cases pass; 271 tests pass.

**The lesson worth keeping:** three of this kernel's free variables -- the
split, the body, and the order of application -- were all pinned by a single
historical fact, that N=256 was once the only `C == 16` size. A constant fitted
for one case (`_TILE_RUN_K`), an operand layout (`GmTile`), a specialised body
(`kCat = S * R`) and a path threshold (`_BITPLAN_TILE`) each encoded that
assumption, and each read as deliberate.

##### Reference: per size, which factor is which and which goes first

`Y = H_R . X . H_C` with the sample viewed as `R x C` row-major, so **`H_R` is
the LEFT multiplication** (acts on the row index, the HIGH bits of the element
index) and **`H_C` is the RIGHT multiplication** (the column index, the LOW
bits). The order rule is one predicate, and it holds identically in all three
tile bodies (`tilePhaseA`/`tilePhaseB` deep, `hadamardTile` shallow dense,
`hadamardTile` + `shallowBlockA` blocked):

> `C == 16` -> **left multiplication first** (`H_R . X`, then `T . H_C`).
> `C != 16` -> **right multiplication first** (`X . H_C`, then `H_R . T`).

|       N | R = left | C = right | S | smaller | body | applied FIRST | phase-A FIX descriptors |
|---|---:|---:|---:|---|---|---|---:|
|     256 |  16 |  16 | 32 | equal | deep cat | **left** `H_16` | 1 |
|     512 |  32 |  16 | 16 | **right** (16) | deep cat | **left** `H_32` | 1 |
|    1024 |  64 |  16 |  8 | **right** (16) | deep cat | **left** `H_64` | 1 |
|    2048 |  32 |  64 |  4 | **left** (32) | deep general | **right** `H_64` | 4 |
|    4096 |  64 |  64 |  2 | equal | deep general | **right** `H_64` | 2 |
|    8192 |  64 | 128 |  1 | **left** (64) | deep general | **right** `H_128` | 1 |
|   16384 | 128 | 128 |  1 | equal | shallow general | **right** `H_128` | 1 |
|   32768 | 128 | 256 |  1 | **left** (128) | shallow blocked | **right** `+-H_64` | 1 |

(`kBlockIt` is true only at N=32768, where both sides block: `kMa = kMb = 64`,
so `kBlkA` and `kBlkB` are both set. `kDbuf` is true for N=256..8192 and false
from N=16384 up.)

**The larger factor goes first in every unequal case** -- but that is
descriptive, not the mechanism. At N=256 the two are equal and the order was
still worth 1.269x, so size is not what pays. What pays is the **intermediate's
shape**: the cat body writes `T` as `(R x S*C)` and hands it to phase B as one
`TMOV`, while the general body writes `(S*R x C)` and must turn the vertical
stack into a horizontal concatenation with **S separate `TEXTRACT`s**. The
penalty is therefore proportional to S, which is exactly why it is worth
chasing at small N and worth nothing at large N: the right-hand column collapses
to 1 as soon as `S == 1`.

So there is **no order win left at N=8192, 16384 or 32768** -- they run one
sample per tile, so both orders cost one descriptor. And the order cannot be
flipped at N=2048 or 4096 either: the cat body's read-wide view needs each
sample's `(R x C)` block to be ONE NZ block-column, which is true only at
`C == 16`, and buying `C == 16` at those sizes costs more MACs than it saves
(measured above). The two sizes where S is large and `C == 16` is affordable --
512 and 1024 -- are the two that moved.

#### Why the L0C turnaround cannot absorb a layout change (it already absorbs the free one)

Asked whether the permutation could be folded into the `L0C -> L1` turnaround,
which moves anyway. It already is, where it is free: `tilePhaseA` writes the
accumulator as `(R x S*R)` and re-reads the same bytes as `(kStackRows x C)`,
and the general body's `TEXTRACT` loop turns a vertical stack into a horizontal
concatenation by choosing each block's L1 offset. Both are **reinterprets** --
the NZ images coincide, so no data moves.

What is not available there is a permutation that is *not* the identity on the
bytes. The fixpipe does ND and NZ, not an arbitrary bit permutation, so a
non-trivial reorder costs one `TMOV` per fractal group, i.e. **FIX
descriptors** -- and FIX is the measured bottleneck at N=512..8192 (store
13-39% plus turnaround 7-30%) while MTE2 is never above 8%. Paying the
bottleneck to relieve the idlest pipe is the wrong direction. The `C == 16`
split above is the better form of the same idea: it makes the permutation the
identity **by construction**, instead of hiring a pipe to perform it.

#### Unifying the order: the source unifies, the order cannot, and the generated size is elsewhere

Asked whether "the larger factor goes first" could be made the actual rule, so
that one body serves every size and both text and generated code shrink. Three
separate answers, because the three parts of that question come apart.

**The pattern is real but it is a consequence, not a rule.** The order is
selected by one predicate, `C == 16`, and nothing in the kernel mentions size.
It comes out as "larger first" only because the `C == 16` sizes are exactly the
ones where C is the *small* factor (16 < R) and the rest keep `R <= C` from the
balanced split. At N=256 the two are equal and the order still mattered 1.269x,
so size is not the mechanism and would be the wrong thing to key on.

**The order CAN be unified, and the first attempt at this answer was wrong.**
"Delete the `C == 16` body" was rejected at 1.083x geomean / 1.269x -- but that
measured a body which paid its S `TEXTRACT`s unconditionally, including at
`C == 16` where the reinterpret was sitting there unused. The condition is
`C == 16` and it is **symmetric in direction**: `(S*R x C)` and `(R x S*C)` are
the same fractals in the same order, so at `C == 16` BOTH hand-offs are free,
whichever way round they run. Give the single body that reinterpret on the
turnaround and on the store and the second body is redundant:

| | turnaround | store | total FIX descriptors |
|---|---|---|---:|
| `C == 16` | 1 `TMOV` | 1 whole-tile `TSTORE` | **2** |
| `C != 16` | S `TEXTRACT`s | `rows` `TSTORE`s | 2S |

Measured, two bodies -> one, 54 cells, 5 rotated passes x 50 reps:

|       N | geomean | worst |
|---|---:|---:|
|     256 | 1.0020 | 0.9838 |
|     512 | 0.9996 | 0.9802 |
|    1024 | 1.0045 | 0.9750 |
|    2048 | 1.0009 | 0.9915 |
|    4096 | 0.9987 | 0.9875 |
|    8192 | 0.9999 | 0.9961 |
| **ALL** | **1.0009** | 0.9750 |

null geomean 0.9990, band 0.9844 - 1.0123. Every per-N geomean is inside the
null band, and the 1.28x was always the descriptor count, never the ordering.
864/864 fp64 cases pass.

**But "free" is a statement about the geomean, and a tighter sweep shows it
averages over a SIGN FLIP.** 6 rotated passes x 80 reps at the two sizes with
unequal factors, with the tile height the dispatch actually picks:

| N | S | batches | larger-first / smaller-first | winner |
|---|---:|---|---|---|
| 1024 | 2, 4 | 2 - 64 | 1.0239 - **1.0417** | **larger-first** |
| 1024 | 8 | 128 - 32768 | 0.9704 - 0.9954 | **smaller-first**, to 1.030x |
| 512 | 4, 8 | 1 - 128 | 1.0000 - 1.0072 | tie |
| 512 | 16 | 256 - 32768 | 0.9682 - 1.0010 | **smaller-first**, to 1.033x |

12 of 14 cells at N=1024 fall outside the null band (0.9948 - 1.0079), so this
is structure, not scatter. The sign is set by the **tile height**, not the
batch: the crossover at N=1024 sits exactly at batch 128 = `TILES_TALL * S`,
the tier boundary. **The tall tile always prefers smaller-first**, at both
sizes; the short tiles prefer larger-first only where the MAC split is lopsided
(`R/C` = 4 at N=1024, where the two matmuls are 524k and 131k MACs; `R/C` = 2 at
N=512 reads as a tie).

So order is worth up to ~4%, in whichever direction the tile height says, and
the net over a batch sweep is ~0 because the regions cancel. Keeping the single
body is still right: it carries **smaller-first, which is the correct order for
the tall tile**, and the tall tile is where the absolute time is (54 us at batch
32768 against 1.4 us at batch 8). The ~4% left at N=1024's short tiles is
0.06 us, under the 46 us host launch floor, and collecting it means carrying
both orders -- more generated code, which is the opposite of the point here.

So the kernel now runs **one** tile body, right-multiply-first everywhere, and
"larger factor first" is abandoned with no cost, which is the cleanest proof
that it was never the mechanism. Swapping `(R, C)` is not needed for this and
would not have helped on its own: the orientation only moves the layout
(`C == 16`) and the MAC count, both already measured, and at equal MACs the
`C == 16` orientation already wins.

What does NOT unify is the pair of hand-offs at `C > 16`, and that is
structural: the S samples sit side by side in L0C, and the fixpipe L0C->GM path
is 2D-to-2D only, so one store cannot express them. The deep tile path is down
to exactly **two** size-dependent branches, both the same predicate, both
saying the same thing -- is the reinterpret free.

**The source does unify, though, and byte-identity proves it costs nothing.**
There are only two distinct matmul shapes in the whole kernel, and each phase
runs one of each -- in opposite order. For a given shape both phases use the
same L0A and L0B addresses; only the L1 source and the L0C accumulator move:

|  | phase A | phase B |
|---|---|---|
| `C == 16` | shape L, then **1** `TMOV` | shape S, then **1** `TSTORE` |
| `C != 16` | shape S, then **S** `TEXTRACT`s | shape L, then **`rows`** `TSTORE`s |

That table is the whole cost story -- **2S descriptors against 2** -- and it is
why this is worth chasing where S is large and worth nothing at `S == 1`. The
first version of this refactor kept both orders and shared them through
`stageTileData<OperandLeft>` / `multiplyTile<OperandLeft>` helpers; once the
single body was shown to be free, each helper had one caller and was inlined
back. Net: one staging path and one matmul per phase, with the two `C == 16`
branches above.

Two deletions, both **byte-identical** and so proved free without a device:

- `hadamardTile` carried its own `C == 16` branch -- **56 lines of dead code**.
  The shallow path asserts `S == 1` while every `C == 16` row has `kDbuf` and
  takes the deep ring, so it was never instantiated. It is also where the
  `kCat = S * R` bug lived in duplicate. Replaced by
  `static_assert(C != 16, ...)`, which turns adding such a size into a build
  error instead of a silently slower path.
- the two phases, factored as above.

`fast_hadamard_twosided.cpp` **1248 -> 1158 lines**, nine `TMATMUL` sites down
from eleven, and the deep tile path went from two bodies carrying two matmul
shapes each to one body carrying one. 864/864 fp64 cases pass, 271 tests pass.

**The generated size is not in the duplication -- it is in the tile tiers.**
Measured by rebuilding with shallower dispatch:

| dispatch | `.so` bytes | vs shipped | regret geomean | regret worst |
|---|---:|---:|---:|---:|
| **3 tiers (shipped)** | **100552** | -- | **1.0092** | 1.0846 |
| 2 tiers (S, S/2) | 84200 | **-16%** | 1.0155 | 1.0859 |
| 1 tier (S only) | 57704 | **-42%** | 1.0553 | 1.1918 |

So `if constexpr` already costs nothing for the untaken branches, and the only
large lever on generated size is instantiating fewer tile heights. The 3 -> 2
step is the interesting one: **16% of the binary for 0.6% of mean performance**,
with worst-case regret essentially unchanged (1.0859 against 1.0846). The
3 -> 1 step is a bad trade at 42% for 4.6% mean and 19% worst case. Neither is
taken here -- the tiers were measured in and this is a size decision, not a
performance one -- but the price is now on record.

#### Found by audit, measured, not yet taken

A function-by-function pass over the kernel with per-pipe "delete one resource,
keep every flag" probes attributes device time as: **FIX is the bottleneck for
N=512 .. 8192** (store 13% - 39% plus L0C->L1 turnaround 7% - 30%, and both
halves are structural -- a two-matmul tile pushes 32 KB through FIX per 16 KB of
output, twice the one-sided kernel's fixpipe traffic); **MTE2 is never the
bottleneck** (<= 8%); the cube is 1% - 7% below N=16384 and then 30% / 56%; and
a fixed floor of 0.94 - 2.4 us is 20% - 53% of device time at batch <= 256.

Three items came out of it that are recorded rather than applied:

- **N=32768 is fully serialized.** Its exposed shares sum to 100%: deleting the
  matmuls is 2.290x, and the independent MAC floor (12.6 MMAC/sample x 42.67
  samples/core / 4096 MAC/cycle at 1.8 GHz = 73 us) matches the 74.7 us the
  matmuls take -- so the cube runs at ~100% of peak while it runs and the other
  58 us overlaps nothing, because `kBuf == 1` there (one tile's accumulator is
  all 128 KB of L0C). Perfect overlap would be **1.78x**. The route is to
  pipeline over output COLUMN blocks: `accA(c)` and `accB(c)` are 32 KB each, so
  two column blocks in flight is exactly 128 KB, at the price of 32 matmuls per
  sample instead of 20 and a second `TMOV` of X. Forecast 1.3x - 1.6x on the
  slowest cell in the grid. Settle it first with a wrong-answer probe that
  halves both accumulators and runs the halves as a 2-stage pipeline -- under
  ~100 us means build it, ~125 us means the extra matmuls ate it. Fewer MACs is
  NOT the lever: a 3-way split is unavailable because a matmul's field must
  contain the four low bits of its view's row or column index, and the best plan
  respecting those anchors costs 288 against 384 MACs per element.
- **The N=512 split launch pays two task-queue drains and two tensor slices.**
  Host clock, `METH-360`: shipped 104.4 us, one drain 74.0, one drain plus
  pointer arithmetic instead of slicing **44.5 us (2.35x)**. It affects only
  N=512 with `batch % 16 != 0`, and the pointer form is *strictly safer* against
  the launch race than the current code, which creates a tensor slice after the
  stream pointer has been read.
- **The stack -> cat turnaround folds at every `C`, not only at `C == 16`**,
  contradicting the comment at the top of the kernel. One whole-tile `TEXTRACT`
  yields the concatenated view with its column blocks permuted by
  `J = jc*S + s`; phase B left-multiplies and never mixes columns, so the
  permutation passes through and the result read back as the stack is
  byte-identical. 864/864 validate, geomean 1.0019 with best 1.0391 (real only
  at large `S`), `.so` 13% smaller. Worth taking as a simplification; it is not
  a speed lever, because **FIX cost is bytes, not descriptors** -- the same
  change measured at constant bytes is 1.0042 with 17 of 20 shapes inside the
  null band.

Measured and worth nothing: a deeper X ring (MTE2 is never binding), MAC
reduction at N <= 8192 (the cube is 1% - 7%), the prologue `Sync`s (1.0127, and
provably redundant), sharing one L1 copy of a factor between L0A and L0B, and
`GmNz` on the X/Y legs for `S > 1, C > 16`.

#### What does not port, and one method note

**`static_assert(kL0aSlots == 2)` has nothing to guard.** The one-sided kernel
needs it because a tall `S` silently drops its L0A slot count to 1 while the
ring's edges still alternate by parity -- two event ids guarding one buffer.
This kernel's equivalent, `kSlotsA`, is already read through
`if constexpr (L::kSlotsA == 2)`, and the `kSlotsA == 1` arm uses a single
`(EV_SRA0, EV_SWA0)` pair on a single slot. The parity-addressed buffers are
likewise gated: `l0aX(par)` and `l0cStack(par)` only ever see `par == true` when
`kBuf == 2`, which the L0A and L0C capacity asserts already enforce. The defect
is unreachable by construction, so an assert would document a different kernel.

The uncommitted one-sided **flag audit** is the newest change and it splits.
Grouping this kernel's ids by **source** pipe shows the same shape of hazard:
`Sync`'s `id = 0` default lands on `(MTE2 -> MTE1)` and `(MTE1 -> MTE2)` id 0,
which the deep ring spends as `EV_D0`. Making the id **mandatory** so every call
site names its channel compiles to a **byte-identical** `.so` -- a free
source-level guard, taken. Moving the two `loadHadamardFactors` sequencing
points off id 0 is **not** taken: it measures 0.9994x against a null band of
0.9925x - 1.0276x for +288 bytes, and unlike the one-sided case the collision is
not live -- `loadHadamardFactors` runs once per core strictly before any tile
load, and the lazy form primes nothing, so no `EV_D0` credit can be in flight
when it runs.

**A codegen diff only proves neutrality when both arms compile from one
identical path.** `.aicore_binary` encodes the source **file name**: the same
content built as `k.cpp` and as `k_copy.cpp` produces different section hashes,
which reads exactly like a codegen change. Build both arms over one path,
and keep a null arm -- same content, same path, twice -- to show the comparison
can return "identical" at all.

### Re-routing one-sided vs two-sided, and retiring the bit-plan tile

Both kernels had moved under the dispatcher since its crossover was fitted, so
the routing was re-measured from scratch: every batch from 1 to 65536 at the two
sizes they share, each kernel at its own schedule, four interleaved passes with
the arm order rotated per pass. The null arm -- the same config run twice --
read 1.0000x and 1.0032x, so a 3% margin is signal.

| batch | N=256 one/two | N=512 one/two |
|---|---|---|
| 1 | 1.10 | 1.25 |
| 16 | 1.17 | 1.46 |
| 64 | 1.22 | 1.35 |
| 256 | 1.16 | 1.25 |
| 1024 | 1.13 | 1.30 |
| 4096 | 1.04 | 1.55 |
| 8192 | 1.03 | 1.68 |
| 16384 | 1.00 | 1.77 |
| 32768 | 0.97 | 1.82 |
| 65536 | 0.96 | 1.82 |

**The two columns run in opposite directions, and that is the whole result.**
At N=512 the one-sided kernel is compute-bound -- 262144 MACs per sample against
the split's 24576 -- so the more work a launch has, the further behind it falls.
At N=256 its MACs drop 4x while its bytes only halve, both kernels reach the
memory roofline (2485 against 2380 GB/s at batch 65536), and what is left to
separate them is the store pattern, where the one-sided kernel is marginally
ahead. So the crossover that used to be at N=512 is gone, and a new one appears
at N=256 pointing the other way.

The dispatcher now reads

```python
ONESIDED_BATCH_MIN = {256: 32768}
```

with batch 16384 left to the split: at 0.9983x it is inside the null band, and
a tie is not a reason to switch kernels.

**The bit-plan tile is retired.** It bought conversion-free GM transfers by
paying `A + B = 2*sqrt(T)` MACs per value instead of `2*sqrt(N)`, and at N=512
that was 2.19x - 2.42x when the classic path still did a 32-byte-per-row ND2NZ
gather. The `C == 16` split landed earlier in this log gives the classic path
the same conversion-free transfers structurally, at no MAC cost at all, so there
is nothing left for the bit-plan to buy:

| batch | bit-plan / classic at N=512 |
|---|---|
| 8192 | 1.0495 |
| 16384 | 1.0299 |
| 32768 | 1.0152 |
| 65536 | 1.0082 |

It decays toward parity as the fixed cost amortises but never crosses it.

Before removing it, the one place it could still have paid was checked: `N=2048`
and `N=4096` are the only sizes whose classic split is neither `C == 16` nor
`S == 1`, so they are the only ones still doing the gather. Driving them through
the bit-plan's `T=8192` tile -- same tile volume, same tile count, 2x the MACs,
conversion-free transfers -- loses 1.25x - 1.33x at small batch for at most
1.026x at batch >= 4096. `T=16384` does not validate at any `N`.

The cheaper way to buy `kNzGm` at those two sizes was checked too and is worse:
forcing `S == 1` makes the transfers conversion-free at **no** MAC cost, and
still loses up to 2.37x, because the tile it shrinks is what fills the pipes.

| batch | N=2048 `S=1`/`S=4` | N=4096 `S=1`/`S=2` |
|---|---|---|
| 8 | 1.01 | 1.01 |
| 128 | 1.26 | 1.09 |
| 2048 | 2.09 | 1.40 |
| 8192 | 2.37 | 1.48 |

**So the gather is not worth buying at any price these sizes can pay**, and
`_BITPLAN_TILE`, `_BITPLAN_MIN_BATCH`, `bitplan_view`, `_bit_operand`,
`bitplan_operands` and the launcher's bulk-plus-remainder split are gone with
it. The launcher is back to one launch per call.

#### A method note: the first block of an msprof session reads high

Forcing `S == 1` at N=2048 first measured **1.35x faster** at batch 1 -- where
the dispatch macro proves both arms run the identical `runTwosided<2048,32,64,1>`
instantiation. A null arm (the same `.so` copied to a second path, both arms
rotated over four passes) read 1.0069x and the effect vanished. The 1.35x was
the first config in that session; nothing before it had warmed the path. Every
sweep here now opens with a throwaway block, and no conclusion rests on a
config that ran first.

### Cube versus vector over the whole grid

`cube/bench_cube_vs_vector.py` measures the dispatcher's choice against the
vector kernel over `BENCH_HIDDEN_DIMS x BENCH_BATCHES`, 104 shapes, and
`plot_hadamard_cube_compare.py` draws it. Device time, three interleaved passes
with the arm order rotated, null arm 1.013x at the 3 us scale. Every arm is
measured at two candidate `block_dim`s -- the best of a coarse sweep and the
kernel's own tile-matched schedule -- and the better taken, because a fixed
sweep grid misses the counts that divide the tile count evenly and reads
10% - 25% high wherever it does (`N=16384` at batch 5, 10, 32 and 40). Each
pass also opens with a throwaway launch OF EACH KIND, not just one: the first
launch of a given kernel in an msprof session reads up to 1.5x high, and one
warm block only covers the kind it ran.

`outputs/csv/hadamard_three_way.csv` carries the same run broken out per
kernel -- vector, one-sided and two-sided side by side at every shape, each
with the `block_dim` it was measured at. A cube column is blank where that
kernel cannot run the size (`N < 256` two-sided, `N > 512` one-sided), and over
this whole grid the two-sided kernel wins every shape it can run, so "best
cube" and "what the dispatcher picks" coincide at all 104 cells.

**It has to report two numbers, because the two sides are not equally tuned.**
The cube kernels carry a per-shape `block_dim` schedule; the vector kernel
launches on a fixed count whatever the shape. Giving each kernel its own best
count compares the kernels; letting each pick for itself compares what a caller
gets today.

| | geomean | min | max | cells the cube loses |
|---|---|---|---|---|
| each at its own best `block_dim` | **1.61x** | 0.85x | 3.66x | 5 of 104 |
| as shipped | **2.10x** | 1.67x | 3.66x | 0 of 104 |

Speedup at each kernel's own best `block_dim`:

| batch \ N | 128 | 256 | 512 | 1024 | 2048 | 4096 | 8192 | 16384 |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.85 | 0.96 | 0.88 | 1.12 | 1.24 | 1.57 | 1.88 | 2.47 |
| 5 | 0.93 | 1.09 | 1.13 | 1.29 | 1.36 | 1.62 | 1.85 | 2.41 |
| 8 | 0.99 | 1.11 | 1.17 | 1.27 | 1.35 | 1.57 | 1.80 | 2.33 |
| 10 | 1.03 | 1.12 | 1.26 | 1.35 | 1.48 | 1.69 | 1.85 | 2.31 |
| 16 | 1.05 | 1.18 | 1.33 | 1.36 | 1.41 | 1.59 | 1.76 | 2.18 |
| 20 | 1.04 | 1.20 | 1.31 | 1.43 | 1.53 | 1.66 | 1.82 | 2.18 |
| 32 | 1.11 | 1.30 | 1.39 | 1.44 | 1.53 | 1.62 | 1.77 | 1.81 |
| 40 | 1.12 | 1.30 | 1.41 | 1.53 | 1.58 | 1.69 | 1.75 | 1.82 |
| 64 | 1.18 | 1.36 | 1.43 | 1.56 | 1.59 | 1.91 | 2.16 | 2.41 |
| 128 | 1.28 | 1.40 | 1.53 | 1.54 | 1.83 | 2.02 | 2.39 | 2.27 |
| 256 | 1.44 | 1.48 | 1.56 | 1.76 | 1.96 | 2.39 | 2.98 | 2.65 |
| 512 | 1.63 | 1.54 | 1.70 | 1.78 | 2.19 | 2.67 | 3.25 | 2.60 |
| 1024 | 1.87 | 1.69 | 1.77 | 2.03 | 2.54 | 3.07 | 3.66 | 2.75 |

**The five cells the cube loses are all at batch <= 8 and `N` <= 512**: `N=128`
at batch 1, 5 and 8 (0.85x, 0.93x, 0.99x) and batch 1 at `N=256` and `N=512`
(0.96x, 0.88x). They are stable -- six rotated passes read 0.776 - 0.849 at
`N=128` batch 1. A cube tile is at least 16 samples and its operands must reach
L0A/L0B before any of them move, so at one to eight samples there is nothing to
amortise that fixed cost against, while the vector kernel's butterfly starts
on the first row. `N=128` is also one of the two sizes the split cannot reach,
so the one-sided kernel is the only cube option there and the loss runs
furthest.

The win grows along both axes and peaks at `N=8192`, where the cube reads 2124
GB/s against the vector kernel's 581. It falls back at `N=16384` because that
is one row per 32 KB UB tile for the vector kernel -- its best case, not the
cube's worst.

#### What the as-shipped column is actually measuring

The vector kernel launches on `BLOCK_DIM = cube_core_num = 24`. A sweep to 48
shows that is the right constant and not an oversight: at `block_dim` 32, 40 and
48 the part runs two waves of 24 blocks, so batch 1024 at N=16384 reads 123.7 us
at 24 against 177.8, 145.7 and 125.2. What it is missing is a **schedule** --
fewer cores at small batch, exactly what both cube kernels carry. Its own best
count beats its shipped one by up to **1.77x**, all of it at batch <= 256:

| N | batch 16 | batch 64 | batch 256 | batch 1024 |
|---|---|---|---|---|
| 256 | 1.61 | 1.55 | 1.27 | 1.00 |
| 1024 | 1.31 | 1.23 | 1.00 | 1.00 |
| 16384 | 1.07 | 1.05 | 1.00 | 1.00 |

So roughly half the as-shipped 2.10x is the cube kernels' schedule rather than
their arithmetic, and giving the vector kernel the same treatment would close
most of the gap below batch 256 and none of it above. The cube schedules
themselves are within **1.0107x** geomean of their own per-shape best over the
same grid, worst 1.234x at N=256 batch 8.

**Two controls that had to run first.** The vector kernel is in place, so the
harness cycles a pool of pre-filled buffers; a pool-size sweep shows device time
is flat in the pool (`pool` 1, 2, 10, 20 within 0.3%) except at the single
largest shape, where a 335 MB working set costs 2.7%. The reported numbers use
`pool=1`, the reading most favourable to the baseline. And the vector kernel's
butterfly is branch-free, so fp16 values diverging to `inf` across repeated
in-place launches cannot change its timing -- which the same control confirms,
`pool=1` timing identical to `pool=20`.
