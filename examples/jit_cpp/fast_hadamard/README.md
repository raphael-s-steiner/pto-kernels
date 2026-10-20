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
 {"tag":"bp", "kind":"bitplan", "batch":8192,"n":512, "bd":24,"reps":10,"warmup":3},
 {"tag":"std","kind":"std",     "batch":8192,"n":4096,"bd":24,"reps":10,"warmup":3}]
EOF
python cube/devtime_bench.py /tmp/cfg.json
```

### The combined entry point (`auto/`)

`auto/jit_util_hadamard_auto.py` exposes one out-of-place `fn(x, y)` that picks
the kernel from `N`:

| `N` | kernel | why |
|---|---|---|
| 64 .. 512 | one-sided | `N*N` MACs per sample, but one pipe pass per tile |
| 1024 .. 32768 | two-sided | `N*(R+C)` MACs, at the cost of a turnaround |

The one-sided product's MACs grow 4x per doubling of `N` against 2x for its
bytes, so it wins while `N` is small and loses once arithmetic dominates; the
split's extra turnaround costs more than it saves while `N` is small. They cross
between 512 and 1024. Each back end is compiled the first time a size needs it,
and `pick_implementation(n)` answers the routing question without compiling
anything.

**The vector kernel is not one of the branches, because it is never the fastest
choice in this range** -- see the table below. It stays in `standard/` as the
baseline, and as the only kernel that runs `N < 64`.

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
that weakest column has since improved: at batch 256 - 1024 the entry point now
returns the bit-plan two-sided tile and reads 1.28x - 1.45x rather than 1.24x -
1.26x. Every other column reproduces: re-measured at batch 1024, N=16384 is
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
  256 up (1.02x, 1.13x, 1.41x, 1.70x). `BITPLAN_BATCH_MIN = 256` in `auto/` is
  exactly this boundary, and the 1.02x at the boundary itself says the fit is
  tight rather than lucky.
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

#### One host-side cost the device-time grid hides: `batch < 16`

The one-sided kernel's smallest tile is 16 samples, so its wrapper pads a
shorter batch into a cached 16-row scratch. That costs two extra torch copies
around the launch, and each one forces another `torch_npu` task-queue drain:
wall-clock goes from ~63 us at `batch >= 16` to **243 - 279 us below it**,
against ~57 - 59 us for the vector kernel at the same shapes. Device time is
unchanged -- the kernel still runs one 16-sample tile -- so this does not appear
anywhere in the grid above.

It matters because `auto` routes every `N <= 512` to this kernel, so at
`batch < 16` the dispatcher returns the kernel that is ~4.3x slower in
wall-clock while being 1.4 - 2.1x faster on device. If short batches are a real
workload, the fix is to pad without the round trip (launch directly on a
16-row view of a caller-sized allocation), not to re-route.

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
| two-sided N=512 | ND->NZ converting | **yes**, bit-plan T=8192 | **2.19-2.42x, shipped** |
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
| `PTO_OS_TILE_SPLIT` | split per-core work by samples instead of tiles | the sample split walks the halving chain on most cores |
| `PTO_TS_BITPLAN` (host) | bit-plan on/off | always on; it is 2.19-2.42x at the one size it applies to |

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
when this table was measured. At N=512 it now ships the bit-plan tile instead,
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
(8, 16, 16, 16) to (7.51, 7.23, 7.51, 22.57). **Sweep every integer `block_dim`,
not a subset.** A first fit over `{1,2,3,4,6,8,12,16,20,24}` picked constants
that emit `bd = 15` and `23`, values it had never measured, and scored them
against their nearest neighbour. It reported 1.013x - 1.031x worst regret; the
real cost at N=64 batch 8192 was 1.12x, because 32 tiles divide evenly over 16
cores and badly over 15. Refitted over all 24 values: worst regret 1.000x -
1.041x, against 1.18x - 1.21x for a constant 24.

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
