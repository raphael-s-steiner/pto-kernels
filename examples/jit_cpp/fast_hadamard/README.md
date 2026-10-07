## Fast Hadamard

### Layout

- `standard/`: plain Hadamard kernel, benchmark, plot, and test files
- `cube/`: cube-unit (matmul) Hadamard kernel, benchmark, plot, and test files
- `onesided/`: cube-unit Hadamard as a SINGLE dense product `Y = X * H_N`, for
  `N` in 64..512 -- benchmark, plot, and test files
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
# cube-unit Hadamard, also timed against the standard vector kernel
python benchmark.py --target cube --npu 0
# one-sided (single dense matmul) Hadamard over N in {64, 128, 256, 512}, timed
# against the vector kernel and, where N >= 256, the two-sided cube kernel
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
# cube-vs-vector duration, bandwidth, and speedup plots
python plot.py --target cube
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
python test.py --target cube -v --npu "npu:0"
python test.py --target onesided -v --npu "npu:0"
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
- `cube/outputs/csv/fht_cube_compare_bd*.csv` from `benchmark.py --target cube`
- `onesided/outputs/csv/fht_onesided_compare_bd*.csv` from `benchmark.py --target onesided`
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

In `benchmark.py --target cube`, the cube kernel applies the balanced
Kronecker split `H_n = H_R (x) H_C` as `Y = H_R * X * H_C`, i.e. two matmuls
on the cube unit per tile of samples instead of `log2(N)` vector butterfly
stages. It is out-of-place (`fn(x, y)`) and supports `N` in `[256, 16384]`,
so its hidden-dimension grid starts at 256 instead of 128. The same shapes
are also timed with the `standard/` vector kernel, reported as `vector_*` in
the CSV and plots alongside the `cube_speedup_vs_vector` column.

The cube arm launches with the wrapper's per-shape `block_dim` schedule, so the
`bd` in the CSV filename is the cap (`cube_core_num`) and the `cube_block_dim`
column records what each shape actually ran at.

Both arms resolve the stream per call: `torch.npu.current_stream()` drains the
torch_npu async task queue, which is what orders a raw ctypes launch after the
torch ops that fill its inputs. That drain costs ~27 us per call, more than the
device time of every shape in the grid, so these CSV durations are launch-bound
for both kernels and the `cube_speedup_vs_vector` column mostly compares
wrapper overhead. Hoisting the stream pointer out of the loop does not fix
this -- it drops the drain, and the kernel can then run ahead of the torch ops
setting up its inputs and of the events meant to bracket it.

Device time needs a profiler. `cube/devtime_bench.py` runs a list of shapes
under `msprof` and reads the per-launch duration off the device task record:

```bash
cat > /tmp/cfg.json <<'EOF'
[{"tag":"cube","kind":"cube","batch":8192,"n":4096,"bd":24,"reps":10,"warmup":3},
 {"tag":"std", "kind":"std", "batch":8192,"n":4096,"bd":24,"reps":10,"warmup":3}]
EOF
python cube/devtime_bench.py /tmp/cfg.json
```

### Measured device time

msprof `op_summary`, median of 20 launches over 3 interleaved passes, 910B2,
with the shipped `block_dim` schedule. The ratio is `vector_us / cube_us`, so
**above 1.0 means the cube kernel is faster** and below 1.0 means the vector
kernel in `standard/` wins. The `cube_speedup_vs_vector` column in the CSV uses
the same convention.

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
for small `N` is a single dense `Y = X * H_N` product, which trades 8x the FLOPs
for far fewer descriptors; it is not implemented here because it is no longer
the two-sided split this kernel is about.

### Why N=256 is the weak case

The cube kernel's *slope* is the better one at every size: fitted across batch
at N=256 it is 0.629 ns/sample against the vector kernel's 1.196, i.e. 1.90x
leaner. What it carried was a ~13 us intercept. Four candidate causes were
measured and all four refuted -- block dispatch is 0.90 us at `block_dim=24`,
the per-core Hadamard-factor prologue is free (0.90 us, identical to an empty
kernel), deleting both `TMATMUL`s saves 2-13% (`PTO_HAD_NOMM`), and reps=20 and
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

- **`PTO_HAD_DBUF`** splits the C==16 tile body at its L1 turnaround into two
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
from git HEAD, not from the `PTO_HAD_DBUF=0` switch) over all 7 sizes x 5
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

Both are gated: build with `-DPTO_HAD_DBUF=0` for the single-buffered control,
and pass `block_dim=` to override the schedule. `cube/build_variant.sh` builds
one switch combination and `cube/validate_variant.py` checks a prebuilt variant
against an exact fp64 reference -- run it before timing anything, since a switch
that computes the wrong answer is a broken lever, not a slow technique. `block_dim` must not change
results, and `test_block_dim_does_not_change_result` asserts bit-identical
output across eight values of it.

One caution learned here: the deep path needed extra `set_flag`/`wait_flag`
event ids, and renumbering the *existing* ids -- in a path that was otherwise
untouched -- cost a reproducible 0.92x at N=1024 batch 65536. The two drivers
now keep separate id names over a shared pool so the shallow path's flags are
exactly what they were. Measure renumbering as its own change.

### The one-sided kernel (`onesided/`, N = 64..512)

`cube/` applies the balanced Kronecker split and pays two matmuls of about
sqrt(N) on a side. `onesided/` instead does the whole transform as one dense
product `Y = X * H_N`. That costs sqrt(N)/2 times as many MACs -- 8x at N=256 --
and buys a tile body that touches each pipe exactly ONCE:

```
GM --MTE2--> L1 --MTE1--> L0A --M--> L0C --FIX--> GM
```

which is precisely what the two-sided tile cannot do. Its tile has to turn the
intermediate around through L1 between the two multiplies, so it visits MTE1 and
M twice and consecutive tiles serialize behind the second visit -- the effect
measured above, where a tile costs 340 ns on a core with a short run against
91 ns on a long one. On the cube a fp16 matmul is nearly free beside the data
movement, so trading MACs for a straight pipe is the right side of the trade
while N is small.

Measured, paired over 3 interleaved passes, with each kernel's own `block_dim`
schedule. Ratios above 1.0 mean the one-sided kernel is faster:

| N | batch | bd | one-sided us | GB/s | vector | vs vector | two-sided | vs two-sided |
|---:|------:|---:|-------------:|-----:|-------:|----------:|----------:|-------------:|
|  64 |   256 |  3 |   1.90 |   34 |   3.25 |  1.71x | n/a | -- |
|  64 |  1024 |  6 |   2.02 |  130 |   4.00 |  1.98x | n/a | -- |
|  64 |  8192 | 16 |   2.74 |  765 |  10.67 |  3.89x | n/a | -- |
|  64 | 65536 | 24 |   5.51 | 3045 |  63.14 | **11.46x** | n/a | -- |
| 128 |   256 |  6 |   2.36 |   56 |   3.29 |  1.39x | n/a | -- |
| 128 |  1024 | 11 |   2.66 |  197 |   4.19 |  1.58x | n/a | -- |
| 128 |  8192 | 24 |   3.90 | 1076 |  12.73 |  3.26x | n/a | -- |
| 128 | 65536 | 24 |  10.24 | 3277 |  79.27 |  7.74x | n/a | -- |
| 256 |   256 |  8 |   3.26 |   80 |   3.38 |  1.04x |  6.88 | **2.11x** |
| 256 |  1024 | 16 |   3.86 |  272 |   4.06 |  1.05x | 12.16 | **3.15x** |
| 256 |  8192 | 24 |   8.36 | 1003 |  12.74 |  1.52x | 20.12 | **2.41x** |
| 256 | 65536 | 24 |  39.63 | 1693 |  81.34 |  2.05x | 42.78 | 1.08x |
| 512 |   256 |  8 |   6.26 |   84 |   3.66 |  0.58x | 12.08 | 1.93x |
| 512 |  1024 | 16 |   7.04 |  298 |   5.47 |  0.78x | 17.10 | 2.43x |
| 512 |  8192 | 24 |  31.12 |  539 |  21.57 |  0.69x | 29.03 | 0.93x |
| 512 | 65536 | 24 | 208.58 |  644 | 149.29 |  0.72x | 112.72 | 0.54x |

Faster than the vector kernel at **all 12 shapes with `N <= 256`**, geometric mean
**2.35x**, up to 11.46x. It also adds N=64 and N=128 to the cube path, which the
two-sided split cannot reach at all (it needs `N = R * C` with both factors
16-aligned, so `N >= 256`). And at N=256 it is 1.08x to 3.15x faster than the
two-sided kernel -- exactly the weak case diagnosed above: `N=256` batch 1024
goes from 0.66x of the vector kernel to 1.05x.

The GB/s column exceeds HBM at the small sizes because at those batches the
whole working set is L2-resident (N=64 batch 65536 is 16.8 MB). Both arms read
the same data, so the comparison is unaffected.

It is also the most accurate of the three: fp32 accumulation rounded to fp16
exactly ONCE, measured 2.16e-4 worst relative RMS against an exact fp64
transform, against ~3.0e-4 for the two-sided split (two roundings) and 6.3e-4 to
8.4e-4 for the vector butterfly (`log2(N)` roundings).

**Where it stops paying, and why 512 is the ceiling.** All of H is read once per
tile, so H's L1->L0B traffic per sample is `N*N*2 / S` bytes against `4*N` bytes
of HBM traffic for X and Y. With `S*N` fixed by L0C capacity that ratio grows as
`N^2`, giving three regimes:

| N | H size | where H lives | H traffic vs data | result |
|---:|-------:|---|---|---|
| <= 128 | <= 32 KB | **L0B, for the whole kernel** | zero | 1.4x - 11.5x vs vector |
| 256 | 128 KB | L1 (too big for L0B's 64 KB) | ~1:1 | 1.0x - 2.1x vs vector |
| 512 | 512 KB | nowhere -- it IS all of L1, so it streams from GM per tile | ~4:1 | 0.6x - 0.8x vs vector |

That ratio is also why the two sizes want different tile heights, which is the
one tuning decision here that does not generalise across `N`. Doubling the tile
halves H's per-sample traffic but costs the second L0C accumulator: at N=512,
where H traffic is 4:1, it is worth **1.57x - 1.90x**; at N=256, where it is only
~1:1, the same change **loses** 0.84x - 0.99x. So `PTO_OS_S512` is 64 while
`PTO_OS_S256` stays at 16384/256. `S * N = 32768` fills L0C, so N=512 is already
at the tallest tile the part allows and 4:1 is its floor.

So N=512 is supported but is not the right kernel there: prefer the vector kernel
at N=512, except below ~1024 batch where this one is still 1.9x - 2.4x faster
than `cube/`. A noop-floor probe (`PTO_OS_NOMM=1`,
matmuls deleted, every transfer kept) confirms the mechanism rather than
assuming it -- the matmuls are **0%** of N=512's time and 7-8% of N=256's, so
both are purely movement-bound and no amount of overlapping the chunk moves with
the matmuls can help. That is also why the chunk-level double buffering is
allocated but deliberately left unexploited: the probe says there is nothing
there to win. At N=64/128 the matmuls are 6-33%, and those sizes keep H resident
so they have no chunk loop at all.

Guidance, by `N`:

| N | use |
|---:|---|
| 64, 128 | `onesided/` |
| 256 | `onesided/` |
| 512 | `standard/` (vector); `onesided/` below ~1024 batch, `cube/` above ~8192 |
| >= 1024 | `cube/` (two-sided) |

**One gotcha worth recording**, because it is the flip side of the whole design.
The straight-pipe body has no edge holding MTE1 back, so MTE1 runs ahead by
slots and tile `i+2`'s `TMOV` lands on the L0A slot tile `i`'s `TMATMUL` is still
reading. The two-sided kernel gets that WAR protection for free from its own L1
turnaround; here it has to be an explicit `M -> MTE1` flag per slot. Without it
the kernel faults, at a shape that moves between runs. A second instance of the
same class: staging an H chunk with the generic `Sync<PIPE_MTE2, PIPE_MTE1>()`
helper, whose id 0 collides with `EV_X0` on that very pipe pair -- only the
streamed-H case runs that Sync inside the tile loop, so it corrupted the X ring's
credits at N=512 and nowhere else.

Compile-time switches are documented at the top of the kernel;
`onesided/build_variant.sh` builds one combination and
`onesided/validate_variant.py` checks a prebuilt variant against the exact fp64
reference over 168 cases. Validate before timing: a switch that computes the
wrong answer is a broken lever, not a slow technique.

The cube path is also the more accurate of the two: it accumulates in fp32 and
rounds to fp16 twice instead of `log2(N)` times, which measures a relative RMS
error of 2.9e-4 against an exact fp64 reference, versus 6.3e-4 to 8.4e-4 for
the vector kernel.
