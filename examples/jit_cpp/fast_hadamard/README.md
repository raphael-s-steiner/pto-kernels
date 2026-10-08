## Fast Hadamard

### Layout

- `standard/`: plain Hadamard kernel, benchmark, plot, and test files
- `cube/`: cube-unit (matmul) Hadamard kernel, benchmark, plot, and test files
- `onesided/`: cube-unit Hadamard as a single right-multiplication `Y = X * H_N`,
  with `H_N = H_B (x) H_M` applied as fixed-size matmuls accumulated in L0C, for
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
# one-sided (blocked right-multiply) Hadamard over N in {64, 128, 256, 512}, timed
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
for small `N` is the one-sided `Y = X * H_N` product, which trades 8x the FLOPs
for a tile body that touches each pipe once; that lives in `onesided/` rather
than here, because it is no longer the two-sided split this kernel is about, and
it is 1.25x - 2.44x faster than this kernel at every `N <= 512`.

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
sqrt(N) on a side. `onesided/` instead applies `H_N` from the right only, which
costs `N*N` MACs per sample -- sqrt(N)/2 times as many, 8x at N=256 -- and buys a
tile body that touches each pipe exactly ONCE:

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
binds instead is arithmetic. A noop-matmul probe (`PTO_OS_NOMM=1`, matmuls
deleted, every transfer and handshake kept) says so directly -- dropping the
matmuls is **3.96x** faster at N=512 batch 65536 and 2.34x at N=256, where
before the change the same probe read 0% and 7-8%. The kernel moved from
movement-bound to compute-bound, which is the intended direction: the bound it
now sits against is `N*N` MACs per sample, not a buffer size.

**That makes 512 a compute crossover rather than a capacity limit.**
`PTO_OS_BIG=1` compiles N=1024 and 2048, which the blocked form computes
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
| 64 .. 512 | `onesided/` |
| >= 1024 | `cube/` (two-sided) |

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

Compile-time switches are documented at the top of the kernel;
`onesided/build_variant.sh` builds one combination and
`onesided/validate_variant.py` checks a prebuilt variant against the exact fp64
reference over 168 cases (`onesided/bigcheck_variant.py` covers the
`PTO_OS_BIG` sizes). Validate before timing: a switch that computes the wrong answer is a
broken lever, not a slow technique.
