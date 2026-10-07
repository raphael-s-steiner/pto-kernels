## Fast Hadamard

### Layout

- `standard/`: plain Hadamard kernel, benchmark, plot, and test files
- `cube/`: cube-unit (matmul) Hadamard kernel, benchmark, plot, and test files
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

msprof `op_summary`, median of 20 launches, `block_dim=24`, 910B2. Speedup is
cube over the `standard/` vector kernel.

| batch | N=256 | N=1024 | N=4096 |
|------:|------:|-------:|-------:|
|    64 | 1.04x | 1.26x  | 1.50x  |
|   256 | 0.52x | 1.28x  | 1.80x  |
|  1024 | 0.36x | 1.47x  | 1.97x  |
|  8192 | 0.63x | 1.64x  | 2.09x  |
| 65536 | 1.51x | 1.45x  | 1.74x  |

At batch 8192 that is 20.3 / 26.8 / 95.0 us for the cube kernel against
12.8 / 43.9 / 198.4 us for the vector one, i.e. up to 1413 GB/s.

`N=256` is the weak case: the balanced split makes both Hadamard factors
16x16, so each tile does very little work per descriptor, and the kernel only
wins once the batch is large enough to amortise that. The alternative for small
`N` is a single dense `Y = X * H_N` product, which trades 8x the FLOPs for far
fewer descriptors; it is not implemented here because it is no longer the
two-sided split this kernel is about.

`block_dim` is a runtime argument, not a property of the build, and the default
(`cube_core_num`, 24) is only optimal for the larger shapes. It is worth up to
1.71x to lower it on small ones -- measured optima:

| shape | best `block_dim` | gain over 24 |
|---|---:|---:|
| 256 x 256   |  3 | 1.68x |
| 1024 x 256  |  4 | 1.71x |
| 4096 x 256  |  8 | 1.36x |
| 8192 x 256  | 12 | 1.20x |
| 64 x 1024   |  8 | 1.24x |
| 64 x 4096   | 16 | 1.09x |
| batch >= 1024, N >= 1024 | 24 | 1.00x |

The optimum does not follow a clean rule across `N`, so the wrapper ships the
constant rather than a fitted schedule; pass `block_dim=` to override it.

The cube path is also the more accurate of the two: it accumulates in fp32 and
rounds to fp16 twice instead of `log2(N)` times, which measures a relative RMS
error of 2.9e-4 against an exact fp64 reference, versus 6.3e-4 to 8.4e-4 for
the vector kernel.
