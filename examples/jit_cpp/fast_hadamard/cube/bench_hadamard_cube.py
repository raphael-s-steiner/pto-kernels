import argparse
import math
from pathlib import Path

import torch
import torch_npu  # noqa

from bench_common import (
    POOL_EMPTY_FP16,
    POOL_RANDN_FP16,
    add_common_benchmark_args,
    bandwidth_gbs,
    benchmark_batches,
    benchmark_npu_us,
    benchmark_trials_us,
    ensure_output_dir,
    make_shape_pools,
    pool_item,
    resolve_dir_arg,
    validate_benchmark_args,
    write_csv_records,
)
from cube.jit_util_hadamard_cube import cube_block_dim_schedule
from cube.jit_util_hadamard_cube import jit_compile as jit_compile_cube
from standard.jit_util_hadamard import jit_compile as jit_compile_vector

DEFAULT_WARMUP = 10
DEFAULT_REPEATS = 100

# The cube path needs N = R * C with both factors 16-aligned, so its grid starts
# at 256 rather than the 128 the vector kernel supports.
BENCH_CUBE_HIDDEN_DIMS = [256, 512, 1024, 2048, 4096, 8192, 16384]

# The cube kernel is out of place and needs an output pool of its own; the
# vector kernel transforms its input buffer in place.
CUBE_COMPARE_POOL_KINDS = {
    "cube_x": POOL_RANDN_FP16,
    "cube_y": POOL_EMPTY_FP16,
    "vector_x": POOL_RANDN_FP16,
}

CSV_HEADER = (
    "batch,N,cube_block_dim,cube_duration_us,vector_duration_us,"
    "cube_bandwidth_gbs,vector_bandwidth_gbs,cube_speedup_vs_vector,trials,"
    "cube_duration_mean_us,cube_duration_std_us,cube_duration_min_us,"
    "cube_duration_max_us,cube_duration_cv_pct,vector_duration_mean_us,"
    "vector_duration_std_us,vector_duration_min_us,vector_duration_max_us,"
    "vector_duration_cv_pct\n"
)


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the cube-unit Hadamard PTO kernel against the vector "
            "butterfly kernel and save CSV outputs."
        )
    )
    return add_common_benchmark_args(
        parser,
        default_warmup=DEFAULT_WARMUP,
        default_repeats=DEFAULT_REPEATS,
    ).parse_args()


def cube_hidden_dims(args):
    return args.hidden_dims if args.hidden_dims else BENCH_CUBE_HIDDEN_DIMS


def benchmark_cube_hadamard_us(
    cube_func,
    x_list,
    y_list,
    *,
    block_dim,
    warmup,
    repeats,
) -> float:
    return benchmark_npu_us(
        warmup,
        repeats,
        lambda i: cube_func(
            pool_item(x_list, i),
            pool_item(y_list, i),
            block_dim=block_dim,
        ),
    )


def benchmark_vector_hadamard_us(
    vector_func,
    x_list,
    *,
    batch,
    n,
    log2_n,
    block_dim,
    warmup,
    repeats,
) -> float:
    return benchmark_npu_us(
        warmup,
        repeats,
        lambda i: vector_func(
            pool_item(x_list, i),
            batch,
            n,
            log2_n,
            block_dim=block_dim,
        ),
    )


def measure_cube_hadamard_shape(
    cube_func,
    vector_func,
    *,
    batch,
    n,
    cube_block_dim,
    vector_block_dim,
    warmup,
    repeats,
    trials,
    device,
):
    pools = make_shape_pools(
        batch,
        n,
        warmup,
        repeats,
        device=device,
        pool_kinds=CUBE_COMPARE_POOL_KINDS,
    )
    cube_stats = benchmark_trials_us(
        trials,
        lambda: benchmark_cube_hadamard_us(
            cube_func,
            pools["cube_x"],
            pools["cube_y"],
            block_dim=cube_block_dim,
            warmup=warmup,
            repeats=repeats,
        ),
    )
    vector_stats = benchmark_trials_us(
        trials,
        lambda: benchmark_vector_hadamard_us(
            vector_func,
            pools["vector_x"],
            batch=batch,
            n=n,
            log2_n=int(math.log2(n)),
            block_dim=vector_block_dim,
            warmup=warmup,
            repeats=repeats,
        ),
    )

    cube_us = cube_stats["median_us"]
    vector_us = vector_stats["median_us"]
    data_bytes = 2 * batch * n * 2
    result = {
        "batch": batch,
        "n": n,
        # What this shape actually launched with: cube_block_dim is None when the
        # wrapper's per-shape schedule is in charge, so resolve it for the record.
        "cube_block_dim": (
            cube_block_dim
            if cube_block_dim is not None
            else cube_block_dim_schedule(batch, n, cube_func.block_dim)
        ),
        "cube_us": cube_us,
        "vector_us": vector_us,
        "cube_bw": bandwidth_gbs(data_bytes, cube_us),
        "vector_bw": bandwidth_gbs(data_bytes, vector_us),
        "cube_speedup_vs_vector": vector_us / cube_us if cube_us > 0 else 0.0,
        "trials": trials,
    }
    for prefix, stats in (("cube", cube_stats), ("vector", vector_stats)):
        result.update(
            {
                f"{prefix}_mean_us": stats["mean_us"],
                f"{prefix}_std_us": stats["std_us"],
                f"{prefix}_min_us": stats["min_us"],
                f"{prefix}_max_us": stats["max_us"],
                f"{prefix}_cv_pct": stats["cv_pct"],
            }
        )
    return result


def print_cube_hadamard_shape_summary(result) -> None:
    print(
        f"{result['batch']:>6d}  {result['n']:>6d}"
        f"  {result['cube_us']:>10.2f}  {result['cube_cv_pct']:>8.2f}"
        f"  {result['vector_us']:>10.2f}  {result['vector_cv_pct']:>8.2f}"
        f"  {result['cube_bw']:>12.2f}  {result['vector_bw']:>14.2f}"
        f"  {result['cube_speedup_vs_vector']:>15.3f}"
    )


def format_cube_hadamard_csv_record(result) -> str:
    return (
        f"{result['batch']},{result['n']},{result['cube_block_dim']},"
        f"{result['cube_us']:.4f},"
        f"{result['vector_us']:.4f},{result['cube_bw']:.4f},"
        f"{result['vector_bw']:.4f},{result['cube_speedup_vs_vector']:.4f},"
        f"{result['trials']},{result['cube_mean_us']:.4f},"
        f"{result['cube_std_us']:.4f},{result['cube_min_us']:.4f},"
        f"{result['cube_max_us']:.4f},{result['cube_cv_pct']:.4f},"
        f"{result['vector_mean_us']:.4f},{result['vector_std_us']:.4f},"
        f"{result['vector_min_us']:.4f},{result['vector_max_us']:.4f},"
        f"{result['vector_cv_pct']:.4f}"
    )


def benchmark(
    cube_func,
    vector_func,
    *,
    warmup: int,
    repeats: int,
    trials: int,
    output_dir: Path,
    device: str,
    batches,
    hidden_dims,
):
    ensure_output_dir(output_dir)
    # None lets the cube wrapper apply its per-shape block_dim schedule, which is
    # what ships; pinning a constant here would benchmark a configuration nobody
    # runs. The vector kernel has no schedule, so it keeps its resolved default.
    cube_block_dim = None
    vector_block_dim = vector_func.block_dim

    # Both arms leave stream_ptr at None so that each call resolves it through
    # torch.npu.current_stream(). That is not a cacheable lookup: torch_npu
    # enqueues its ops on an async task queue, a raw ctypes launch goes straight
    # to the ACL stream, and resolving the stream drains the queue. Hoisting the
    # pointer out of the loop drops the drain and the kernel can then run before
    # the torch ops that fill its inputs -- and before the start/end events that
    # are supposed to bracket it, which invalidates the timing as well.
    #
    # The drain costs ~27 us per call, so shapes whose device time is below that
    # report launch cost rather than kernel cost. Both arms pay it equally, so
    # the comparison stays fair, but separating the kernels on device time needs
    # a profiler (msprof op_summary), not this harness.

    print(f"\n{'=' * 96}")
    print(
        f"CUBE vs VECTOR HADAMARD BENCHMARK "
        f"(cube BLOCK_DIM=scheduled per shape, max {cube_func.block_dim}, "
        f"vector BLOCK_DIM={vector_block_dim})"
    )
    print(f"{'=' * 96}")
    header = (
        f"{'batch':>6s}  {'N':>6s}"
        f"  {'cube_us':>10s}  {'cube_cv%':>8s}"
        f"  {'vector_us':>10s}  {'vec_cv%':>8s}"
        f"  {'cube_bw_gbs':>12s}  {'vector_bw_gbs':>14s}"
        f"  {'cube_vs_vector':>15s}"
    )
    print(header)
    print("-" * len(header))

    records = []

    for batch in batches:
        for n in hidden_dims:
            result = measure_cube_hadamard_shape(
                cube_func,
                vector_func,
                batch=batch,
                n=n,
                cube_block_dim=cube_block_dim,
                vector_block_dim=vector_block_dim,
                warmup=warmup,
                repeats=repeats,
                trials=trials,
                device=device,
            )
            print_cube_hadamard_shape_summary(result)
            records.append(format_cube_hadamard_csv_record(result))

    # Keyed on the cap, not the value used: the schedule picks per shape, and
    # the cube_block_dim column records what each row actually ran at.
    csv_path = output_dir / f"fht_cube_compare_bd{cube_func.block_dim}.csv"
    write_csv_records(csv_path, CSV_HEADER, records)
    print(f"\nSaved to {csv_path}")


def main():
    args = _parse_args()
    validate_benchmark_args(args)

    torch.npu.set_device(args.npu)
    base = Path(__file__).resolve().parent
    standard_base = base.parent / "standard"

    cube_kernel_path = base / "fast_hadamard_cube.cpp"
    vector_kernel_path = standard_base / "fast_hadamard.cpp"
    csv_dir = resolve_dir_arg(base, args.csv_dir)

    print(f"Using device: {args.npu}")
    print("Compiling fast_hadamard_cube.cpp ...")
    cube_func = jit_compile_cube(str(cube_kernel_path), verbose=True, device=args.npu)
    print("Compiling fast_hadamard.cpp ...")
    vector_func = jit_compile_vector(
        str(vector_kernel_path), verbose=True, device=args.npu
    )

    benchmark(
        cube_func,
        vector_func,
        warmup=args.warmup,
        repeats=args.repeats,
        trials=args.trials,
        output_dir=csv_dir,
        device=args.npu,
        batches=benchmark_batches(args),
        hidden_dims=cube_hidden_dims(args),
    )


if __name__ == "__main__":
    main()
