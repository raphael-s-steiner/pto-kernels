"""Device-time benchmark of the cube Hadamard kernels against the vector one.

The event-timed harness in `benchmark.py` cannot see these kernels at all: a
launch costs ~46-59 us on the host against 1-8 us of device work for most of
the grid, so the reported duration is flat across a 64x range of work. This
drives `devtime_bench.py` instead, which reads the duration off msprof's device
task record.

Two block_dim policies are reported side by side, because the two sides are not
equally tuned. The cube kernels carry a per-shape schedule; the vector kernel
launches on a fixed count whatever the shape, which costs it most of its small
batches. `speedup_best` gives each kernel its own best count and so compares
the kernels; `speedup_shipped` uses what each one picks on its own and so
compares what a caller gets today. Quoting only one of them would be a claim
about the other thing.

Usage (cwd = this example's root, needs msprof on PATH):
    python3 cube/bench_cube_vs_vector.py [--passes 3] [--csv-dir outputs/csv]
"""
import argparse
import json
import os
import pathlib
import statistics
import subprocess
import sys
import tempfile
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from auto.jit_util_hadamard_auto import pick_implementation
from bench_common import BENCH_BATCHES, BENCH_HIDDEN_DIMS, ensure_output_dir
from cube.jit_util_hadamard_onesided import onesided_block_dim_schedule
from cube.jit_util_hadamard_twosided import twosided_block_dim_schedule
from jit_util_common import get_cube_block_dim

BASE = pathlib.Path(__file__).resolve().parents[1]
CSV_NAME = "hadamard_cube_vs_vector.csv"
CSV_HEADER = (
    "batch,N,impl,cube_best_us,cube_best_bd,vector_best_us,vector_best_bd,"
    "cube_shipped_us,cube_shipped_bd,vector_shipped_us,vector_shipped_bd,"
    "speedup_best,speedup_shipped,cube_best_gbs,vector_best_gbs"
)

# The vector kernel can be launched on more blocks than the part has cube
# cores, so its sweep has to reach past them or it cannot find its own best.
CUBE_BLOCK_DIMS = [1, 2, 4, 8, 12, 16, 20, 24]
VECTOR_BLOCK_DIMS = CUBE_BLOCK_DIMS + [32, 40, 48]


def _interpreter_wrapper(tmp: pathlib.Path) -> pathlib.Path:
    """A shell script that execs this interpreter, for msprof to launch.

    msprof resolves its --application through the filesystem, and a venv's
    `bin/python` is a symlink to the system one -- which has no torch. Handing
    it a script instead defeats the resolution.
    """
    wrapper = tmp / "python-wrapper"
    wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    wrapper.chmod(0o755)
    return wrapper


def run_configs(configs, label):
    """Hand one msprof session a config list and return it with timings.

    Both the profile directory and the interpreter are passed through the
    environment and point into this scratch directory, so a checkout whose
    `outputs/` is not writable still runs and two runs cannot collide.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        cfg_path, out_path = tmp / "cfg.json", tmp / "out.json"
        cfg_path.write_text(json.dumps(configs))
        print(f"  {label}: {len(configs)} configs", flush=True)
        proc = subprocess.run(
            [sys.executable, str(BASE / "cube" / "devtime_bench.py"),
             str(cfg_path), str(out_path)],
            capture_output=True, text=True,
            env={**os.environ, "PTO_PROF_DIR": str(tmp / "prof"),
                 "PTO_PY": str(_interpreter_wrapper(tmp))},
        )
        if proc.returncode != 0:
            sys.stderr.write(proc.stdout[-4000:] + proc.stderr[-4000:])
            raise RuntimeError(f"devtime_bench failed for {label}")
        return json.loads(out_path.read_text())


def shipped_block_dim(n, batch, impl, max_block_dim):
    if impl == "twosided":
        return twosided_block_dim_schedule(batch, n, max_block_dim)
    return onesided_block_dim_schedule(batch, n, max_block_dim)


def sweep_best_block_dim(grid, kind_of, block_dims, reps, warmup):
    """The block_dim each shape is fastest at, one pass per (shape, count)."""
    configs = [{"tag": "warm", "kind": "std", "batch": 64, "n": 1024, "bd": 24,
                "reps": 5, "warmup": 5}]
    for n, batch in grid:
        for bd in block_dims:
            if bd > max(1, batch):
                continue
            configs.append({"tag": f"{n}|{batch}|{bd}", "batch": batch, "n": n,
                            "kind": kind_of(n, batch), "bd": bd,
                            "reps": reps, "warmup": warmup})
    seen = defaultdict(dict)
    for row in run_configs(configs, "block_dim sweep"):
        if row["tag"] == "warm":
            continue
        n, batch, bd = (int(v) for v in row["tag"].split("|"))
        seen[(n, batch)][bd] = row["median_us"]
    return {k: min(d, key=d.get) for k, d in seen.items()}


def measure_arms(grid, arms, passes, reps, warmup):
    """Run every arm of every shape in one session, rotating the arm order.

    All four arms of a shape go back to back so that whatever drifts over a
    session drifts for both sides of a ratio, and the order flips between
    passes because the first arm of a pair reads a few percent high at the
    3 us scale -- see the null-arm check in the README.
    """
    configs = [{"tag": "warm", "kind": "std", "batch": 64, "n": 1024, "bd": 24,
                "reps": 5, "warmup": 5}]
    for p in range(passes):
        for n, batch in grid:
            block = [{"tag": f"{n}|{batch}|{p}|{name}", "kind": kind, "bd": bd,
                      "batch": batch, "n": n, "reps": reps, "warmup": warmup}
                     for name, kind, bd in arms(n, batch)]
            if p % 2:
                block.reverse()
            configs += block
    out = defaultdict(list)
    for row in run_configs(configs, f"{passes} timed passes"):
        if row["tag"] == "warm":
            continue
        n, batch, _, name = row["tag"].split("|")
        out[(name, int(n), int(batch))].append(row["median_us"])
    return {k: statistics.median(v) for k, v in out.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=int, nargs="*",
                        default=BENCH_BATCHES)
    parser.add_argument("--hidden-dims", type=int, nargs="*",
                        default=BENCH_HIDDEN_DIMS)
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--reps", type=int, default=15)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--csv-dir", type=str, default="outputs/csv")
    args = parser.parse_args()

    max_bd = get_cube_block_dim()
    grid = [(n, b) for n in args.hidden_dims for b in args.batches]

    print("cube arm:")
    cube_best = sweep_best_block_dim(
        grid, pick_implementation, CUBE_BLOCK_DIMS, args.reps, args.warmup)
    print("vector arm:")
    vector_best = sweep_best_block_dim(
        grid, lambda n, b: "std", VECTOR_BLOCK_DIMS, args.reps, args.warmup)

    def arms(n, batch):
        impl = pick_implementation(n, batch)
        return [
            ("cube_best", impl, cube_best[(n, batch)]),
            ("vector_best", "std", vector_best[(n, batch)]),
            ("cube_shipped", impl,
             shipped_block_dim(n, batch, impl, max_bd)),
            ("vector_shipped", "std", max_bd),
        ]

    print("timed comparison:")
    timed = measure_arms(grid, arms, args.passes, args.reps, args.warmup)

    records, speedups = [], []
    for batch in args.batches:
        for n in args.hidden_dims:
            impl = pick_implementation(n, batch)
            cb = timed[("cube_best", n, batch)]
            vb = timed[("vector_best", n, batch)]
            cs = timed[("cube_shipped", n, batch)]
            vs = timed[("vector_shipped", n, batch)]
            moved = 2 * batch * n * 2 / 1e9
            speedups.append(vb / cb)
            records.append(
                f"{batch},{n},{impl},{cb:.3f},{cube_best[(n, batch)]},"
                f"{vb:.3f},{vector_best[(n, batch)]},{cs:.3f},"
                f"{shipped_block_dim(n, batch, impl, max_bd)},"
                f"{vs:.3f},{max_bd},"
                f"{vb / cb:.4f},{vs / cs:.4f},"
                f"{moved / (cb / 1e6):.1f},{moved / (vb / 1e6):.1f}"
            )

    csv_dir = pathlib.Path(args.csv_dir)
    if not csv_dir.is_absolute():
        csv_dir = BASE / csv_dir
    ensure_output_dir(csv_dir)
    csv_path = csv_dir / CSV_NAME
    csv_path.write_text(CSV_HEADER + "\n" + "\n".join(records) + "\n")
    print(f"\nwrote {csv_path} ({len(records)} shapes)")
    print("speedup at each kernel's own best block_dim: "
          f"geomean {statistics.geometric_mean(speedups):.3f}x, "
          f"min {min(speedups):.3f}x, max {max(speedups):.3f}x")


if __name__ == "__main__":
    main()
