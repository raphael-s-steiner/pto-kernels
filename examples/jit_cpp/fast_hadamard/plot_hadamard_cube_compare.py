"""Plot the cube Hadamard kernels against the vector one from device time.

Input is the CSV written by `cube/bench_cube_vs_vector.py`. Both block_dim
policies it records are plotted, because they answer different questions: the
`best` pair compares the two kernels, the `shipped` pair compares what a caller
gets today, and the gap between them is the vector kernel's missing per-shape
schedule rather than anything about the algorithms.
"""
import argparse
from pathlib import Path

from plot_common import (
    add_common_plot_args,
    ensure_matplotlib,
    load_nonempty_rows,
    make_batched_line_plot,
    make_speedup_heatmap,
    resolve_dir_arg,
)

BASE_DIR = Path(__file__).resolve().parent
CSV_NAME = "hadamard_cube_vs_vector.csv"

HEATMAPS = (
    ("speedup_best", "hadamard_cube_vs_vector_speedup_best.png",
     "per-shape best", "Cube speedup over the vector kernel"),
    ("speedup_shipped", "hadamard_cube_vs_vector_speedup_shipped.png",
     "each kernel's own default", "Cube speedup over the vector kernel"),
)

LINE_PLOTS = (
    (
        "hadamard_cube_vs_vector_time_best.png",
        "per-shape best",
        (("cube_best_us", "cube", "tab:blue", "o-"),
         ("vector_best_us", "vector", "tab:orange", "s--")),
        "device time (us)",
        "Hadamard device time, cube vs vector",
    ),
    (
        "hadamard_cube_vs_vector_bandwidth.png",
        "per-shape best",
        (("cube_best_gbs", "cube", "tab:blue", "o-"),
         ("vector_best_gbs", "vector", "tab:orange", "s--")),
        "effective GB/s",
        "Hadamard effective bandwidth, cube vs vector",
    ),
)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_plot_args(parser)
    return parser.parse_args()


def plot_cube_compare(csv_path: Path, plot_dir: Path):
    rows = load_nonempty_rows(csv_path)
    if not rows:
        return

    for key, name, policy, title in HEATMAPS:
        make_speedup_heatmap(rows, policy, plot_dir / name, key, title)

    for name, policy, series, y_label, title in LINE_PLOTS:
        make_batched_line_plot(rows, policy, plot_dir / name, series,
                               y_label, title)


def main():
    args = _parse_args()
    if not ensure_matplotlib():
        return
    csv_path = resolve_dir_arg(BASE_DIR, args.csv_dir) / CSV_NAME
    if not csv_path.exists():
        print(f"Warning: {csv_path} not found; "
              "run cube/bench_cube_vs_vector.py first.")
        return
    plot_cube_compare(csv_path, resolve_dir_arg(BASE_DIR, args.plot_dir))


if __name__ == "__main__":
    main()
