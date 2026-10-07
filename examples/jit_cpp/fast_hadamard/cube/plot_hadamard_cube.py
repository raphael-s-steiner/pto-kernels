import argparse
from pathlib import Path

from plot_common import (
    add_common_plot_args,
    plot_comparison_csv,
    plot_csv_collection,
    resolve_dir_arg,
)

CSV_PREFIX = "fht_cube_compare_bd"

LINE_PLOTS = (
    {
        "filename": "hadamard_cube_duration_bd{block_dim}.png",
        "series": (
            ("cube_duration_us", "Cube PTO (matmul)", "#dc2626", "s-"),
            ("vector_duration_us", "Vector PTO (butterfly)", "#2563eb", "o-"),
        ),
        "y_label": "Duration (us)",
        "title": "Cube vs Vector Hadamard Duration",
    },
    {
        "filename": "hadamard_cube_bandwidth_bd{block_dim}.png",
        "series": (
            ("cube_bandwidth_gbs", "Cube PTO (matmul)", "#dc2626", "s-"),
            ("vector_bandwidth_gbs", "Vector PTO (butterfly)", "#2563eb", "o-"),
        ),
        "y_label": "Bandwidth (GB/s)",
        "title": "Cube vs Vector Hadamard Bandwidth",
    },
)

HEATMAPS = (
    {
        "filename": "hadamard_cube_speedup_vs_vector_bd{block_dim}.png",
        "key": "cube_speedup_vs_vector",
        "title": "Cube PTO Speedup over Vector PTO",
    },
)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Plot cube-vs-vector Hadamard benchmark results from CSV files."
    )
    return add_common_plot_args(parser).parse_args()


def plot_hadamard_cube(csv_path, plot_dir):
    plot_comparison_csv(
        csv_path,
        plot_dir,
        prefix=CSV_PREFIX,
        line_plots=LINE_PLOTS,
        heatmaps=HEATMAPS,
    )


def main():
    args = _parse_args()
    base = Path(__file__).resolve().parent

    csv_dir = resolve_dir_arg(base, args.csv_dir)
    plot_dir = resolve_dir_arg(base, args.plot_dir)

    plot_csv_collection(
        csv_dir,
        plot_dir,
        pattern=f"{CSV_PREFIX}*.csv",
        prefix=CSV_PREFIX,
        warning="no cube Hadamard benchmark CSV files found",
        plot_csv_fn=plot_hadamard_cube,
    )


if __name__ == "__main__":
    main()
