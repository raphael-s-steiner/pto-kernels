"""Device-time benchmark: run configs under msprof, report per-launch device us.

The event-timed harness in bench_hadamard_cube.py cannot see this kernel: a
launch costs ~46-59 us on the host (most of it torch_npu's task-queue drain)
while the kernel itself is 3-95 us on device, so the reported duration is flat
across a 64x range of work. msprof reads the duration off the device task
record instead, which is the only way to compare kernels here.

Usage (cwd = this example's root, needs msprof on PATH):
    python3 cube/devtime_bench.py <configs.json> [out.json]

where configs.json is a list of
    {"tag": "...", "kind": "cube"|"std", "so": "<path>", "batch": N,
     "n": N, "bd": 24, "reps": 30, "warmup": 5}

Each config runs `warmup` + `reps` launches back to back. The AI_CORE tasks in
msprof's op_summary come out in launch order, so the blocks are chunked in the
same order the configs were run and the warmup launches are dropped.
"""
import csv
import glob
import json
import pathlib
import shutil
import statistics
import subprocess
import sys

BASE = pathlib.Path(__file__).resolve().parent
PROF_DIR = BASE / "outputs" / "prof"

KERNEL_MARKERS = ("fast_hadamard_cube", "fast_hadamard_fp16")


def run_msprof(cfg_path: pathlib.Path) -> pathlib.Path:
    if PROF_DIR.exists():
        shutil.rmtree(PROF_DIR, ignore_errors=True)
    PROF_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        "msprof",
        f"--application=python3 {BASE / 'devtime_run.py'} {cfg_path}",
        f"--output={PROF_DIR}",
        "--ai-core=on",
        "--aicpu=off",
        "--task-time=on",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout[-4000:] + proc.stderr[-4000:])
        raise RuntimeError("msprof failed")
    matches = glob.glob(str(PROF_DIR / "PROF_*" / "mindstudio_profiler_output" / "op_summary_*.csv"))
    if not matches:
        raise RuntimeError("no op_summary produced")
    return pathlib.Path(sorted(matches)[-1])


def kernel_tasks(summary: pathlib.Path):
    rows = []
    with summary.open() as fh:
        for row in csv.DictReader(fh):
            name = row.get("Op Name", "")
            if any(m in name for m in KERNEL_MARKERS):
                rows.append(
                    (float(row["Task Start Time(us)"].strip()),
                     float(row["Task Duration(us)"]), name)
                )
    rows.sort(key=lambda r: r[0])
    return rows


def main():
    cfg_path = pathlib.Path(sys.argv[1]).resolve()
    configs = json.loads(cfg_path.read_text())
    tasks = kernel_tasks(run_msprof(cfg_path))

    expected = sum(int(c.get("warmup", 5)) + int(c.get("reps", 30)) for c in configs)
    if len(tasks) != expected:
        raise RuntimeError(f"expected {expected} kernel tasks, msprof reported {len(tasks)}")

    results, cursor = [], 0
    for cfg in configs:
        warmup, reps = int(cfg.get("warmup", 5)), int(cfg.get("reps", 30))
        block = tasks[cursor + warmup: cursor + warmup + reps]
        cursor += warmup + reps
        durations = sorted(d for _, d, _ in block)
        batch, n = int(cfg["batch"]), int(cfg["n"])
        median = statistics.median(durations)
        results.append({
            **cfg,
            "median_us": median,
            "min_us": durations[0],
            "max_us": durations[-1],
            "p25_us": durations[len(durations) // 4],
            "gbs": (2 * batch * n * 2 / 1e9) / (median / 1e6) if median else 0.0,
        })

    width = max((len(str(r.get("tag", ""))) for r in results), default=3)
    print(f"{'tag':<{width}} {'kind':<6} {'batch':>7} {'n':>6} {'bd':>3} "
          f"{'median_us':>10} {'min_us':>9} {'max_us':>9} {'GB/s':>8}")
    for r in results:
        print(f"{str(r.get('tag', '')):<{width}} {r.get('kind', 'cube'):<6} "
              f"{r['batch']:>7} {r['n']:>6} {r.get('bd', 24):>3} "
              f"{r['median_us']:>10.3f} {r['min_us']:>9.3f} {r['max_us']:>9.3f} {r['gbs']:>8.1f}")

    if len(sys.argv) > 2:
        pathlib.Path(sys.argv[2]).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
