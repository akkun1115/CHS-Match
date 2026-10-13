#!/usr/bin/env python3
"""Summarize the nine CHS-Match main runs (not third-party baselines).

After running bash scripts/run_main.sh, compare these results with the
CHS-Match row of Table 3.  This script does not execute training.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

MAIN_RUNS = {
    "LastFM": {"layers": 2, "seeds": (167, 39, 193), "paper": (91.19, 0.45)},
    "IMDB": {"layers": 3, "seeds": (84, 61, 146), "paper": (98.80, 0.15)},
    "DBLP": {"layers": 5, "seeds": (315, 366, 88), "paper": (96.01, 0.12)},
}


def is_main_run(result: dict) -> bool:
    dataset = result.get("dataset")
    expected = MAIN_RUNS.get(dataset)
    return bool(
        expected
        and result.get("seed") in expected["seeds"]
        and result.get("layers") == expected["layers"]
        and result.get("hop") == 1
        and result.get("normalization") == "partial_sinkhorn"
        and result.get("sinkhorn_iters") == 5
        and result.get("ablation") == "full"
        and result.get("lambda_mm") == 0.2
        and result.get("epochs") == 500
        and result.get("batch_size") == 32
        and result.get("hidden_dim") == 64
        and result.get("heads") == 4
        and result.get("learning_rate") == 1e-3
    )


def summarize(results_dir: Path, require_complete: bool = False) -> int:
    runs: dict[tuple[str, int], tuple[Path, dict]] = {}
    for path in sorted(results_dir.rglob("*.json")):
        with path.open(encoding="utf-8") as file:
            result = json.load(file)
        if not isinstance(result, dict) or not is_main_run(result):
            continue
        key = (result["dataset"], result["seed"])
        if key in runs:
            print(
                f"Duplicate main run {key}: {runs[key][0]} and {path}. "
                "Move one result outside the results directory before summarizing.",
                file=sys.stderr,
            )
            return 2
        runs[key] = (path, result)

    missing = []
    print("Dataset  Completed  Measured mean ± sample std (%)  Paper Table 3 (%)  Difference (pt)")
    print("-" * 99)
    for dataset, config in MAIN_RUNS.items():
        values = []
        for seed in config["seeds"]:
            entry = runs.get((dataset, seed))
            if entry is None:
                missing.append(f"{dataset} seed={seed}")
                continue
            value = float(entry[1]["test_acc"])
            if not 0 <= value <= 1:
                raise ValueError(f"Expected fractional test_acc (0..1) in {entry[0]}")
            values.append(value * 100)
        count = len(values)
        if count == 3:
            mean, sd = statistics.mean(values), statistics.stdev(values)
            paper_mean, paper_sd = config["paper"]
            print(
                f"{dataset:<8} {count}/3        {mean:6.2f} ± {sd:5.2f}"
                f"                 {paper_mean:5.2f} ± {paper_sd:.2f}"
                f"         {mean-paper_mean:+6.2f}"
            )
        else:
            print(f"{dataset:<8} {count}/3        (not enough runs to compare)")
    if missing:
        print("Missing runs: " + ", ".join(missing))
    print(
        "Values may vary with the environment. Matching the reported means requires "
        "running training; this script alone does not verify model equivalence."
    )
    return 2 if require_complete and missing else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    if not args.results_dir.is_dir():
        parser.error(f"Results directory not found: {args.results_dir}")
    sys.exit(summarize(args.results_dir, args.require_complete))


if __name__ == "__main__":
    main()
