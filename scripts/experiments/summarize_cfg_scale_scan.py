#!/usr/bin/env python3
"""Summarize terminal-geometry sensitivity results from cfg_scale_scan.slurm.sh."""

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("scan_dir", type=Path, help="Root containing scale_*/ output directories")
    parser.add_argument("--rmsd-cutoff", type=float, default=1.0)
    parser.add_argument("--expected-per-cell", type=int, default=10)
    return parser.parse_args()


def scale_from_dir(path):
    return float(path.name.removeprefix("scale_").replace("p", "."))


def wilson_interval(successes, total, z=1.96):
    if total == 0:
        return math.nan, math.nan
    rate = successes / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denominator
    return center - margin, center + margin


def read_records(scan_dir, cutoff):
    records = []
    for scale_dir in sorted(scan_dir.glob("scale_*"), key=scale_from_dir):
        scale = scale_from_dir(scale_dir)
        for result_path in sorted(scale_dir.glob("*/candidates/results.jsonl")):
            cyclization_type = result_path.parents[1].name
            with result_path.open() as handle:
                for line_number, line in enumerate(handle, 1):
                    row = json.loads(line)
                    target = row["id"]
                    value = row.get("terminal_rmsd")
                    if value is None:
                        raise ValueError(f"Missing terminal_rmsd in {result_path}:{line_number}")
                    terminal_rmsd = float(value)
                    records.append(
                        {
                            "scale": scale,
                            "target": target,
                            "cyclization_type": cyclization_type,
                            "terminal_rmsd": terminal_rmsd,
                            "success": terminal_rmsd < cutoff,
                        }
                    )
    if not records:
        raise FileNotFoundError(f"No results.jsonl files found under {scan_dir}")
    return records


def summarize(records, group_fields):
    groups = defaultdict(list)
    for row in records:
        groups[tuple(row[field] for field in group_fields)].append(row)

    summary = []
    for key, rows in groups.items():
        rmsds = [row["terminal_rmsd"] for row in rows]
        successes = sum(row["success"] for row in rows)
        ci_low, ci_high = wilson_interval(successes, len(rows))
        summary.append(
            {
                **dict(zip(group_fields, key)),
                "n": len(rows),
                "rmsd_evaluable": len(rmsds),
                "successes": successes,
                "success_rate": successes / len(rows),
                "success_ci95_low": ci_low,
                "success_ci95_high": ci_high,
                "mean_terminal_rmsd": statistics.fmean(rmsds),
                "median_terminal_rmsd": statistics.median(rmsds),
            }
        )
    return sorted(summary, key=lambda row: tuple(row[field] for field in group_fields))


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    scan_dir = args.scan_dir.resolve()
    records = read_records(scan_dir, args.rmsd_cutoff)
    groupings = {
        "summary_overall.csv": ["scale"],
        "summary_by_target.csv": ["scale", "target"],
        "summary_by_type.csv": ["scale", "cyclization_type"],
        "summary_by_target_and_type.csv": ["scale", "target", "cyclization_type"],
    }
    summaries = {name: summarize(records, fields) for name, fields in groupings.items()}

    incomplete = [
        row
        for row in summaries["summary_by_target_and_type.csv"]
        if row["n"] != args.expected_per_cell
    ]
    if incomplete:
        examples = ", ".join(
            f"scale={row['scale']}, target={row['target']}, type={row['cyclization_type']}, n={row['n']}"
            for row in incomplete[:5]
        )
        raise ValueError(f"Found {len(incomplete)} incomplete cells; expected n={args.expected_per_cell}. Examples: {examples}")

    for name, rows in summaries.items():
        write_csv(scan_dir / name, rows)
    print(f"Summarized {len(records)} generations under {scan_dir}")


if __name__ == "__main__":
    main()
