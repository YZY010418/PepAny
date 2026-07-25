#!/usr/bin/env python3
"""Print terminal-RMSD success rates from generation results JSONL files."""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Print overall and grouped terminal-RMSD success rates from one or "
            "more results.jsonl files."
        )
    )
    parser.add_argument(
        "results_jsonl",
        type=Path,
        nargs="+",
        help="One or more generated candidates/results.jsonl files",
    )
    parser.add_argument("--rmsd-cutoff", type=float, default=1.0)
    return parser.parse_args()


def cyclization_type_from_path(path):
    if path.parent.name == "candidates":
        return path.parent.parent.name
    return path.parent.name


def finite_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def read_records(result_paths, cutoff):
    records = []
    missing_rmsd = 0
    for result_path in result_paths:
        result_path = result_path.resolve()
        if not result_path.is_file():
            raise FileNotFoundError(f"results.jsonl file not found: {result_path}")

        cyclization_type = cyclization_type_from_path(result_path)
        with result_path.open() as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    target = row["id"]
                except (json.JSONDecodeError, KeyError) as exc:
                    raise ValueError(f"Invalid record at {result_path}:{line_number}: {exc}") from exc

                terminal_rmsd = finite_float(row.get("terminal_rmsd"))
                if terminal_rmsd is None:
                    missing_rmsd += 1
                records.append(
                    {
                        "target": target,
                        "cyclization_type": cyclization_type,
                        "rmsd_evaluable": terminal_rmsd is not None,
                        "success": terminal_rmsd is not None and terminal_rmsd < cutoff,
                    }
                )

    if not records:
        raise ValueError("No generation records found in the supplied results.jsonl files")
    if missing_rmsd:
        print(
            f"Warning: {missing_rmsd} records have no finite terminal_rmsd and "
            "are counted as failures.",
            file=sys.stderr,
        )
    return records


def summarize(records, group_fields):
    groups = defaultdict(list)
    for record in records:
        key = tuple(record[field] for field in group_fields)
        groups[key].append(record)

    rows = []
    for key, group in groups.items():
        n = len(group)
        successes = sum(record["success"] for record in group)
        rows.append(
            {
                **dict(zip(group_fields, key)),
                "n": n,
                "rmsd_evaluable": sum(record["rmsd_evaluable"] for record in group),
                "successes": successes,
                "success_rate": successes / n,
            }
        )
    return sorted(rows, key=lambda row: tuple(str(row[field]) for field in group_fields))


def print_table(title, rows, group_fields):
    metric_fields = ("n", "rmsd_evaluable", "successes", "success_rate")
    fields = tuple(group_fields) + metric_fields
    print(f"\n# {title}")
    print("\t".join(fields))
    for row in rows:
        values = []
        for field in fields:
            value = row[field]
            values.append(f"{value:.6f}" if field == "success_rate" else str(value))
        print("\t".join(values))


def main():
    args = parse_args()
    records = read_records(args.results_jsonl, args.rmsd_cutoff)

    print(f"RMSD cutoff: {args.rmsd_cutoff:g}")
    print_table("Overall", summarize(records, []), [])
    print_table("By target", summarize(records, ["target"]), ["target"])
    print_table(
        "By cyclization type",
        summarize(records, ["cyclization_type"]),
        ["cyclization_type"],
    )
    print_table(
        "By target and cyclization type",
        summarize(records, ["target", "cyclization_type"]),
        ["target", "cyclization_type"],
    )


if __name__ == "__main__":
    main()
