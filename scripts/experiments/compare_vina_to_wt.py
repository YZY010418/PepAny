#!/usr/bin/env python3
"""Compare designed-peptide Vina scores with target-specific WT scores."""

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


@dataclass
class Summary:
    matched: int = 0
    valid: int = 0
    invalid_candidate: int = 0
    invalid_wt: int = 0
    better: int = 0
    unmatched: int = 0


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "For every candidate CSV in a directory, print the proportion of "
            "valid affinity scores that outperform the corresponding valid WT score."
        )
    )
    parser.add_argument(
        "wt_csv", type=Path, help="CSV containing one WT row per target"
    )
    parser.add_argument(
        "candidate_dir",
        type=Path,
        help="Directory containing candidate CSV files",
    )
    parser.add_argument(
        "--better",
        choices=("lower", "higher"),
        default="lower",
        help=(
            "score direction: 'lower' means a smaller/more negative score is better; "
            "'higher' means a larger score is better (default: lower)"
        ),
    )
    parser.add_argument(
        "--pattern",
        default="*.csv",
        help="glob used to select candidate CSV files (default: *.csv)",
    )
    parser.add_argument(
        "--denominator",
        choices=("valid", "matched"),
        default="valid",
        help=(
            "ratio denominator: valid affinity scores only, or all candidates "
            "matched to a target with a valid WT score (default: valid)"
        ),
    )
    return parser.parse_args()


def finite_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def read_csv(path):
    try:
        handle = path.open(encoding="utf-8-sig", newline="")
    except OSError as exc:
        raise OSError(f"Cannot open CSV: {path}: {exc}") from exc

    with handle:
        reader = csv.DictReader(handle)
        required = {"input", "affinity"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            columns = ", ".join(sorted(missing))
            raise ValueError(f"{path} is missing required column(s): {columns}")
        yield from reader


def load_wt_scores(path):
    if not path.is_file():
        raise FileNotFoundError(f"WT CSV not found: {path}")

    scores = {}
    for row_number, row in enumerate(read_csv(path), start=2):
        target = (row.get("input") or "").strip()
        score = finite_float(row.get("affinity"))
        if not target:
            raise ValueError(f"Missing WT input at {path}:{row_number}")
        if target in scores:
            raise ValueError(f"Duplicate WT input {target!r} at {path}:{row_number}")
        scores[target] = score

    if not scores:
        raise ValueError(f"No WT scores found in {path}")
    return scores


def target_from_candidate_input(candidate_input, wt_scores):
    # Candidate paths can come from either Unix or Windows-produced CSV files.
    path_parts = PurePosixPath(candidate_input.strip().replace("\\", "/")).parts
    matches = [target for target in wt_scores if target in path_parts]
    if len(matches) > 1:
        raise ValueError(
            f"Candidate input matches more than one WT target: {candidate_input!r}"
        )
    return matches[0] if matches else None


def summarize_csv(path, wt_scores, better):
    summary = Summary()
    for row_number, row in enumerate(read_csv(path), start=2):
        candidate_input = row.get("input") or ""
        try:
            target = target_from_candidate_input(candidate_input, wt_scores)
        except ValueError as exc:
            raise ValueError(f"{path}:{row_number}: {exc}") from exc

        if target is None:
            summary.unmatched += 1
            continue

        wt_score = wt_scores[target]
        if wt_score is None:
            summary.invalid_wt += 1
            continue

        summary.matched += 1
        score = finite_float(row.get("affinity"))
        if score is None:
            summary.invalid_candidate += 1
            continue

        summary.valid += 1
        if (better == "lower" and score < wt_score) or (
            better == "higher" and score > wt_score
        ):
            summary.better += 1

    return summary


def candidate_csv_paths(directory, pattern, wt_csv):
    if not directory.is_dir():
        raise NotADirectoryError(f"Candidate CSV directory not found: {directory}")

    wt_csv = wt_csv.resolve()
    paths = sorted(
        (
            path
            for path in directory.glob(pattern)
            if path.is_file() and path.resolve() != wt_csv
        ),
        key=lambda path: path.name,
    )
    if not paths:
        raise FileNotFoundError(
            f"No candidate CSV files matching {pattern!r} found in {directory}"
        )
    return paths


def print_summary(rows, better, denominator):
    print(f"better={better} (strict comparison; denominator={denominator})")
    print(
        "csv\tmatched\tvalid\tinvalid_candidate\tinvalid_wt\t"
        "unmatched\tbetter\tproportion"
    )
    for path, summary in rows:
        denominator_count = (
            summary.valid if denominator == "valid" else summary.matched
        )
        proportion = (
            summary.better / denominator_count if denominator_count else None
        )
        formatted = f"{proportion:.2%}" if proportion is not None else "NA"
        print(
            f"{path.name}\t{summary.matched}\t{summary.valid}\t"
            f"{summary.invalid_candidate}\t{summary.invalid_wt}\t"
            f"{summary.unmatched}\t{summary.better}\t{formatted}"
        )


def main():
    args = parse_args()
    try:
        wt_scores = load_wt_scores(args.wt_csv)
        paths = candidate_csv_paths(args.candidate_dir, args.pattern, args.wt_csv)
        rows = [
            (path, summarize_csv(path, wt_scores, args.better))
            for path in paths
        ]
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    valid_wt = sum(score is not None for score in wt_scores.values())
    print(
        f"wt_targets={len(wt_scores)} wt_valid={valid_wt} "
        f"wt_invalid={len(wt_scores) - valid_wt}"
    )
    print_summary(rows, args.better, args.denominator)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
