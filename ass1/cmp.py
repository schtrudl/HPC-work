#!/usr/bin/env python3

import argparse
import math

HEADER = "\033[95m"
OKBLUE = "\033[94m"
OKCYAN = "\033[96m"
OKGREEN = "\033[92m"
WARNING = "\033[93m"
FAIL = "\033[91m"
ENDC = "\033[0m"
BOLD = "\033[1m"
UNDERLINE = "\033[4m"

parser = argparse.ArgumentParser(
    description="Script for comparing timings produced by run script."
)
parser.add_argument(
    "--baseline",
    type=str,
    help="baseline timings file",
    default="timings.baseline.log",
)
parser.add_argument(
    "--current",
    type=str,
    help="current/new timings file",
    default="timings.new.log",
)
parser.add_argument(
    "--significance-threshold",
    type=float,
    default=7.0,
    help="number of standard deviations for significance (default: 7.0)",
)
parser.add_argument(
    "--speedup",
    action="store_true",
    help="also compute and display speedup (baseline_time / current_time)",
)
parser.add_argument(
    "--bless",
    action="store_true",
    help="bless current timings as new baseline (overwrites baseline file)",
)
args = parser.parse_args()


def read_timings(file: str) -> dict[str, dict[str, tuple[float, float]]]:
    timings: dict[str, dict[str, tuple[float, float]]] = {}
    current_image = None
    with open(file, "r") as f:
        for line in f:
            if not line.strip():
                continue
            # Lines starting with whitespace are timing entries
            if line[0].isspace():
                if current_image is None:
                    continue
                label, rest = line.strip().split(":", 1)
                label = label.strip()
                # Parse "0.2751 s ± 0.0000 s"
                parts = rest.strip().split()
                time = float(parts[0])
                sigma = float(parts[3]) if len(parts) >= 4 else 0.0
                timings[current_image][label] = (time, sigma)
            else:
                # Image name line (e.g., "valve.png:")
                current_image = line.strip().rstrip(":")
                timings[current_image] = {}
    return timings


baseline_timings = read_timings(args.baseline)
current_timings = read_timings(args.current)

print("Comparison of timings baseline ± diff = new (rel diff):")
for image in baseline_timings:
    print(f"{image}:")
    for label in baseline_timings[image]:
        baseline_time, baseline_sigma = baseline_timings[image][label]
        current_time, current_sigma = current_timings[image][label]
        diff = current_time - baseline_time
        # Error propagation: σ_diff = sqrt(σ_baseline² + σ_current²)
        diff_sigma = math.sqrt(baseline_sigma**2 + current_sigma**2)
        diff_percent = (diff / baseline_time) * 100 if baseline_time != 0 else 0
        sig_marker = ""
        # Check significance: is |diff| > 7*σ_diff
        significant = (
            diff_sigma > 0 and abs(diff) > args.significance_threshold * diff_sigma
        )
        sig_marker = f" {BOLD}[SIGNIFICANT]{ENDC}"

        def color(v: float, s: str) -> str:
            if v == 0.0:
                return s
            if significant:
                return f"{BOLD}{OKGREEN if v < 0 else FAIL}{s}{ENDC}"
            else:
                return f"{OKCYAN if v < 0 else WARNING}{s}{ENDC}"

        print(
            f"  {label:>20}: {baseline_time:7.4f} s {color(diff, f'{diff:+.4f}')} s = {current_time:7.4f} s ({color(diff_percent, f'{diff_percent:+6.2f}')}%) {sig_marker if significant else ''}"
            + (
                f" [Speedup: {baseline_time / current_time:.2f}x]"
                if args.speedup
                else ""
            )
        )

if args.bless:
    import shutil

    shutil.copyfile(args.current, args.baseline)
    print(f"Blessed {args.current} as new baseline {args.baseline}")
