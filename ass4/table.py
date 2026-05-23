#!/usr/bin/env python3
"""Generate a LaTeX timing table from timings_{number}_*.log files.

Usage:
    ./table.py timings_1_baseline.log timings_2_dense.log ... > table.tex

The first file is treated as the baseline for speedup calculations.
Precision is selected from standard deviation (stddev):
- time values use decimal places implied by their own stddev
- speedup values use decimal places implied by propagated stddev
"""

import math
import os
import re
import sys
from typing import Dict, List, Tuple


Entry = Tuple[float, float]  # (mean, stddev)


def parse_log(path: str) -> Dict[int, Entry]:
    data: Dict[int, Entry] = {}
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    pattern = r"(\d+):\s+full:\s+([\d.]+)\s+s\s+±\s+([\d.]+)\s+s"
    for m in re.finditer(pattern, content):
        n = int(m.group(1))
        mean = float(m.group(2))
        std = float(m.group(3))
        data[n] = (mean, std)
    return data


def decimal_places_from_std(std: float) -> int:
    if std <= 0.0:
        return 3
    return max(0, -math.floor(math.log10(std)))


def fmt_value(mean: float, std: float) -> str:
    decimals = decimal_places_from_std(std)
    return f"{mean:.{decimals}f}"


def speedup_and_std(base: Entry, value: Entry) -> Entry:
    t_base, s_base = base
    t_val, s_val = value

    speedup = t_base / t_val
    # Standard uncertainty propagation for a ratio: s = a / b
    speedup_std = speedup * math.sqrt((s_base / t_base) ** 2 + (s_val / t_val) ** 2)
    return speedup, speedup_std


def fmt_speedup(base: Entry, value: Entry) -> str:
    speedup, speedup_std = speedup_and_std(base, value)
    decimals = decimal_places_from_std(speedup_std)
    return f"({speedup:.{decimals}f}x)"


def order_key(path: str) -> Tuple[int, str]:
    name = os.path.splitext(os.path.basename(path))[0]
    m = re.match(r"timings_(\d+)_", name)
    if m:
        return (int(m.group(1)), name)
    return (10**9, name)


def pretty_label(path: str) -> str:
    name = os.path.splitext(os.path.basename(path))[0]
    m = re.match(r"timings_\d+_(.*)", name)
    token = m.group(1) if m else name

    return token.replace("_", " ")


def main(paths: List[str]) -> int:
    if not paths:
        print(__doc__)
        return 1

    ordered_paths = sorted(paths, key=order_key)
    files = [(path, parse_log(path)) for path in ordered_paths]

    baseline_path, baseline_data = files[0]
    if not baseline_data:
        print(f"error: no timing entries found in baseline '{baseline_path}'", file=sys.stderr)
        return 1

    # Keep columns consistent with baseline runs (e.g., 256..4096).
    sizes = sorted(baseline_data.keys())

    out: List[str] = []
    out.append(r"\begin{table}[H]")
    out.append(r"\centering")
    out.append(r"%\resizebox{\linewidth}{!}{")
    out.append(r"\begin{tabular}{|l|" + "c|" * len(sizes) + "}")
    out.append(r"\hline")
    out.append(" & " + " & ".join(f"\\textbf{{{n}}}" for n in sizes) + " \\\\")
    out.append(r"\hline")

    for idx, (path, data) in enumerate(files):
        label = pretty_label(path)
        row_label = f"\\textbf{{{label}}}"

        cells: List[str] = []
        for n in sizes:
            value = data.get(n)
            if value is None:
                cells.append("-")
                continue

            t_str = fmt_value(value[0], value[1])
            if idx == 0:
                cells.append(t_str)
            else:
                base_val = baseline_data.get(n)
                if base_val is None:
                    cells.append(t_str)
                else:
                    cells.append(f"{t_str} {fmt_speedup(base_val, value)}")

        out.append(f"{row_label} ")
        out.append("& " + "\n& ".join(cells) + " \\\\")
        out.append("")

    out.append(r"\hline")
    out.append(r"\end{tabular}")
    out.append(r"%}")
    out.append(
        r"\caption{Time in seconds and speedup relative to baseline (first file). Precision follows stddev.}"
    )
    out.append(r"\end{table}")

    print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
