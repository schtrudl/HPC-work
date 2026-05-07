#!/usr/bin/env python3
"""Generate a LaTeX timing table from timings_cpu_*.log files.

Usage:
    python3 gen_cpu_table.py <file1.log> [<file2.log> ...]

The first file is treated as the baseline for speedup calculations.
Each file must contain lines like:
    <N>:
      full: <mean> s ± <std> s
"""

import sys
import re
import math
import os


def parse_log(path):
    data = {}
    with open(path) as f:
        content = f.read()
    for m in re.finditer(r"(\d+):\s+full: ([\d.]+) s ± ([\d.]+) s", content):
        n, mean, std = int(m.group(1)), float(m.group(2)), float(m.group(3))
        data[n] = (mean, std)
    return data


def decimal_places(std):
    return max(0, -math.floor(math.log10(std)))


def fmt_time(mean, std):
    d = decimal_places(std)
    return f"{mean:.{d}f}"


def fmt_speedup(t_base, std_base, t_ver, std_ver):
    s = t_base / t_ver
    ds = s * math.sqrt((std_base / t_base) ** 2 + (std_ver / t_ver) ** 2)
    d = decimal_places(ds)
    return f"({s:.{d}f}x)"


def label_from_path(path):
    name = os.path.splitext(os.path.basename(path))[0]
    for prefix in ("timings_cpu_", "timings_"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    return name.replace("_", r"\_")


def main(paths):
    files = [(p, parse_log(p)) for p in paths]
    ns = sorted(set(n for _, d in files for n in d))
    _, baseline = files[0]

    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    col_spec = "|l|" + "c|" * len(ns)
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\hline")
    header_cols = " & ".join(f"\\textbf{{{n}}}" for n in ns)
    lines.append(f" & {header_cols} \\\\")
    lines.append(r"\hline")

    for i, (path, data) in enumerate(files):
        label = label_from_path(path)
        if i == 0:
            cells = " & ".join(fmt_time(data[n][0], data[n][1]) for n in ns)
        else:
            cells = " & ".join(
                fmt_time(data[n][0], data[n][1])
                + " "
                + fmt_speedup(baseline[n][0], baseline[n][1], data[n][0], data[n][1])
                for n in ns
            )
        lines.append(f"\\textbf{{{label}}} & {cells} \\\\")
        lines.append("")

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{Time in seconds and speedup relative to baseline (first file).}"
    )
    lines.append(r"\label{tab:cpu-timings}")
    lines.append(r"\end{table}")

    print("\n".join(lines))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1:])
