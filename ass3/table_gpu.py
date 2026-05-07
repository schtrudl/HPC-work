#!/usr/bin/env python3
"""Generate LaTeX step-timing tables for GPU block size comparison.

Usage:
    ./table_gpu.py timings_gpu_b32.log timings_gpu_b64.log ...

Produces two tables (N=1000 and N=8000), rows=steps, columns=block sizes.
"""

import sys, re, math, os

STEPS = ["first_update", "force", "pe", "second_update", "ke"]
STEP_LABELS = {
    "first_update": r"first\_update",
    "force": "force",
    "pe": "PE",
    "second_update": r"second\_update",
    "ke": "KE",
}


def parse_log(path):
    """Returns {n: {step: (mean, std)}}"""
    data = {}
    with open(path) as f:
        content = f.read()
    for section in re.split(r"\n(?=\d+:)", content.strip()):
        m = re.match(r"^(\d+):", section)
        if not m:
            continue
        n = int(m.group(1))
        data[n] = {}
        for step in STEPS + ["full"]:
            sm = re.search(rf"  {step}: ([\d.]+) s ± ([\d.]+) s", section)
            if sm:
                data[n][step] = (float(sm.group(1)), float(sm.group(2)))
    return data


def decimal_places(std):
    if std == 0:
        return 4
    return max(0, -math.floor(math.log10(std)))


def fmt(mean, std):
    d = decimal_places(std)
    return f"{mean:.{d}f}"


def col_label(path):
    name = os.path.splitext(os.path.basename(path))[0]
    # strip timings_gpu_ prefix
    for prefix in ("timings_gpu_", "timings_"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    return name.replace("_", r"\_")


def make_table(files, n):
    cols = [col_label(p) for p, _ in files]
    col_spec = "|l|" + "c|" * len(cols)
    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\hline")
    header = " & ".join(f"\\textbf{{{c}}}" for c in cols)
    lines.append(f"\\textbf{{Step}} & {header} \\\\")
    lines.append(r"\hline")
    for step in STEPS:
        label = STEP_LABELS[step]
        values = []
        for _, data in files:
            if n in data and step in data[n]:
                values.append(data[n][step])
            else:
                values.append(None)
        formatted = [fmt(v[0], v[1]) if v is not None else None for v in values]
        present = [s for s in formatted if s is not None]
        best_str = min(present, key=lambda s: float(s)) if present else None
        cells = []
        for s in formatted:
            if s is None:
                cells.append("--")
            elif s == best_str:
                cells.append(f"\\textbf{{{s}}}")
            else:
                cells.append(s)
        lines.append(f"\\textbf{{{label}}} & {' & '.join(cells)} \\\\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(f"\\caption{{GPU step timings (seconds) for $N={n}$ particles.}}")
    lines.append(f"\\label{{tab:gpu-steps-{n}}}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main(paths):
    files = [(p, parse_log(p)) for p in paths]
    tables = [make_table(files, n) for n in [1000, 8000]]
    print(r"\begin{table}[H]")
    print(r"\centering")
    for i, (n, tbl) in enumerate(zip([1000, 8000], tables)):
        # strip outer \begin{table}...\end{table} wrapper, keep inner content
        inner = re.sub(r"\\begin\{table\}\[H\]\n\\centering\n", "", tbl)
        inner = re.sub(r"\n\\end\{table\}$", "", inner)
        print(r"\begin{minipage}[t]{0.48\textwidth}")
        print(r"\centering")
        print(inner)
        print(r"\end{minipage}")
        if i == 0:
            print(r"\hfill")
    print(r"\end{table}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1:])
