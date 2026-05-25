#!/usr/bin/env python3
"""Generate LaTeX tables for MPI timings with halos suffix.

Usage:
    ./table_nodes_mpc.py timings_mpc_*_*_*.log > table_mpc.tex

The script expects filenames where the last three underscore-separated fields are:
    ..._<nodes>_<process_per_node>_<halos>.log

Examples:
    timings_mpc_1_32_1.log
    timings_mpc_2_64_3.log
"""

import math
import os
import re
import sys
from typing import Dict, List, Tuple

Entry = Tuple[float, float]  # (mean, stddev)
ParsedFile = Tuple[int, int, int, Dict[int, Entry]]  # (nodes, ppn, halos, size->entry)


def parse_log(path: str) -> Dict[int, Entry]:
    data: Dict[int, Entry] = {}
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    pattern = r"(\d+):\s+full:\s+([\d.]+)\s+s\s+±\s+([\d.]+)\s+s"
    for m in re.finditer(pattern, content):
        size = int(m.group(1))
        mean = float(m.group(2))
        std = float(m.group(3))
        data[size] = (mean, std)
    return data


def parse_name(path: str) -> Tuple[int, int, int]:
    stem = os.path.splitext(os.path.basename(path))[0]
    parts = stem.split("_")
    if len(parts) < 4:
        raise ValueError(f"cannot parse nodes/ppn/halos from '{path}'")

    try:
        nodes = int(parts[-3])
        ppn = int(parts[-2])
        halos = int(parts[-1])
    except ValueError as exc:
        raise ValueError(f"cannot parse nodes/ppn/halos from '{path}'") from exc

    return nodes, ppn, halos


def decimal_places_from_std(std: float) -> int:
    if std <= 0.0:
        return 3
    return max(0, -math.floor(math.log10(std)))


def fmt_time(mean: float, std: float) -> str:
    d = decimal_places_from_std(std)
    return f"{mean:.{d}f}"


def process_label(nodes: int, ppn: int) -> str:
    if nodes == 2:
        return f"{ppn}+{ppn}"
    return str(ppn)


def build_tabular(nodes: int, rows: List[ParsedFile]) -> List[str]:
    ppn_values = sorted({ppn for _, ppn, _, _ in rows})
    data_map = {ppn: data for _, ppn, _, data in rows}

    # Keep sizes in ascending order across available files.
    sizes = sorted({size for _, _, _, data in rows for size in data})

    out: List[str] = []
    out.append(r"\begin{tabular}{l|" + "r" * len(ppn_values) + "}")
    out.append(r"\hline")
    out.append(
        r"\textbf{Size} & "
        + rf"\multicolumn{{{len(ppn_values)}}}{{c}}{{\textbf{{Processes}}}}"
        + " \\\\"
    )
    out.append(rf"\cline{{2-{len(ppn_values) + 1}}}")

    header = " & ".join(f"\\textbf{{{process_label(nodes, p)}}}" for p in ppn_values)
    out.append(" & " + header + r" \\")
    out.append(r"\hline")

    for size in sizes:
        row_entries: List[Entry] = []
        for ppn in ppn_values:
            maybe_entry = data_map.get(ppn, {}).get(size)
            if maybe_entry is not None:
                row_entries.append(maybe_entry)

        best_mean = min(entry[0] for entry in row_entries) if row_entries else None

        cells: List[str] = []
        for ppn in ppn_values:
            entry = data_map.get(ppn, {}).get(size)
            if entry is None:
                cells.append("-")
                continue

            value = fmt_time(entry[0], entry[1])
            if best_mean is not None and math.isclose(
                entry[0], best_mean, rel_tol=1e-12, abs_tol=1e-12
            ):
                value = f"\\textbf{{{value}}}"
            cells.append(value)
        out.append(f"{size} & " + " & ".join(cells) + r" \\")

    out.append(r"\hline")
    out.append(r"\end{tabular}")

    return out


def build_side_by_side_table(grouped: Dict[int, List[ParsedFile]], halos: int) -> str:
    left_nodes = 1
    right_nodes = 2

    out: List[str] = []
    out.append(r"\begin{table}[H]")
    out.append(r"\centering")

    if left_nodes in grouped:
        out.append(r"\begin{minipage}[t]{0.48\textwidth}")
        out.append(r"\centering")
        out.append(r"\textbf{1 node}\\")
        out.extend(build_tabular(left_nodes, grouped[left_nodes]))
        out.append(r"\end{minipage}")

    if left_nodes in grouped and right_nodes in grouped:
        out.append(r"\hfill")

    if right_nodes in grouped:
        out.append(r"\begin{minipage}[t]{0.48\textwidth}")
        out.append(r"\centering")
        out.append(r"\textbf{2 nodes}\\")
        out.extend(build_tabular(right_nodes, grouped[right_nodes]))
        out.append(r"\end{minipage}")

    out.append(rf"\caption{{Execution time (seconds), halos = {halos}.}}")
    out.append(r"\end{table}")

    return "\n".join(out)


def build_tables_by_halos(parsed: List[ParsedFile]) -> str:
    by_halos: Dict[int, List[ParsedFile]] = {}
    for item in parsed:
        by_halos.setdefault(item[2], []).append(item)

    tables: List[str] = []
    for halos in sorted(by_halos):
        by_nodes: Dict[int, List[ParsedFile]] = {}
        for row in by_halos[halos]:
            by_nodes.setdefault(row[0], []).append(row)
        tables.append(build_side_by_side_table(by_nodes, halos))

    return "\n\n".join(tables)


def main(paths: List[str]) -> int:
    if not paths:
        print(__doc__)
        return 1

    parsed: List[ParsedFile] = []
    for path in sorted(paths):
        try:
            nodes, ppn, halos = parse_name(path)
        except ValueError:
            continue
        data = parse_log(path)
        if data:
            parsed.append((nodes, ppn, halos, data))

    if not parsed:
        print("error: no parsable timing files provided", file=sys.stderr)
        return 1

    print(build_tables_by_halos(parsed))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
