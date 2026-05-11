#!/usr/bin/env python3

import os
import shutil
import subprocess
import argparse
import re

parser = argparse.ArgumentParser(description="This script will check gif correctness.")
parser.add_argument(
    "--bless",
    action="store_true",
    help="bless current images as result",
)
parser.add_argument(
    "--srun",
    action="store_true",
    help="Use srun to run the program (for cluster execution)",
)
parser.add_argument(
    "--full",
    action="store_true",
    help="Verify all combinations",
)
parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Just cmp and exit",
)
args = parser.parse_args()
binary = "lj.out"

if not args.dry_run:
    try:
        os.remove(binary)
    except OSError:
        pass
    subprocess.run(
        ["make", binary],
        check=True,
    )
    shutil.rmtree("result/out", ignore_errors=True)
    configs = [(100, 100), (1000, 100)]
    if args.full:
        configs += [
            (1000, 5000),
            (2000, 5000),
            (4000, 5000),
            (8000, 5000),
        ]

    for particles, steps in configs:
        cmd = [f"./{binary}", f"{particles}", f"{steps}"]
        if args.srun:
            cmd = ["srun"] + cmd
        output = subprocess.check_output(
            cmd,
        ).decode("utf-8")
        # keep only lines starting with Final
        result = (
            "\n".join(line for line in output.splitlines() if line.startswith("Final"))
            + "\n"
        )

        out = f"result/out/{particles}_{steps}"
        os.makedirs(out, exist_ok=True)
        with open(f"{out}/result.txt", "w") as f:
            f.write(result)

if args.bless and not args.dry_run:
    shutil.copytree("result/out", "result/blessed", dirs_exist_ok=True)
else:
    import difflib

    # diff all result.txt files
    for out in os.listdir("result/out"):
        with open(f"result/out/{out}/result.txt") as f:
            out_result = f.read()
        with open(f"result/blessed/{out}/result.txt") as f:
            blessed_result = f.read()
        diff = "\n".join(
            difflib.unified_diff(
                blessed_result.splitlines(),
                out_result.splitlines(),
                fromfile=f"blessed/{out}/result.txt",
                tofile=f"out/{out}/result.txt",
                lineterm="",
            )
        )
        if diff:
            print(diff)
        else:
            print(f"{out} is OK")
