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
# valid binaries are lenia_cpu and lenia_gpu
parser.add_argument(
    "binary",
    choices=["lj_cpu", "lj_gpu", "lj_cpu2", "cpu", "gpu", "cpu2", "lj_gpu2", "gpu2"],
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

binary = args.binary if args.binary.startswith("lj_") else f"lj_{args.binary}"

if not args.dry_run:
    subprocess.run(
        ["make", binary],
        check=True,
        env={"GENERATE_GIF": "1", **os.environ},
    )

    shutil.rmtree("result/out", ignore_errors=True)
    configs = [(100, 100)]
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
        shutil.move("./simulation.gif", f"{out}/simulation.gif")
        subprocess.run(
            ["ffmpeg", "-i", f"{out}/simulation.gif", "-vsync", "0", f"{out}/%d.png"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

if args.bless and not args.dry_run:
    shutil.copytree("result/out", "result/blessed", dirs_exist_ok=True)
else:
    subprocess.run(
        ["kompari-cli", "report", "result/out", "result/blessed"],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if not args.srun:
        subprocess.run(
            ["xdg-open", "report.html"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
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
