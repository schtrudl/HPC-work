#!/usr/bin/env python3

import os
import subprocess
import argparse
import re
from time import sleep

parser = argparse.ArgumentParser(
    description="This script will run program in selected input n times and extract and compute timing statistics."
)
parser.add_argument(
    "-n",
    type=int,
    default=20,
    help="Number of times to repeat the execution for each image (default: 20)",
)
parser.add_argument(
    "--size",
    action="append",
    help="Sizes to test (default: 128,256,512,1024,2048,4096)",
)
parser.add_argument(
    "--binary",
    type=str,
    default="lenia",
    help="Binary to run (default: lenia)",
)
args = parser.parse_args()

if not args.size:
    args.size = [128, 256, 512, 1024, 2048, 4096]

timings: dict[int, dict[str, list[float]]] = {}
# get slurm ntasks
SLURM_NTASKS = os.getenv("SLURM_NTASKS") or "1"
in_slurm = os.getenv("SLURM_NTASKS") is not None
master = not in_slurm or os.getenv("SLURM_PROCID") == "0"
for size in args.size:
    binary = f"{args.binary}.{size}.bin"
    timings[size] = {}
    if master:
        subprocess.run(
            ["make", args.binary],
            check=True,
            env={"SIZE": str(size), **os.environ},
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        while not os.path.exists(binary):
            sleep(0.1)
    for i in range(args.n):
        cmd = [
            "mpirun",
            "-mca",
            "pml",
            "ob1",
            "-np",
            f"{SLURM_NTASKS}",
            f"./{binary}",
        ]
        output = subprocess.check_output(cmd).decode("utf-8")
        # parse Time(...): ... s
        regex = r"Time\((.*?)\): ([0-9.]+) s"
        times: dict[str, float] = {}
        for label, time in re.findall(regex, output):
            if label not in times:
                times[label] = 0.0
            times[label] += float(time)
        for label, time in times.items():
            if label not in timings[size]:
                timings[size][label] = []
            timings[size][label].append(time)

for size, timing in timings.items():
    print(f"{size}:")
    for label, times_list in timing.items():
        avg_time = sum(times_list) / len(times_list)
        std_time = (
            sum((t - avg_time) ** 2 for t in times_list) / len(times_list)
        ) ** 0.5
        print(f"  {label}: {avg_time:.4f} s ± {std_time:.4f} s")

if master:
    subprocess.run(
        ["make", "clean"],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
