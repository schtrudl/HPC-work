#!/usr/bin/env python3

import os
import subprocess
import argparse
import re

parser = argparse.ArgumentParser(
    description="This script will run program in selected input n times and extract and compute timing statistics."
)
parser.add_argument(
    "-n",
    type=int,
    default=20,
    help="Number of times to repeat the execution for each image (default: 20)",
)
# valid binaries are lenia_cpu and lenia_gpu
parser.add_argument(
    "binary",
    choices=["lenia_cpu", "lenia_gpu", "cpu", "gpu"],
)
parser.add_argument(
    "--size",
    action="append",
    help="Sizes to test (default: 256,512,1024,2048,4096)",
)
parser.add_argument(
    "--srun",
    action="store_true",
    help="Use srun to run the program (for cluster execution)",
)
args = parser.parse_args()

binary = "lenia_cpu" if "cpu" in args.binary else "lenia_gpu"
if not args.size:
    args.size = [256, 512, 1024, 2048, 4096]

timings: dict[int, dict[str, list[float]]] = {}
for size in args.size:
    timings[size] = {}
    subprocess.run(
        ["make", binary],
        check=True,
        env={"N": str(size), **os.environ},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for i in range(args.n):
        cmd = [f"./{binary}"]
        if args.srun:
            cmd = ["srun"] + cmd
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
