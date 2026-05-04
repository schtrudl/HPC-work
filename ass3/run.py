#!/usr/bin/env python3

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
    choices=["lj_cpu", "lj_gpu", "lj_cpu2", "cpu", "gpu", "cpu2"],
)
parser.add_argument(
    "--particles",
    action="append",
    help="Sizes to test (default: 1000, 2000, 4000, 8000)",
)
parser.add_argument(
    "--steps",
    type=int,
    default=5000,
    help="Number of steps to simulate (default: 5000)",
)
parser.add_argument(
    "--srun",
    action="store_true",
    help="Use srun to run the program (for cluster execution)",
)
args = parser.parse_args()

binary = args.binary if args.binary.startswith("lj_") else f"lj_{args.binary}"
if not args.particles:
    args.particles = [1000, 2000, 4000, 8000]

subprocess.run(
    ["make", binary],
    check=True,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)

timings: dict[int, dict[str, list[float]]] = {}
for particles in args.particles:
    timings[particles] = {}
    for i in range(args.n):
        cmd = [f"./{binary}", f"{particles}", f"{args.steps}"]
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
            if label not in timings[particles]:
                timings[particles][label] = []
            timings[particles][label].append(time)

for particles, timing in timings.items():
    print(f"{particles}:")
    for label, times_list in timing.items():
        avg_time = sum(times_list) / len(times_list)
        std_time = (
            sum((t - avg_time) ** 2 for t in times_list) / len(times_list)
        ) ** 0.5
        print(f"  {label}: {avg_time:.4f} s ± {std_time:.4f} s")
