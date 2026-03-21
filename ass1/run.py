#!/usr/bin/env python3

import subprocess
from pathlib import Path
import argparse
import re

parser = argparse.ArgumentParser(
    description="This script will run program in selected input n times and extract and compute timing statistics."
)
parser.add_argument(
    "-n",
    type=int,
    default=20,
    help="Number of times to repeat the execution for each image (default: 5)",
)
parser.add_argument(
    "--bless",
    action="store_true",
    help="bless current images as result",
)
parser.add_argument(
    "images",
    nargs="*",
    type=Path,
    help="Images to run on",
    default=[
        Path("test_images/valve.png"),
        Path("test_images/720x480.png"),
        Path("test_images/1024x768.png"),
        Path("test_images/1920x1200.png"),
        Path("test_images/3840x2160.png"),
        Path("test_images/7680x4320.png"),
    ],
)
args = parser.parse_args()

timings: dict[Path, dict[str, list[float]]] = {}
out_folder_name = "result" if args.bless else "out"
for image in args.images:
    timings[image] = {}
    # add subfolder out
    out_image = image.parent / out_folder_name / image.name
    out_image.parent.mkdir(parents=True, exist_ok=True)
    for i in range(args.n):
        output = subprocess.check_output(
            ["./sample", str(image), str(out_image)]
        ).decode("utf-8")
        # parse Time(...): ... s
        regex = r"Time\((.*?)\): ([0-9.]+) s"
        times: dict[str, float] = {}
        for label, time in re.findall(regex, output):
            if label not in times:
                times[label] = 0.0
            times[label] += float(time)
        for label, time in times.items():
            if label not in timings[image]:
                timings[image][label] = []
            timings[image][label].append(time)

for image, timing in timings.items():
    print(f"{image}:")
    for label, times_list in timing.items():
        avg_time = sum(times_list) / len(times_list)
        std_time = (
            sum((t - avg_time) ** 2 for t in times_list) / len(times_list)
        ) ** 0.5
        print(f"  {label}: {avg_time:.4f} s ± {std_time:.4f} s")
