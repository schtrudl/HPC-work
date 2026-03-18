#!/usr/bin/env python3

import subprocess
from pathlib import Path
import argparse
import re

parser = argparse.ArgumentParser()
parser.add_argument(
    "-n",
    type=int,
    default=5,
    help="Number of times to repeat the execution for each image (default: 5)",
)
# all others are images if none use all images
parser.add_argument(
    "images",
    nargs="*",
    type=Path,
    help="Images to run on",
    default=[
        Path("valve.png"),
        Path("test_images/720x480.png"),
        Path("test_images/1024x768.png"),
        Path("test_images/1920x1200.png"),
        Path("test_images/3840x2160.png"),
        Path("test_images/7680x4320.png"),
    ],
)
args = parser.parse_args()
print(f"Running on {len(args.images)} images, {args.n} times each.")

timings: dict[Path, dict[str, list[float]]] = {}
for image in args.images:
    timings[image] = {}
    out_image = image.with_suffix(".out.png")
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

print("\nAverage timings:")
for image, timing in timings.items():
    print(f"{image}:")
    for label, times_list in timing.items():
        avg_time = sum(times_list) / len(times_list)
        std_time = (
            sum((t - avg_time) ** 2 for t in times_list) / len(times_list)
        ) ** 0.5
        print(f"  {label}: {avg_time:.4f} s ± {std_time:.4f} s")
