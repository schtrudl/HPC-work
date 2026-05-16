#!/usr/bin/env python3

import os
import shutil
import subprocess
import argparse

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
args = parser.parse_args()

binary = "lenia"

if os.path.exists("lenia.gif"):
    os.remove("lenia.gif")

subprocess.run(
    ["make", binary],
    check=True,
    env={"GENERATE_GIF": "1", "SIZE": "64", **os.environ},
)
cmd = [f"./{binary}"]
if args.srun:
    cmd = ["srun"] + cmd
subprocess.run(
    cmd,
    check=True,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
shutil.rmtree("result/out", ignore_errors=True)
os.makedirs("result/out", exist_ok=True)
subprocess.run(
    ["ffmpeg", "-i", "lenia.gif", "-vsync", "0", "result/out/%d.png"],
    check=True,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
shutil.move("lenia.gif", "result/out/lenia.gif")

if args.bless:
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
