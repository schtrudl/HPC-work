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
    "--size",
    action="append",
    help="Sizes to test (default: 64,128,256,512,1024,2048,4096)",
)
args = parser.parse_args()

binary = "lenia"

if not args.size:
    args.size = [64, 128, 256, 512, 1024, 2048]

shutil.rmtree("result/out", ignore_errors=True)
os.makedirs("result/out", exist_ok=True)

for size in args.size:
    if os.path.exists(binary):
        os.remove(binary)

    if os.path.exists("lenia.gif"):
        os.remove("lenia.gif")

    subprocess.run(
        ["make", binary],
        check=True,
        env={"GENERATE_GIF": "1", "SIZE": str(size), **os.environ},
    )
    cmd = ["mpirun", "-np", "1", f"./{binary}"]
    subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    os.makedirs(f"result/out/{size}", exist_ok=True)
    # exctract first, middle and last frame from lenia.gif
    subprocess.run(
        [
            "ffmpeg",
            "-i",
            "lenia.gif",
            "-vsync",
            "0",
            "-vf",
            "select='eq(n,0)'",
            f"result/out/{size}/1.png",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-i",
            "lenia.gif",
            "-vsync",
            "0",
            "-vf",
            "select='eq(n,49)'",
            f"result/out/{size}/50.png",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-i",
            "lenia.gif",
            "-vsync",
            "0",
            "-vf",
            "select='eq(n,99)'",
            f"result/out/{size}/100.png",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    shutil.move("lenia.gif", f"result/out/{size}/lenia.gif")

if args.bless:
    shutil.copytree("result/out", "result/blessed", dirs_exist_ok=True)
else:
    subprocess.run(
        ["kompari-cli", "report", "result/out", "result/blessed"],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if os.getenv("SLURM_NTASKS") is None:
        subprocess.run(
            ["xdg-open", "report.html"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
print("DONE")
