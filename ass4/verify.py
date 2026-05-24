#!/usr/bin/env python3

import os
import shutil
import subprocess
import argparse
import shlex
from pathlib import Path
import time
import re

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
parser.add_argument(
    "--binary",
    type=str,
    default="lenia",
    help="Binary to run (default: lenia)",
)
parser.add_argument(
    "-np",
    type=int,
    default=1,
    help="Number of processes to run (default: 1)",
)
args = parser.parse_args()

if not args.size:
    args.size = [64, 128, 256, 512, 1024, 2048]

in_slurm = os.getenv("SLURM_NTASKS") is not None
master = not in_slurm or os.getenv("SLURM_PROCID") == "0"
SLURM_NTASKS = os.getenv("SLURM_NTASKS") or args.np
SLURM_NTASKS = int(SLURM_NTASKS)


def run_ffmpeg(ffmpeg_args, quiet=True):
    stdout = subprocess.DEVNULL if quiet else None
    stderr = subprocess.DEVNULL if quiet else None
    if in_slurm:
        # `module` is a shell command; run it together with ffmpeg in a login shell.
        cmd = "module load FFmpeg && ffmpeg " + " ".join(
            shlex.quote(x) for x in ffmpeg_args
        )
        subprocess.run(["bash", "-lc", cmd], check=True, stdout=stdout, stderr=stderr)
    else:
        subprocess.run(
            ["ffmpeg", *ffmpeg_args], check=True, stdout=stdout, stderr=stderr
        )


if master:
    shutil.rmtree("result/out", ignore_errors=True)
    os.makedirs("result/out", exist_ok=True)

for size in args.size:
    binary = f"{args.binary}.{size}.bin"
    if master:
        if os.path.exists("lenia.gif"):
            os.remove("lenia.gif")

        subprocess.run(
            ["make", args.binary],
            check=True,
            env={"GENERATE_GIF": "1", "SIZE": str(size), **os.environ},
        )
    else:
        while not os.path.exists(binary):
            time.sleep(0.1)
    cmd = ["mpirun", "--oversubscribe", "-mca", "pml", "ob1", "-np", f"{SLURM_NTASKS}", f"./{binary}"]
    if not in_slurm and SLURM_NTASKS != 1:
        print(f"Spawning {SLURM_NTASKS} processes with mpirun.")
        procs = [subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) for _ in range(SLURM_NTASKS)]
        for p in procs:
            if p.wait() != 0:
                raise subprocess.CalledProcessError(p.returncode, cmd)
    else:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    if master:
        os.makedirs(f"result/out/{size}", exist_ok=True)
        # exctract first, middle and last frame from lenia.gif
        run_ffmpeg(
            [
                "-i",
                "lenia.gif",
                "-vsync",
                "0",
                "-vf",
                "select='eq(n,0)'",
                f"result/out/{size}/1.png",
            ]
        )
        run_ffmpeg(
            [
                "-i",
                "lenia.gif",
                "-vsync",
                "0",
                "-vf",
                "select='eq(n,49)'",
                f"result/out/{size}/50.png",
            ],
        )
        run_ffmpeg(
            [
                "-i",
                "lenia.gif",
                "-vsync",
                "0",
                "-vf",
                "select='eq(n,99)'",
                f"result/out/{size}/100.png",
            ],
        )
        shutil.move("lenia.gif", f"result/out/{size}/lenia.gif")

if master:
    subprocess.run(
        ["make", "clean"],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if args.bless:
        shutil.copytree("result/out", "result/blessed", dirs_exist_ok=True)
    else:
        subprocess.run(
            ["kompari-cli", "report", "result/out", "result/blessed"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        if not in_slurm:

            def compare_images_with_threshold(img1, img2, threshold=0.1):
                cmd = [
                    "magick",
                    "compare",
                    "-metric",
                    "RMSE",
                    img1,
                    img2,
                    "null:",
                ]

                # ImageMagick outputs its metrics to stderr, and 'compare' exits with code 1
                # if images don't match exactly. We set check=False to prevent python from crashing.
                result = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                )

                # Capture the output from stderr
                output = result.stderr.strip()

                # Use a regex to grab the normalized value inside the parentheses
                # e.g., from "8.75185 (0.000133545)" it extracts "0.000133545"
                match = re.search(r"\((.*?)\)", output)

                if match:
                    normalized_score = float(match.group(1))
                    print(f"Normalized RMSE Score: {normalized_score}")

                    if normalized_score <= threshold:
                        print("PASS: Difference is within the allowed threshold.")
                        return True
                    else:
                        print("FAIL: Difference exceeds threshold.")
                        return False
                else:
                    # If the images match perfectly (0 error), ImageMagick sometimes outputs just '0'
                    if "0" in output:
                        print("PASS: Images are a perfect pixel match (0 error).")
                        return True

                    print(
                        f"Error running compare or unexpected output structure: {output}"
                    )
                    return False

            subprocess.run(
                ["xdg-open", "report.html"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            exit(
                compare_images_with_threshold(
                    "result/out/64/100.png", "result/blessed/64/100.png", threshold=0.01
                )
            )
    print("DONE")
