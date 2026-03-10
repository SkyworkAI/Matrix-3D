"""
VBench batch generation for Matrix-3D.
Runs the two-step pipeline (i2p panorama → video) for all scenery/indoor prompts.

Usage (from Matrix-3D root):
    python code/vbench_batch.py [--output_dir output/vbench/videos] [--num_samples 5] [--seed 0] [--resolution 720]
"""
import argparse
import csv
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time

import psutil
import torch

_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR     = os.path.dirname(_SCRIPT_DIR)
_VBENCH_DATA  = os.path.join(_ROOT_DIR, "..", "VBench", "vbench2_beta_i2v", "vbench2_beta_i2v", "data")
_DEFAULT_JSON = os.path.join(_VBENCH_DATA, "i2v-bench-info.json")
_DEFAULT_CROP = os.path.join(_VBENCH_DATA, "crop", "1-1")
_CATEGORIES   = {"scenery", "indoor"}


def _safe(text):
    return re.sub(r'[<>:"/\\|?*]', "_", text)[:150]


def _fmt_duration(secs):
    h, m, s = int(secs // 3600), int(secs % 3600 // 60), int(secs % 60)
    return f"{h:02d}h{m:02d}m{s:02d}s"


def _sys_stats():
    vm = psutil.virtual_memory()
    ram_used = vm.used / 1024**3
    ram_total = vm.total / 1024**3
    if torch.cuda.is_available():
        gpu_used  = torch.cuda.memory_allocated() / 1024**3
        gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    else:
        gpu_used = gpu_total = 0.0
    return ram_used, ram_total, gpu_used, gpu_total


def run(cmd, cwd=None):
    result = subprocess.run(cmd, cwd=cwd or _ROOT_DIR)
    return result.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir",  default=os.path.join(_ROOT_DIR, "output", "vbench", "videos"))
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--seed",        type=int, default=0)
    parser.add_argument("--resolution",  type=int, default=720)
    parser.add_argument("--vbench_json", default=_DEFAULT_JSON)
    parser.add_argument("--crop_dir",    default=_DEFAULT_CROP)
    args = parser.parse_args()

    info_json = os.path.abspath(args.vbench_json)
    crop_dir  = os.path.abspath(args.crop_dir)
    out_dir   = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    stats_path   = os.path.join(os.path.dirname(out_dir), "vbench_gen_stats.csv")
    stats_is_new = not os.path.exists(stats_path)
    stats_f = open(stats_path, "a", newline="", encoding="utf-8")
    stats_w = csv.writer(stats_f)
    if stats_is_new:
        stats_w.writerow(["timestamp", "task_idx", "prompt", "sample_idx", "seed", "duration_s",
                          "video_count", "total_elapsed_s", "avg_s_per_video",
                          "ram_used_gb", "ram_total_gb", "gpu_used_gb", "gpu_total_gb",
                          "out_path", "status"])

    ram_total_gb = psutil.virtual_memory().total / 1024**3
    gpu_total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0.0

    with open(info_json, encoding="utf-8") as f:
        entries = json.load(f)

    seen, prompts = set(), []
    for e in entries:
        name = e["file_name"]
        if name in seen:
            continue
        if e.get("type") not in _CATEGORIES:
            continue
        seen.add(name)
        caption = e.get("caption", os.path.splitext(name)[0])
        prompts.append((name, caption))

    total = len(prompts) * args.num_samples
    print(f"{'='*70}")
    print(f"[vbench] Matrix-3D VBench batch")
    print(f"[vbench] {len(prompts)} prompts x {args.num_samples} samples = {total} videos")
    print(f"[vbench] categories: {sorted(_CATEGORIES)}  resolution: {args.resolution}p")
    print(f"[vbench] output → {out_dir}")
    print(f"[vbench] stats  → {stats_path}")
    print(f"{'='*70}")

    done = skipped = generated = errors = 0
    ok_total_s = 0.0
    t_start = time.time()

    for task_idx, (image_name, caption) in enumerate(prompts):
        image_path = os.path.join(crop_dir, image_name)
        if not os.path.isfile(image_path):
            print(f"[vbench] SKIP: image not found — {image_path}")
            continue

        for sample_idx in range(args.num_samples):
            seed     = random.randint(0, 2**31 - 1)
            out_path = os.path.join(out_dir, f"{_safe(caption)}-{sample_idx}-{seed}.mp4")

            # ── header ──────────────────────────────────────────────────────
            elapsed   = time.time() - t_start
            pct       = 100 * done / total if total else 0
            eta_str   = ""
            avg_str   = ""
            if generated > 0:
                avg_s   = ok_total_s / generated
                remaining = (total - done) * avg_s
                eta_str = f"  ETA {_fmt_duration(remaining)}"
                avg_str = f"  avg {avg_s/60:.1f} min/video"
            print(f"\n{'─'*70}")
            print(f"[vbench] [{done+1}/{total}  {pct:.0f}%{eta_str}{avg_str}]  elapsed {_fmt_duration(elapsed)}")
            print(f"[vbench] prompt {task_idx+1}/{len(prompts)}  sample {sample_idx+1}/{args.num_samples}  seed {seed}")
            print(f"[vbench] {caption[:70]}")

            if os.path.exists(out_path):
                print(f"[vbench] → SKIP (already exists)")
                skipped += 1
                done += 1
                stats_w.writerow([time.strftime("%Y-%m-%dT%H:%M:%S"), task_idx, caption, sample_idx, seed,
                                  "", generated, f"{elapsed:.1f}", "",
                                  "", ram_total_gb, "", gpu_total_gb, out_path, "skipped"])
                stats_f.flush()
                continue

            work_dir = os.path.join(_ROOT_DIR, "output", "vbench", "_work", f"{task_idx}_{sample_idx}")
            os.makedirs(work_dir, exist_ok=True)

            try:
                st = time.time()

                # Step 1: image → panorama
                print(f"[vbench] step 1/2  panorama generation …")
                t1 = time.time()
                rc = run([
                    sys.executable, "code/panoramic_image_generation.py",
                    "--mode=i2p",
                    f"--input_image_path={image_path}",
                    f"--output_path={work_dir}",
                    f"--seed={seed}",
                ])
                if rc != 0:
                    raise RuntimeError(f"panoramic_image_generation.py exited with code {rc}")
                print(f"[vbench] step 1/2  done  ({time.time()-t1:.0f}s)")

                # Step 2: panorama → video
                print(f"[vbench] step 2/2  video generation …")
                t2 = time.time()
                rc = run([
                    sys.executable, "code/panoramic_image_to_video.py",
                    f"--inout_dir={work_dir}",
                    f"--resolution={args.resolution}",
                    f"--seed={seed}",
                ])
                if rc != 0:
                    raise RuntimeError(f"panoramic_image_to_video.py exited with code {rc}")
                print(f"[vbench] step 2/2  done  ({time.time()-t2:.0f}s)")

                generated_mp4 = os.path.join(work_dir, "generated", "generated.mp4")
                if not os.path.exists(generated_mp4):
                    raise RuntimeError(f"output video not found: {generated_mp4}")

                shutil.copy2(generated_mp4, out_path)
                shutil.rmtree(work_dir, ignore_errors=True)

                ed = time.time()
                duration = ed - st
                ok_total_s += duration
                generated += 1

                ram_used, _, gpu_used, _ = _sys_stats()
                total_elapsed = ed - t_start
                avg_s_per = ok_total_s / generated

                print(f"[vbench] ✓ saved  {os.path.basename(out_path)}")
                print(f"[vbench]   duration {_fmt_duration(duration)}  |  avg {avg_s_per/60:.1f} min/video")
                print(f"[vbench]   RAM {ram_used:.1f}/{ram_total_gb:.0f} GB  |  GPU {gpu_used:.1f}/{gpu_total_gb:.0f} GB")

                stats_w.writerow([time.strftime("%Y-%m-%dT%H:%M:%S"), task_idx, caption, sample_idx, seed,
                                  f"{duration:.1f}", generated, f"{total_elapsed:.1f}", f"{avg_s_per:.1f}",
                                  f"{ram_used:.2f}", f"{ram_total_gb:.2f}", f"{gpu_used:.2f}", f"{gpu_total_gb:.2f}",
                                  out_path, "ok"])
                stats_f.flush()

            except Exception as exc:
                print(f"[vbench] ✗ ERROR: {exc}")
                ram_used, _, gpu_used, _ = _sys_stats()
                stats_w.writerow([time.strftime("%Y-%m-%dT%H:%M:%S"), task_idx, caption, sample_idx, seed,
                                  "", generated, f"{time.time()-t_start:.1f}", "",
                                  f"{ram_used:.2f}", f"{ram_total_gb:.2f}", f"{gpu_used:.2f}", f"{gpu_total_gb:.2f}",
                                  out_path, "error"])
                stats_f.flush()
                errors += 1

            done += 1

    elapsed_total = time.time() - t_start
    stats_f.close()
    print(f"\n{'='*70}")
    print(f"[vbench] DONE  generated={generated}  skipped={skipped}  errors={errors}")
    print(f"[vbench] total elapsed: {_fmt_duration(elapsed_total)}")
    if generated:
        print(f"[vbench] avg per video: {ok_total_s/generated/60:.1f} min")
    print(f"[vbench] videos → {out_dir}")
    print(f"[vbench] stats  → {stats_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
