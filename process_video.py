"""
Detect integer lines in every frame of a RHEED video and track them over time.

Each frame goes through the same preprocessing as the training data
(Pillow "L" grayscale -> 3x3 median -> subtract a sigma-30 Gaussian
background -> stretch so the 99.7th percentile is 255), then through the
U-Net ensemble and the row-profile line detector.

Detections are then linked across frames into tracks by y position. A real
integer line stays at nearly the same height from frame to frame, while false
positives tend to flicker, so tracks shorter than --min_track_frames are
dropped ("kept" = False in the CSV).

Input is either a video file (needs OpenCV: pip3 install opencv-python-headless)
or a folder of frame images sorted by name.

Usage:
    python3 process_video.py --video "~/Downloads/AlGaSb last 2 min_LOSSLESS.mp4" \
        --output_dir results/algasb_last2min --every 5

Outputs in --output_dir:
    lines.csv          one row per detection: frame, time_s, y, x0, x1, score,
                       track, kept
    tracks.json        per-track summary (start/end time, mean y, length)
    timeline.png       line position vs time, lines per frame, line spacing
    overlays/          a sample of frames with detections drawn (--overlay_every)
"""

import argparse
import csv
import glob
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
import torch

from postprocess import profile_lines
from train_unet import UNetResNet18

DEFAULT_CHECKPOINTS = ",".join(
    f"models/ensemble_60_round2/seed{s}.pt" for s in (1337, 7, 2024))


# ---------------------------------------------------------------- input frames

def iterate_video(path, every, start_s, end_s):
    try:
        import cv2
    except ImportError as error:
        raise SystemExit("Reading a video needs OpenCV: pip3 install opencv-python-headless "
                         "(or pass --frames_dir with extracted frames)") from error
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise SystemExit(f"Could not open video {path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    first = int(round(start_s * fps))
    last = total - 1 if end_s is None else min(total - 1, int(round(end_s * fps)))
    if first:
        capture.set(cv2.CAP_PROP_POS_FRAMES, first)
    index = first
    while index <= last:
        ok, bgr = capture.read()
        if not ok:
            break
        if (index - first) % every == 0:
            yield index, index / fps, bgr[..., ::-1]
        index += 1
    capture.release()


def iterate_folder(path, every, fps):
    names = sorted(p for p in glob.glob(os.path.join(path, "*"))
                   if p.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")))
    if not names:
        raise SystemExit(f"No images found in {path}")
    for index, name in enumerate(names[::every]):
        frame_index = index * every
        yield frame_index, frame_index / fps, np.asarray(Image.open(name).convert("RGB"))


def count_frames(args):
    if args.frames_dir:
        n = len(glob.glob(os.path.join(args.frames_dir, "*")))
        return (n + args.every - 1) // args.every
    try:
        import cv2
        capture = cv2.VideoCapture(args.video)
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        first = int(round(args.start_s * fps))
        last = total - 1 if args.end_s is None else min(total - 1, int(round(args.end_s * fps)))
        return max(0, (last - first) // args.every + 1)
    except ImportError:
        return None


# -------------------------------------------------------------- preprocessing

def remove_background(rgb, sigma=30.0, percentile=99.7, median_size=3):
    """
    Raw RGB frame -> uint8 background-removed grayscale, identical to the
    training preprocessing (preprocess_background.py): Pillow "L" grayscale,
    3x3 median denoise, subtract a Gaussian background, stretch the 99.7th
    percentile to 255.
    """
    gray = np.asarray(Image.fromarray(np.ascontiguousarray(rgb)).convert("L"))
    denoised = ndimage.median_filter(gray.astype(np.float32), size=median_size)
    foreground = np.clip(denoised - ndimage.gaussian_filter(denoised, sigma=sigma), 0, None)
    upper = float(np.percentile(foreground, percentile))
    if upper <= 0:
        return np.zeros(gray.shape, dtype=np.uint8)
    return np.round(np.clip(foreground / upper, 0, 1) * 255).astype(np.uint8)


def to_model_input(bg_removed, size):
    height, width = size
    image = Image.fromarray(bg_removed).resize((width, height), Image.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


# ------------------------------------------------------------------- tracking

def track_lines(detections, tolerance, max_gap):
    """
    Link per-frame detections into tracks. ``detections`` is a list of
    (step, [line, ...]) in frame order, where step counts processed frames.
    Each line dict gets a "track" id. A track matches a detection when the
    y difference to the track's last position is within ``tolerance``; a
    track that goes unmatched for more than ``max_gap`` processed frames ends.
    """
    active = {}  # track id -> (last y, last step)
    next_id = 0
    for step, lines in detections:
        active = {t: v for t, v in active.items() if step - v[1] <= max_gap + 1}
        ids = list(active)
        matched = set()
        if ids and lines:
            costs = np.abs(np.array([active[t][0] for t in ids])[:, None] -
                           np.array([line["y"] for line in lines])[None, :])
            rows, cols = linear_sum_assignment(np.where(costs <= tolerance, costs, 1e6))
            for r, c in zip(rows, cols):
                if costs[r, c] <= tolerance:
                    lines[c]["track"] = ids[r]
                    active[ids[r]] = (lines[c]["y"], step)
                    matched.add(c)
        for c, line in enumerate(lines):
            if c not in matched:
                line["track"] = next_id
                active[next_id] = (line["y"], step)
                next_id += 1


# ----------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", help="Video file (needs OpenCV)")
    source.add_argument("--frames_dir", help="Folder of raw frames, sorted by name")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINTS,
                        help="Comma-separated checkpoints; predictions are averaged")
    parser.add_argument("--every", type=int, default=1, help="Process every Nth frame")
    parser.add_argument("--start_s", type=float, default=0.0)
    parser.add_argument("--end_s", type=float, default=None)
    parser.add_argument("--fps", type=float, default=30.0,
                        help="Frame rate for --frames_dir (video files report their own)")
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--track_tolerance_px", type=float, default=15.0,
                        help="Max y jump (original px) between linked detections")
    parser.add_argument("--max_gap", type=int, default=2,
                        help="Processed frames a track may be missing before it ends")
    parser.add_argument("--min_track_frames", type=int, default=5,
                        help="Tracks with fewer detections are treated as flicker")
    parser.add_argument("--overlay_every", type=int, default=100,
                        help="Save an overlay image every N processed frames (0 = none)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    args = parser.parse_args()
    if args.video:
        args.video = os.path.expanduser(args.video)
    if args.every < 1:
        raise SystemExit("--every must be >= 1")

    if args.device != "auto":
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    models, size = [], None
    for path in args.checkpoint.split(","):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if size is not None and tuple(checkpoint["size"]) != size:
            raise ValueError("All checkpoints must use the same input size")
        size = tuple(checkpoint["size"])
        model = UNetResNet18(pretrained=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        models.append(model.to(device).eval())

    os.makedirs(args.output_dir, exist_ok=True)
    overlay_dir = os.path.join(args.output_dir, "overlays")
    if args.overlay_every:
        os.makedirs(overlay_dir, exist_ok=True)

    frames = (iterate_video(args.video, args.every, args.start_s, args.end_s) if args.video
              else iterate_folder(args.frames_dir, args.every, args.fps))
    expected = count_frames(args)
    print(f"Device {device}, {len(models)} model(s), "
          f"{expected if expected is not None else '?'} frame(s) to process", flush=True)

    records = []      # (frame_index, time_s, lines)
    overlay_jobs = {}  # step -> (frame_index, bg_removed)
    batch = []
    started = time.time()

    def flush(batch):
        inputs = torch.from_numpy(np.stack([b[3] for b in batch]))[:, None].to(device)
        with torch.no_grad():
            heatmaps = torch.stack([m(inputs) for m in models]).mean(0)[:, 0].cpu().numpy()
        for (frame_index, time_s, original_size, _), heatmap in zip(batch, heatmaps):
            records.append((frame_index, time_s,
                            profile_lines(heatmap, original_size, threshold=args.threshold)))

    for step, (frame_index, time_s, rgb) in enumerate(frames):
        bg_removed = remove_background(rgb)
        if args.overlay_every and step % args.overlay_every == 0:
            overlay_jobs[step] = (frame_index, bg_removed)
        batch.append((frame_index, time_s, bg_removed.shape, to_model_input(bg_removed, size)))
        if len(batch) == args.batch_size:
            flush(batch)
            batch = []
        if (step + 1) % 100 == 0:
            rate = (step + 1) / (time.time() - started)
            left = f", ~{(expected - step - 1) / rate / 60:.1f} min left" if expected else ""
            print(f"  {step + 1} frames ({rate:.1f}/s{left})", flush=True)
    if batch:
        flush(batch)
    if not records:
        raise SystemExit("No frames were processed")

    track_lines([(step, lines) for step, (_, _, lines) in enumerate(records)],
                args.track_tolerance_px, args.max_gap)
    counts = {}
    for _, _, lines in records:
        for line in lines:
            counts[line["track"]] = counts.get(line["track"], 0) + 1
    kept_tracks = {t for t, n in counts.items() if n >= args.min_track_frames}

    # ---- CSV and track summary
    with open(os.path.join(args.output_dir, "lines.csv"), "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "time_s", "y", "x0", "x1", "score", "track", "kept"])
        for frame_index, time_s, lines in records:
            for line in lines:
                writer.writerow([frame_index, f"{time_s:.3f}", f"{line['y']:.1f}",
                                 f"{line['x0']:.0f}", f"{line['x1']:.0f}",
                                 f"{line['score']:.3f}", line["track"],
                                 int(line["track"] in kept_tracks)])
    tracks = {}
    for frame_index, time_s, lines in records:
        for line in lines:
            t = tracks.setdefault(line["track"], {"track": line["track"], "ys": [],
                                                  "scores": [], "times": []})
            t["ys"].append(line["y"])
            t["scores"].append(line["score"])
            t["times"].append(time_s)
    summary = []
    for t in sorted(tracks.values(), key=lambda t: np.mean(t["ys"])):
        summary.append({"track": t["track"], "kept": t["track"] in kept_tracks,
                        "detections": len(t["ys"]),
                        "start_s": round(min(t["times"]), 3), "end_s": round(max(t["times"]), 3),
                        "mean_y": round(float(np.mean(t["ys"])), 1),
                        "y_range": [round(float(min(t["ys"])), 1), round(float(max(t["ys"])), 1)],
                        "mean_score": round(float(np.mean(t["scores"])), 3)})
    with open(os.path.join(args.output_dir, "tracks.json"), "w") as handle:
        json.dump({"frames_processed": len(records), "every": args.every,
                   "threshold": args.threshold, "min_track_frames": args.min_track_frames,
                   "checkpoints": args.checkpoint.split(","), "tracks": summary},
                  handle, indent=2)

    # ---- Timeline figure
    times = np.array([r[1] for r in records])
    raw_counts = np.array([len(r[2]) for r in records])
    kept_counts = np.array([sum(line["track"] in kept_tracks for line in r[2]) for r in records])
    spacing = []
    for _, _, lines in records:
        ys = sorted(line["y"] for line in lines if line["track"] in kept_tracks)
        spacing.append(float(np.median(np.diff(ys))) if len(ys) >= 2 else np.nan)

    figure, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True,
                                gridspec_kw={"height_ratios": [3, 1, 1]},
                                constrained_layout=True)
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    for t in tracks.values():
        if t["track"] in kept_tracks:
            axes[0].plot(t["times"], t["ys"], ".", ms=2.5,
                         color=colors[t["track"] % 20])
        else:
            axes[0].plot(t["times"], t["ys"], "x", ms=3, color="0.6")
    axes[0].invert_yaxis()
    axes[0].set_ylabel("line y (px, image rows)")
    axes[0].set_title("Integer-line detections: colored = persistent tracks, "
                      "grey x = dropped as flicker")
    axes[1].plot(times, raw_counts, color="0.6", lw=0.8, label="all detections")
    axes[1].plot(times, kept_counts, color="C0", lw=1.0, label="persistent only")
    axes[1].set_ylabel("lines / frame")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[2].plot(times, spacing, color="C3", lw=0.8)
    axes[2].set_ylabel("median spacing (px)")
    axes[2].set_xlabel("time (s)")
    figure.savefig(os.path.join(args.output_dir, "timeline.png"), dpi=150)
    plt.close(figure)

    # ---- Overlays
    step_records = dict(enumerate(records))
    for step, (frame_index, bg_removed) in overlay_jobs.items():
        image = Image.fromarray(bg_removed).convert("RGB")
        draw = ImageDraw.Draw(image)
        for line in step_records[step][2]:
            color = (255, 40, 40) if line["track"] in kept_tracks else (255, 220, 0)
            draw.line([(line["x0"], line["y"]), (line["x1"], line["y"])], fill=color, width=3)
        draw.text((10, 10), f"frame {frame_index}  t={step_records[step][1]:.2f}s  "
                            f"red = persistent, yellow = dropped", fill=(255, 255, 255))
        image.save(os.path.join(overlay_dir, f"frame{frame_index:06d}.png"))

    elapsed = time.time() - started
    print(f"Processed {len(records)} frames in {elapsed / 60:.1f} min; "
          f"{len(tracks)} tracks, {len(kept_tracks)} persistent "
          f"(>= {args.min_track_frames} detections)")
    print(f"Lines/frame: {raw_counts.mean():.2f} raw, {kept_counts.mean():.2f} persistent")
    print(f"Results in {args.output_dir}")


if __name__ == "__main__":
    main()
