"""
K-fold cross-validation with line-level scoring.

Frames are split into contiguous blocks of their numeric order, so
neighbouring (near-duplicate) video frames stay in the same fold. Each
fold's model trains for a fixed number of epochs and never sees its
held-out frames -- they are not used for early stopping or checkpoint
selection either, which would leak them into the score.

Out-of-fold predictions are saved, so the detectors can be re-scored with
different thresholds without retraining (--skip_training).

Usage:
    python cross_validate.py --images_dir data/bg_removed/clean \
        --heatmaps_dir data/heatmaps/60_round2 \
        --labels data/labels/labels_60_round2.json \
        --output_dir experiments/cv_60_round2/seed1337
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image
import torch

from evaluate_unet import calculate_metrics, load_input, load_target, predict
from line_metrics import load_label_lines, match_lines, summarize
from postprocess import lattice_lines, profile_lines
from train_unet import UNetResNet18


def frame_key(name):
    stem = os.path.splitext(name)[0]
    return (0, int(stem)) if stem.isdigit() else (1, stem)


def contiguous_folds(frames, k):
    return [list(block) for block in np.array_split(frames, k)]


def load_model(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = UNetResNet18(pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, tuple(checkpoint["size"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--heatmaps_dir", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold_ranges", default="",
                        help="Explicit folds by frame number, e.g. '1-30;31-60' "
                             "(overrides --folds)")
    parser.add_argument("--epochs", type=int, default=40,
                        help="Fixed training length per fold (no held-out early stopping)")
    parser.add_argument("--skip_training", action="store_true",
                        help="Reuse fold checkpoints/predictions already in output_dir")
    parser.add_argument("--tolerance_px", type=float, default=20.0)
    parser.add_argument("--train_args", default="",
                        help="Extra arguments passed to train_unet.py, e.g. '--pos_weight 20'")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    prediction_dir = os.path.join(args.output_dir, "predictions")
    os.makedirs(prediction_dir, exist_ok=True)

    label_lines = load_label_lines(args.labels)
    frames = sorted(label_lines, key=frame_key)
    if args.fold_ranges:
        folds = []
        for spec in args.fold_ranges.split(";"):
            low, high = (int(v) for v in spec.split("-"))
            folds.append([f for f in frames if low <= frame_key(f)[1] <= high])
        covered = [f for fold in folds for f in fold]
        if sorted(covered, key=frame_key) != frames:
            raise ValueError("--fold_ranges must cover every labeled frame exactly once")
    else:
        folds = contiguous_folds(frames, args.folds)
    here = os.path.dirname(os.path.abspath(__file__))

    for index, held_out in enumerate(folds):
        checkpoint = os.path.join(args.output_dir, f"fold{index}.pt")
        print(f"Fold {index}: held out {held_out}", flush=True)
        if not args.skip_training:
            command = [sys.executable, os.path.join(here, "train_unet.py"),
                       "--images_dir", args.images_dir,
                       "--heatmaps_dir", args.heatmaps_dir,
                       "--exclude_frames", ",".join(held_out),
                       "--epochs", str(args.epochs),
                       "--eval_every", str(args.epochs),
                       "--checkpoint", checkpoint] + args.train_args.split()
            subprocess.run(command, check=True)
        model, size = load_model(checkpoint)
        for name in held_out:
            image = load_input(os.path.join(args.images_dir, name), size)
            np.save(os.path.join(prediction_dir, name.replace(".png", ".npy")),
                    predict(model, image))

    # Score out-of-fold predictions.
    original_sizes = {}
    for name in frames:
        with Image.open(os.path.join(args.images_dir, name)) as source:
            original_sizes[name] = (source.height, source.width)
    predictions = {name: np.load(os.path.join(prediction_dir, name.replace(".png", ".npy")))
                   for name in frames}

    def score(detector):
        matches = {name: match_lines(detector(predictions[name], original_sizes[name]),
                                     label_lines[name], args.tolerance_px)
                   for name in frames}
        return summarize(list(matches.values())), matches

    results = {"folds": [[str(f) for f in fold] for fold in folds], "epochs": args.epochs,
               "tolerance_px": args.tolerance_px, "profile": [], "lattice": []}
    print(f"\n{'method':<8} {'high':>5} {'low':>5}  {'P':>6} {'R':>6} {'F1':>6}  "
          f"{'TP':>3} {'FP':>3} {'FN':>3}  {'|dy|px':>6} {'extIoU':>6}")

    def report(method, high, low, summary):
        print(f"{method:<8} {high:5.2f} {low:5.2f}  {summary['line_precision']:6.3f} "
              f"{summary['line_recall']:6.3f} {summary['line_f1']:6.3f}  "
              f"{summary['tp']:3d} {summary['fp']:3d} {summary['fn']:3d}  "
              f"{summary['y_mae_px']:6.2f} {summary['extent_iou']:6.3f}")

    for threshold in (0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05):
        summary, _ = score(lambda p, s: profile_lines(p, s, threshold=threshold))
        results["profile"].append({"threshold": threshold, **summary})
        report("profile", threshold, threshold, summary)
    for high in (0.5, 0.4, 0.3):
        for low in (0.2, 0.15, 0.1, 0.05):
            summary, _ = score(lambda p, s: lattice_lines(p, s, high=high, low=low))
            results["lattice"].append({"high": high, "low": low, **summary})
            report("lattice", high, low, summary)

    # Per-frame detail and pixel metrics at the current defaults.
    _, profile_matches = score(lambda p, s: profile_lines(p, s, threshold=0.4))
    _, lattice_matches = score(lambda p, s: lattice_lines(p, s, high=0.4, low=0.15))
    per_frame = []
    for name in frames:
        target = load_target(os.path.join(args.heatmaps_dir, name.replace(".png", ".npy")),
                             predictions[name].shape)
        pixel = calculate_metrics(predictions[name], target, 0.4)
        per_frame.append({
            "frame": name,
            "target_lines": len(label_lines[name]),
            "profile_0.40": {k: profile_matches[name][k] for k in ("tp", "fp", "fn", "missed_y")},
            "lattice_0.40_0.15": {k: lattice_matches[name][k] for k in ("tp", "fp", "fn", "missed_y")},
            "pixel_f1": pixel["f1"], "tolerant_f1": pixel["tolerant_f1"],
            "soft_dice": pixel["soft_dice"],
        })
    results["per_frame"] = per_frame
    results["pixel_means"] = {key: float(np.mean([f[key] for f in per_frame]))
                              for key in ("pixel_f1", "tolerant_f1", "soft_dice")}

    print("\nPer frame (lattice high=0.40 low=0.15):")
    for item in per_frame:
        m = item["lattice_0.40_0.15"]
        print(f"  {item['frame']:>7}  target={item['target_lines']}  tp={m['tp']} "
              f"fp={m['fp']} fn={m['fn']}  missed_y={m['missed_y']}")
    print("\nPixel means:", {k: round(v, 3) for k, v in results["pixel_means"].items()})

    out_path = os.path.join(args.output_dir, "cv_results.json")
    with open(out_path, "w") as handle:
        json.dump(results, handle, indent=2)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
