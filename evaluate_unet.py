"""Evaluate or run inference with a trained RHEED line heatmap model."""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
import torch

from line_metrics import load_label_lines, match_lines, summarize
from postprocess import lattice_lines, profile_lines
from train_unet import UNetResNet18


def load_input(path, size):
    height, width = size
    image = Image.open(path).convert("L").resize((width, height), Image.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def predict(model, image):
    with torch.no_grad():
        return model(torch.from_numpy(image)[None, None].float()).squeeze().numpy()


def detect_lines(prediction, original_size, threshold, lattice_low):
    return {
        "profile": profile_lines(prediction, original_size, threshold=threshold),
        "lattice": lattice_lines(prediction, original_size,
                                 high=threshold, low=lattice_low),
    }


def load_target(path, size):
    height, width = size
    target = np.load(path).astype(np.float32)
    target = Image.fromarray((target * 255).astype(np.uint8)).resize(
        (width, height), Image.BILINEAR)
    return np.asarray(target, dtype=np.float32) / 255.0


def line_centers(heatmap, threshold, min_width=15, merge_y=4.0):
    labeled, count = ndimage.label(
        heatmap >= threshold, structure=np.ones((3, 3)))
    centers = []
    for component in range(1, count + 1):
        ys, xs = np.where(labeled == component)
        if len(xs) == 0 or xs.max() - xs.min() + 1 < min_width:
            continue
        centers.append(float(np.average(ys, weights=heatmap[ys, xs])))
    centers.sort()
    merged = []
    for center in centers:
        if merged and abs(center - merged[-1][-1]) <= merge_y:
            merged[-1].append(center)
        else:
            merged.append([center])
    return [float(np.mean(group)) for group in merged]


def calculate_metrics(prediction, target, threshold, tolerance=3):
    pred_mask = prediction >= threshold
    target_mask = target >= 0.3
    true_positive = int(np.logical_and(pred_mask, target_mask).sum())
    predicted = int(pred_mask.sum())
    actual = int(target_mask.sum())
    union = int(np.logical_or(pred_mask, target_mask).sum())
    precision = true_positive / max(predicted, 1)
    recall = true_positive / max(actual, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    iou = true_positive / max(union, 1)

    distance_to_target = ndimage.distance_transform_edt(~target_mask)
    distance_to_pred = ndimage.distance_transform_edt(~pred_mask)
    tolerant_precision = (float(np.mean(distance_to_target[pred_mask] <= tolerance))
                          if predicted else 0.0)
    tolerant_recall = (float(np.mean(distance_to_pred[target_mask] <= tolerance))
                       if actual else 0.0)
    tolerant_f1 = (2 * tolerant_precision * tolerant_recall /
                   max(tolerant_precision + tolerant_recall, 1e-12))

    pred_centers = line_centers(prediction, threshold)
    target_centers = line_centers(target, 0.3)
    matched = 0
    if pred_centers and target_centers:
        costs = np.abs(np.asarray(target_centers)[:, None] -
                       np.asarray(pred_centers)[None, :])
        rows, cols = linear_sum_assignment(np.where(costs <= 5, costs, 1e6))
        matched = int(sum(costs[row, col] <= 5 for row, col in zip(rows, cols)))

    soft_dice = float((2 * np.sum(prediction * target) + 1e-6) /
                      (np.sum(prediction ** 2) + np.sum(target ** 2) + 1e-6))
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "tolerant_f1": tolerant_f1,
        "soft_dice": soft_dice,
        "predicted_lines": len(pred_centers),
        "target_lines": len(target_centers),
        "matched_lines_within_5px": matched,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="Checkpoint path, or several comma-separated paths "
                             "whose predictions are averaged")
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--frames", required=True,
                        help="Comma-separated filenames to evaluate or predict")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--heatmaps_dir", default="",
                        help="Optional ground-truth heatmaps for quantitative evaluation")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--tolerance", type=int, default=3)
    parser.add_argument("--labels", default="",
                        help="labels.json for line-level metrics in original pixels")
    parser.add_argument("--tolerance_px", type=float, default=20.0,
                        help="Max |dy| in original pixels for a line match")
    parser.add_argument("--lattice_low", type=float, default=0.15,
                        help="Weakest peak the lattice step may accept")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    models = []
    size = None
    for path in args.checkpoint.split(","):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if size is not None and tuple(checkpoint["size"]) != size:
            raise ValueError("All checkpoints must use the same input size")
        size = tuple(checkpoint["size"])
        model = UNetResNet18(pretrained=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        models.append(model)

    frames = [name.strip() for name in args.frames.split(",") if name.strip()]
    has_targets = bool(args.heatmaps_dir)
    label_lines = load_label_lines(args.labels) if args.labels else {}
    line_matches = {"profile": [], "lattice": []}
    columns = 4 if has_targets else 3
    figure, axes = plt.subplots(
        len(frames), columns, figsize=(4 * columns, 3.5 * len(frames)),
        squeeze=False, constrained_layout=True)
    results = []

    for row, name in enumerate(frames):
        image_path = os.path.join(args.images_dir, name)
        with Image.open(image_path) as source:
            original_size = (source.height, source.width)
        image = load_input(image_path, size)
        prediction = np.mean([predict(model, image) for model in models], axis=0)
        detected = detect_lines(prediction, original_size,
                                args.threshold, args.lattice_low)
        frame_lines = {}
        for method, lines in detected.items():
            frame_lines[method] = {"lines": lines}
            if name in label_lines:
                match = match_lines(lines, label_lines[name], args.tolerance_px)
                line_matches[method].append(match)
                frame_lines[method]["match"] = match
        np.save(os.path.join(args.output_dir, name.replace(".png", "_prediction.npy")),
                prediction.astype(np.float32))

        target = None
        metrics = None
        if has_targets:
            target = load_target(
                os.path.join(args.heatmaps_dir, name.replace(".png", ".npy")), size)
            metrics = calculate_metrics(
                prediction, target, args.threshold, tolerance=args.tolerance)

        overlay = np.stack([image] * 3, axis=-1)
        if target is not None:
            overlay[..., 0] = np.maximum(overlay[..., 0], target)
        overlay[..., 1] = np.maximum(
            overlay[..., 1], (prediction >= args.threshold).astype(np.float32))
        overlay[..., 2] *= 0.6

        panels = [(image, f"{name} input", "gray")]
        if target is not None:
            panels.append((target, "Ground truth", "magma"))
        panels.extend([
            (prediction, f"Prediction (max {prediction.max():.2f})", "magma"),
            (overlay, "GT red / prediction green" if target is not None
             else "Prediction overlay (green)", None),
        ])
        for column, (array, title, cmap) in enumerate(panels):
            axes[row, column].imshow(array, cmap=cmap,
                                     vmin=0 if cmap else None,
                                     vmax=1 if cmap else None)
            axes[row, column].set_title(title)
            axes[row, column].axis("off")
        # Lattice lines on the overlay: solid = direct, dashed = lattice-filled.
        sy = size[0] / original_size[0]
        sx = size[1] / original_size[1]
        for line in detected["lattice"]:
            axes[row, columns - 1].plot(
                [line["x0"] * sx, line["x1"] * sx], [line["y"] * sy] * 2,
                color="cyan", linewidth=0.8,
                linestyle="-" if line["source"] == "direct" else "--")

        results.append({
            "frame": name,
            "prediction_max": float(prediction.max()),
            "threshold": args.threshold,
            "metrics": metrics,
            "lines": frame_lines,
        })

    panel_path = os.path.join(args.output_dir, "evaluation_panel.png")
    figure.savefig(panel_path, dpi=160)
    plt.close(figure)

    numeric = [item["metrics"] for item in results if item["metrics"] is not None]
    averages = None
    if numeric:
        metric_names = ["precision", "recall", "f1", "iou", "tolerant_f1", "soft_dice"]
        averages = {key: float(np.mean([item[key] for item in numeric]))
                    for key in metric_names}
        averages["line_recall"] = float(np.mean([
            item["matched_lines_within_5px"] / max(item["target_lines"], 1)
            for item in numeric]))

    line_summary = ({method: summarize(matches)
                     for method, matches in line_matches.items()}
                    if label_lines else None)

    summary = {
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_val_dice": checkpoint.get("val_dice"),
        "threshold": args.threshold,
        "lattice_low": args.lattice_low,
        "tolerance_px": args.tolerance_px,
        "averages": averages,
        "line_metrics": line_summary,
        "frames": results,
        "panel": panel_path,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
