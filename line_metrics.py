"""
Line-level evaluation in original-image pixel coordinates.

A predicted line matches a labeled line when their y-positions differ by at
most ``tolerance`` pixels (one-to-one, Hungarian assignment). Horizontal
extent is scored separately as 1-D interval IoU on matched pairs, because
how far a stroke was drawn left or right is a labeling choice and should
not decide whether a line was found.
"""

import json

import numpy as np
from scipy.optimize import linear_sum_assignment


def load_label_lines(labels_path):
    """{frame: [{"y", "x0", "x1"}, ...]} from extract_line_labels.py output."""
    with open(labels_path) as handle:
        entries = json.load(handle)
    result = {}
    for entry in entries:
        lines = []
        for line in entry["lines"]:
            points = np.asarray(line["points"], dtype=np.float64)
            lines.append({"y": float(np.median(points[:, 1])),
                          "x0": float(points[:, 0].min()),
                          "x1": float(points[:, 0].max())})
        result[entry["frame"]] = sorted(lines, key=lambda line: line["y"])
    return result


def interval_iou(a0, a1, b0, b1):
    overlap = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return overlap / union if union > 0 else 0.0


def match_lines(predicted, target, tolerance=20.0):
    pairs = []
    if predicted and target:
        costs = np.abs(np.array([t["y"] for t in target])[:, None] -
                       np.array([p["y"] for p in predicted])[None, :])
        # Out-of-tolerance pairs get a prohibitive cost so the assignment
        # maximizes the number of matches rather than minimizing total
        # distance (which can trade two good matches for three bad ones).
        rows, cols = linear_sum_assignment(
            np.where(costs <= tolerance, costs, 1e6))
        pairs = [(int(r), int(c)) for r, c in zip(rows, cols)
                 if costs[r, c] <= tolerance]
    y_errors = [float(predicted[c]["y"] - target[r]["y"]) for r, c in pairs]
    extent_ious = [float(interval_iou(predicted[c]["x0"], predicted[c]["x1"],
                                target[r]["x0"], target[r]["x1"])) for r, c in pairs]
    matched_targets = {r for r, _ in pairs}
    return {
        "tp": len(pairs),
        "fp": len(predicted) - len(pairs),
        "fn": len(target) - len(pairs),
        "y_errors": y_errors,
        "extent_ious": extent_ious,
        "missed_y": [round(t["y"], 1) for i, t in enumerate(target)
                     if i not in matched_targets],
    }


def summarize(matches):
    """Micro-average across frames: every line counts equally."""
    tp = sum(m["tp"] for m in matches)
    fp = sum(m["fp"] for m in matches)
    fn = sum(m["fn"] for m in matches)
    y_errors = np.abs([e for m in matches for e in m["y_errors"]])
    extent_ious = [e for m in matches for e in m["extent_ious"]]
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "line_precision": precision,
        "line_recall": recall,
        "line_f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "tp": tp, "fp": fp, "fn": fn,
        "y_mae_px": float(y_errors.mean()) if len(y_errors) else None,
        "extent_iou": float(np.mean(extent_ious)) if extent_ious else None,
    }
