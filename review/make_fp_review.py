"""
Build a review sheet of cross-validation false positives.

Out-of-fold predictions from the three seed runs are averaged (the same
ensemble as the final model), lines are detected at threshold 0.4, and every
predicted line with no labeled line within the tolerance is cropped from the
background-removed frame. Labeled lines are drawn in red, the candidate in
yellow (dashed, so the pixels underneath stay visible).
"""

import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from line_metrics import load_label_lines  # noqa: E402
from postprocess import profile_lines  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CV_DIRS = ["experiments/cv_60_reviewed/seed1337", "experiments/cv_60_reviewed/seed7",
           "experiments/cv_60_reviewed/seed2024"]
IMAGES = "data/bg_removed/clean"
LABELS = "data/labels/labels_60_reviewed.json"
OUT = os.path.join(ROOT, "review", "round2")
THRESHOLD, TOLERANCE = 0.4, 20.0
CROP_W, CROP_H, PAD_X = 700, 240, 150
PER_SHEET, COLUMNS = 12, 3


def frame_key(name):
    return int(os.path.splitext(name)[0])


def dashed(draw, x0, x1, y, color, width=3, dash=14):
    for start in range(int(x0), int(x1), 2 * dash):
        draw.line([(start, y), (min(start + dash, x1), y)], fill=color, width=width)


def main():
    os.makedirs(OUT, exist_ok=True)
    labels = load_label_lines(os.path.join(ROOT, LABELS))
    candidates = []
    for name in sorted(labels, key=frame_key):
        stem = name.replace(".png", ".npy")
        heatmap = np.mean([np.load(os.path.join(ROOT, d, "predictions", stem))
                           for d in CV_DIRS], axis=0)
        with Image.open(os.path.join(ROOT, IMAGES, name)) as im:
            size = (im.height, im.width)
        for line in profile_lines(heatmap, size, threshold=THRESHOLD):
            distances = [abs(line["y"] - t["y"]) for t in labels[name]]
            if not distances or min(distances) > TOLERANCE:
                candidates.append({"frame": name, **line,
                                   "nearest_label_dy": min(distances) if distances else None})

    font = ImageFont.load_default(size=22)
    tiles = []
    for index, c in enumerate(candidates, 1):
        c["id"] = index
        image = Image.open(os.path.join(ROOT, IMAGES, c["frame"])).convert("RGB")
        cx = (c["x0"] + c["x1"]) / 2
        left = int(np.clip(min(c["x0"] - PAD_X, cx - CROP_W / 2), 0, image.width - CROP_W))
        top = int(np.clip(c["y"] - CROP_H / 2, 0, image.height - CROP_H))
        crop = image.crop((left, top, left + CROP_W, top + CROP_H))
        plain = crop.copy()
        draw = ImageDraw.Draw(crop)
        for t in labels[c["frame"]]:
            if top <= t["y"] < top + CROP_H:
                draw.line([(t["x0"] - left, t["y"] - top), (t["x1"] - left, t["y"] - top)],
                          fill=(255, 0, 0), width=2)
        dashed(draw, c["x0"] - left, c["x1"] - left, c["y"] - top, (255, 220, 0))
        # Stack: annotated on top, untouched crop below, so faint lines are not hidden.
        tile = Image.new("RGB", (CROP_W, 2 * CROP_H + 36), (30, 30, 30))
        tile.paste(crop, (0, 36))
        tile.paste(plain, (0, 36 + CROP_H))
        ImageDraw.Draw(tile).text(
            (8, 6), f"#{c['id']}  frame {c['frame']}  y={c['y']:.0f}  "
                    f"score={c['score']:.2f}", fill=(255, 255, 255), font=font)
        tiles.append(tile)

    sheets = []
    for start in range(0, len(tiles), PER_SHEET):
        chunk = tiles[start:start + PER_SHEET]
        rows = (len(chunk) + COLUMNS - 1) // COLUMNS
        tw, th = chunk[0].size
        sheet = Image.new("RGB", (COLUMNS * (tw + 10), rows * (th + 10)), (0, 0, 0))
        for i, tile in enumerate(chunk):
            sheet.paste(tile, ((i % COLUMNS) * (tw + 10), (i // COLUMNS) * (th + 10)))
        path = os.path.join(OUT, f"fp_sheet_{len(sheets) + 1}.png")
        sheet.save(path)
        sheets.append(path)

    with open(os.path.join(OUT, "fp_candidates.json"), "w") as handle:
        json.dump(candidates, handle, indent=2)
    print(f"{len(candidates)} false positives -> {len(sheets)} sheets in {OUT}")
    for c in candidates:
        print(f"  #{c['id']:>2} {c['frame']:>7} y={c['y']:6.1f} x={c['x0']:.0f}-{c['x1']:.0f} "
              f"score={c['score']:.2f} nearest_label_dy={c['nearest_label_dy']}")


if __name__ == "__main__":
    main()
