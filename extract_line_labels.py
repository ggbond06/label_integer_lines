"""
Extract integer-line annotations from RHEED frames labeled by drawing
solid white lines directly on top of the grayscale image (as PNGs).

Works by thresholding for near-pure-white pixels, then using connected-
component labeling to separate individual line instances, filtering out
anything that isn't thin and horizontally elongated (so it won't confuse
a bright specular spot in the pattern for an annotation line).

Usage:
    python extract_line_labels.py --input_dir path/to/pngs --output labels.json
"""

import argparse
import json
import glob
import os
import numpy as np
from PIL import Image
from scipy import ndimage


def extract_lines(path, threshold=250, min_width=15, max_height=5):
    """
    Return a list of line instances found in one image. Each instance is
    a list of [x, y] points forming its centerline (one y-value per x
    column, averaged across the stroke's thickness).
    """
    arr = np.array(Image.open(path).convert("L"))
    mask = arr >= threshold

    labeled, n = ndimage.label(mask, structure=np.ones((3, 3)))
    lines = []
    for lbl in range(1, n + 1):
        ys, xs = np.where(labeled == lbl)
        w = xs.max() - xs.min()
        h = ys.max() - ys.min()
        if w < min_width or h > max_height:
            continue  # not thin/elongated enough to be an annotation stroke

        pts = {}
        for x, y in zip(xs, ys):
            pts.setdefault(int(x), []).append(int(y))
        centerline = [[x, float(np.mean(v))] for x, v in sorted(pts.items())]
        lines.append(centerline)

    lines.sort(key=lambda L: L[0][1])  # top to bottom
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Folder of annotated PNGs")
    parser.add_argument("--output", default="labels.json")
    parser.add_argument("--threshold", type=int, default=250,
                         help="Min pixel intensity (0-255) to count as annotation")
    parser.add_argument("--min_width", type=int, default=15,
                         help="Minimum horizontal span (px) to count as a line")
    parser.add_argument("--max_height", type=int, default=5,
                         help="Maximum vertical span (px) of a line's bounding box")
    args = parser.parse_args()

    all_labels = []
    paths = sorted(glob.glob(os.path.join(args.input_dir, "*.png")))
    for path in paths:
        lines = extract_lines(path, args.threshold, args.min_width, args.max_height)
        all_labels.append({
            "frame": os.path.basename(path),
            "lines": [{"instance": i, "points": L} for i, L in enumerate(lines)],
        })
        y_positions = [round(L[0][1]) for L in lines]
        print(f"{os.path.basename(path)}: {len(lines)} line(s), y-positions: {y_positions}")

    with open(args.output, "w") as f:
        json.dump(all_labels, f, indent=2)

    print(f"\nSaved {len(all_labels)} frames' worth of labels to {args.output}")


if __name__ == "__main__":
    main()
