"""
Extract integer-line annotations from RHEED frames labeled by drawing
red lines directly on top of the grayscale image (as RGB PNGs).

The mask uses red-channel dominance rather than brightness, so saturated
white diffraction spots and ring edges cannot be mistaken for annotations.
Connected components separate individual line instances; geometric filters
retain thin, horizontally elongated strokes, including slightly sloped ones.

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


def extract_lines(path, red_min=180, red_margin=70, min_width=15,
                  max_height=25, min_aspect=8.0):
    """
    Return a list of line instances found in one image. Each instance is
    a list of [x, y] points forming its centerline (one y-value per x
    column, averaged across the stroke's thickness).
    """
    rgb = np.array(Image.open(path).convert("RGB"), dtype=np.int16)
    red = rgb[..., 0]
    green = rgb[..., 1]
    blue = rgb[..., 2]
    mask = ((red >= red_min) &
            ((red - green) >= red_margin) &
            ((red - blue) >= red_margin))

    labeled, n = ndimage.label(mask, structure=np.ones((3, 3)))
    lines = []
    for lbl in range(1, n + 1):
        ys, xs = np.where(labeled == lbl)
        w = xs.max() - xs.min() + 1
        h = ys.max() - ys.min() + 1
        aspect = w / max(h, 1)
        if w < min_width or h > max_height or aspect < min_aspect:
            continue  # not thin/elongated enough to be an annotation stroke

        pts = {}
        for x, y in zip(xs, ys):
            pts.setdefault(int(x), []).append(int(y))
        centerline = [[x, float(np.mean(v))] for x, v in sorted(pts.items())]
        lines.append(centerline)

    lines.sort(key=lambda L: np.median([point[1] for point in L]))  # top to bottom
    return lines


def check_pairing(labeled_paths, clean_dir, red_min, red_margin):
    """
    Warn when an annotated image is not a copy of the clean frame with the
    same filename. Each annotated image (red strokes masked out) is compared
    to every clean frame; the closest one should be its namesake.
    """
    names = [os.path.basename(p) for p in labeled_paths
             if os.path.exists(os.path.join(clean_dir, os.path.basename(p)))]
    clean = {n: np.asarray(Image.open(os.path.join(clean_dir, n)).convert("L"),
                           dtype=np.float32) for n in names}
    problems = []
    for name in names:
        rgb = np.array(Image.open(os.path.join(os.path.dirname(labeled_paths[0]), name))
                       .convert("RGB"), dtype=np.int16)
        strokes = ((rgb[..., 0] >= red_min // 2) &
                   ((rgb[..., 0] - rgb[..., 1]) >= red_margin // 2))
        keep = ~ndimage.binary_dilation(strokes, iterations=4)
        gray = rgb.mean(axis=-1)
        diffs = {other: float(np.abs(gray[keep] - image[keep]).mean())
                 for other, image in clean.items() if image.shape == gray.shape}
        best = min(diffs, key=diffs.get)
        if best != name:
            problems.append(name)
            print(f"WARNING: annotated {name} matches clean {best} "
                  f"(diff {diffs[best]:.2f}) better than clean {name} "
                  f"(diff {diffs.get(name, float('nan')):.2f})")
    if not problems:
        print(f"Pairing check passed for {len(names)} frame(s).")
    return problems


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Folder of annotated PNGs")
    parser.add_argument("--output", default="labels.json")
    parser.add_argument("--red_min", type=int, default=180,
                         help="Minimum red-channel value for an annotation pixel")
    parser.add_argument("--red_margin", type=int, default=70,
                         help="Required red dominance over both green and blue")
    parser.add_argument("--min_width", type=int, default=15,
                         help="Minimum horizontal span (px) to count as a line")
    parser.add_argument("--max_height", type=int, default=25,
                         help="Maximum vertical span (px), allowing slightly sloped lines")
    parser.add_argument("--min_aspect", type=float, default=8.0,
                         help="Minimum width/height ratio for a line component")
    parser.add_argument("--clean_dir", default="",
                        help="Clean (unannotated) frames; verifies each annotated file "
                             "is a copy of the clean frame with the same name")
    args = parser.parse_args()

    all_labels = []
    paths = glob.glob(os.path.join(args.input_dir, "*.png"))
    paths.sort(key=lambda p: (0, int(os.path.splitext(os.path.basename(p))[0]))
               if os.path.splitext(os.path.basename(p))[0].isdigit()
               else (1, os.path.basename(p)))
    for path in paths:
        lines = extract_lines(path, args.red_min, args.red_margin,
                              args.min_width, args.max_height, args.min_aspect)
        all_labels.append({
            "frame": os.path.basename(path),
            "lines": [{"instance": i, "points": L} for i, L in enumerate(lines)],
        })
        y_positions = [round(float(np.median([point[1] for point in L]))) for L in lines]
        print(f"{os.path.basename(path)}: {len(lines)} line(s), y-positions: {y_positions}")

    if args.clean_dir:
        check_pairing(paths, args.clean_dir, args.red_min, args.red_margin)

    with open(args.output, "w") as f:
        json.dump(all_labels, f, indent=2)

    print(f"\nSaved {len(all_labels)} frames' worth of labels to {args.output}")


if __name__ == "__main__":
    main()
