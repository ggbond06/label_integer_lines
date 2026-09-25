"""
Render Gaussian-ridge heatmaps from line-coordinate labels (as produced
by extract_line_labels.py). These heatmaps are the training targets for
the network -- a dense "how likely is a line here" map, rather than raw
(x, y) coordinates.

Why a heatmap instead of direct coordinate regression: it tolerates small
labeling imprecision, gives the network a smooth learning signal instead
of a single exact pixel, and lets you recover sub-pixel line positions at
inference by taking a weighted average (soft-argmax) down each column.

Usage:
    python render_heatmaps.py --labels labels.json --images_dir data \
        --output_dir heatmaps --sigma 3.0

Produces one .npy heatmap per frame (float32, same H x W as the source
image, values in [0, 1]), plus a side-by-side PNG preview of all frames
overlaid with their heatmaps so you can sanity-check them visually.
"""

import argparse
import json
import os
import numpy as np
from PIL import Image


def render_heatmap(lines, height, width, sigma=3.0):
    """
    Build a single-channel heatmap for one frame. For every labeled line,
    lay down a 1D Gaussian in the y-direction at each x column along the
    line's centerline, and take the max across lines at each pixel (so
    two nearby lines don't add up and saturate).
    """
    heatmap = np.zeros((height, width), dtype=np.float32)
    for line in lines:
        for x, y in line["points"]:
            x = int(round(x))
            if x < 0 or x >= width:
                continue
            y0 = max(0, int(y - 3 * sigma))
            y1 = min(height, int(y + 3 * sigma) + 1)
            ys = np.arange(y0, y1)
            vals = np.exp(-((ys - y) ** 2) / (2 * sigma ** 2))
            heatmap[y0:y1, x] = np.maximum(heatmap[y0:y1, x], vals)
    return heatmap


def extract_lines_from_heatmap(heatmap, threshold=0.3, min_width=15):
    """
    The inverse operation: given a predicted heatmap (from the trained
    model), recover discrete line instances with sub-pixel y positions.
    Useful later for turning model output back into labeled coordinates
    during semi-automated labeling.
    """
    from scipy import ndimage

    mask = heatmap >= threshold
    labeled, n = ndimage.label(mask, structure=np.ones((3, 3)))
    lines = []
    for lbl in range(1, n + 1):
        ys, xs = np.where(labeled == lbl)
        if xs.max() - xs.min() < min_width:
            continue
        pts = {}
        for x, y, w in zip(xs, ys, heatmap[ys, xs]):
            pts.setdefault(int(x), []).append((y, w))
        centerline = []
        for x, vals in sorted(pts.items()):
            ys_here = np.array([v[0] for v in vals])
            ws_here = np.array([v[1] for v in vals])
            y_subpixel = float(np.average(ys_here, weights=ws_here))  # soft-argmax
            centerline.append([x, y_subpixel])
        lines.append(centerline)
    lines.sort(key=lambda L: L[0][1])
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True, help="labels.json from extract_line_labels.py")
    parser.add_argument("--images_dir", required=True, help="Folder with the source frames (for size + preview)")
    parser.add_argument("--output_dir", required=True, help="Where to save .npy heatmaps")
    parser.add_argument("--sigma", type=float, default=3.0, help="Gaussian ridge width in pixels")
    parser.add_argument("--preview", default="heatmap_preview.png", help="Path for the visual sanity-check PNG")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.labels) as f:
        all_labels = json.load(f)

    previews = []
    for entry in all_labels:
        frame_path = os.path.join(args.images_dir, entry["frame"])
        img = np.array(Image.open(frame_path).convert("L"))
        h, w = img.shape

        heatmap = render_heatmap(entry["lines"], h, w, sigma=args.sigma)

        out_path = os.path.join(args.output_dir, entry["frame"].replace(".png", ".npy"))
        np.save(out_path, heatmap)
        print(f"{entry['frame']}: heatmap saved to {out_path} "
              f"({len(entry['lines'])} lines, max value {heatmap.max():.2f})")

        previews.append((entry["frame"], img, heatmap))

    # Build a side-by-side visual sanity check
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n = len(previews)
        cols = min(4, n)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
        axes = np.array(axes).reshape(-1)
        for ax, (name, img, heatmap) in zip(axes, previews):
            rgb = np.stack([img] * 3, axis=-1).astype(np.float32) / 255.0
            rgb[..., 0] = np.maximum(rgb[..., 0], heatmap)
            rgb[..., 1] *= (1 - 0.6 * heatmap)
            rgb[..., 2] *= (1 - 0.6 * heatmap)
            ax.imshow(np.clip(rgb, 0, 1))
            ax.set_title(name)
            ax.axis("off")
        for ax in axes[len(previews):]:
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(args.preview, dpi=100, bbox_inches="tight")
        print(f"\nSaved visual sanity check to {args.preview}")
    except ImportError:
        print("\n(matplotlib not installed, skipping visual preview -- "
              "pip install matplotlib to get heatmap_preview.png)")


if __name__ == "__main__":
    main()
