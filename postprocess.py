"""
Turn a predicted heatmap into discrete integer lines.

Two detectors share the same row profile (the strongest x-smoothed heatmap
value in each row):

- ``profile_lines``: every profile peak above one threshold is a line.
- ``lattice_lines``: confident peaks (``high``) act as anchors. Integer
  orders are close to evenly spaced, so the spacing is estimated from the
  anchors and the detector walks outward from each anchor one spacing at a
  time, accepting a weaker peak (``low``) only if it sits near an expected
  lattice position. This recovers faint orders without accepting faint
  peaks that are off the lattice.

All positions are returned in original-image pixel coordinates.
"""

import numpy as np
from scipy import ndimage, signal


def row_profile(heatmap, smooth_x=9):
    smoothed = ndimage.uniform_filter1d(heatmap, smooth_x, axis=1)
    return smoothed, smoothed.max(axis=1)


def profile_peaks(profile, low, min_separation):
    peaks, props = signal.find_peaks(profile, height=low, distance=min_separation)
    return peaks, props["peak_heights"]


def estimate_spacing(anchor_rows, weights, min_spacing, max_spacing, step=0.25):
    """
    Largest spacing whose lattice explains the anchors. Phase coherence is
    1 for d and for every sub-multiple d/n, so the largest near-maximal
    spacing is chosen to avoid halving the lattice.
    """
    rows = np.asarray(anchor_rows, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    spacings = np.arange(min_spacing, max_spacing + step, step)
    phases = np.exp(2j * np.pi * rows[None, :] / spacings[:, None])
    coherence = np.abs((phases * weights).sum(axis=1)) / weights.sum()
    good = np.where(coherence >= 0.9 * coherence.max())[0]
    return float(spacings[good.max()]), float(coherence[good.max()])


def _extent(smoothed, row, level):
    """Contiguous x-run around the row's maximum that stays above ``level``."""
    band = smoothed[max(0, row - 2):row + 3].max(axis=0)
    center = int(np.argmax(band))
    above = band >= level
    x0 = center
    while x0 > 0 and above[x0 - 1]:
        x0 -= 1
    x1 = center
    while x1 < len(band) - 1 and above[x1 + 1]:
        x1 += 1
    return x0, x1


def _to_lines(smoothed, profile, rows, sources, scale_x, scale_y):
    lines = []
    for row, source in sorted(zip(rows, sources)):
        score = float(profile[row])
        x0, x1 = _extent(smoothed, row, 0.5 * score)
        # Sub-row refinement: parabola through the peak and its neighbours.
        y = float(row)
        if 0 < row < len(profile) - 1:
            a, b, c = profile[row - 1], profile[row], profile[row + 1]
            denom = a - 2 * b + c
            if denom < 0:
                y += float(0.5 * (a - c) / denom)
        lines.append({
            "y": float((y + 0.5) * scale_y - 0.5),
            "x0": float(x0 * scale_x),
            "x1": float((x1 + 1) * scale_x - 1),
            "score": score,
            "source": source,
        })
    return lines


def profile_lines(heatmap, original_size, threshold=0.4, min_separation=10):
    """Baseline detector: profile peaks above a single threshold."""
    smoothed, profile = row_profile(heatmap)
    peaks, _ = profile_peaks(profile, threshold, min_separation)
    scale_y = original_size[0] / heatmap.shape[0]
    scale_x = original_size[1] / heatmap.shape[1]
    return _to_lines(smoothed, profile, list(peaks), ["direct"] * len(peaks),
                     scale_x, scale_y)


def lattice_lines(heatmap, original_size, high=0.4, low=0.15,
                  min_spacing=18.0, max_spacing=40.0, window=0.3, max_gap=1,
                  min_separation=10):
    """
    Anchor-and-walk lattice completion. Spacings, window and separation are
    in heatmap rows (training resolution); defaults match the 275-row input,
    where observed order spacings are roughly 22-33 rows.
    """
    smoothed, profile = row_profile(heatmap)
    peaks, heights = profile_peaks(profile, low, min_separation)
    anchors = [int(p) for p, h in zip(peaks, heights) if h >= high]
    accepted = {row: "direct" for row in anchors}

    if len(anchors) >= 2:
        spacing, _ = estimate_spacing(
            anchors, [profile[a] for a in anchors], min_spacing, max_spacing)
        n_rows = len(profile)
        for anchor in anchors:
            for direction in (-1, 1):
                position = float(anchor)
                misses = 0
                while True:
                    expected = position + direction * spacing
                    if expected < 0 or expected > n_rows - 1:
                        break
                    near = [p for p in peaks
                            if abs(p - expected) <= window * spacing]
                    if near:
                        best = max(near, key=lambda p: profile[p])
                        accepted.setdefault(int(best), "lattice")
                        position = float(best)
                        misses = 0
                    else:
                        misses += 1
                        if misses > max_gap:
                            break
                        position = expected

    rows = sorted(accepted)
    scale_y = original_size[0] / heatmap.shape[0]
    scale_x = original_size[1] / heatmap.shape[1]
    return _to_lines(smoothed, profile, rows, [accepted[r] for r in rows],
                     scale_x, scale_y)
