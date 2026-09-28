# RHEED integer-line detection (U-Net)

Detects integer-order lines in RHEED frames. Only integer lines are labelled;
reconstruction lines are deliberately left unlabelled (background).

## Layout

```
*.py                      scripts (run from this folder)
data/
  original_18/            first 18 frames, no background removal (clean + labeled)
  bg_removed/
    clean/                60 background-removed frames (model input)
    labeled/              same frames with red strokes on integer lines
    sources_31_60.json    which video frames 31-60 came from
  labels/                 line labels extracted from bg_removed/labeled
    labels_60_reviewed.json   361 lines (after review round 1)
    labels_60_round2.json     368 lines (after review round 2) <- current
  heatmaps/               training targets rendered from each labels file
  backups/                labeled images before a review round edited them
models/
  ensemble_60_reviewed/   final 3-seed ensemble (seed1337, seed7, seed2024)
experiments/
  cv_60_reviewed/<seed>/  5-fold cross-validation: cv_results.json, cv.log,
                          out-of-fold predictions/
review/
  round1/, round2/        false-positive review sheets and decisions
  make_fp_review.py       builds a review sheet from CV predictions
test_frames/              unseen frames (raw, background_removed) + predictions
```

## Preprocessing (raw RHEED frame -> model input)

1. Grayscale with Rec.709 weights: 0.2126 R + 0.7152 G + 0.0722 B.
2. Subtract a Gaussian blur (sigma 30), clip negatives to 0.
3. Stretch so the 99.7th percentile maps to 255.

## Pipeline

```bash
python3 extract_line_labels.py --input_dir data/bg_removed/labeled \
    --clean_dir data/bg_removed/clean --output data/labels/labels_60_round2.json
python3 render_heatmaps.py --labels data/labels/labels_60_round2.json \
    --images_dir data/bg_removed/clean --output_dir data/heatmaps/60_round2 \
    --preview data/heatmaps/60_round2_preview.png
python3 train_unet.py --images_dir data/bg_removed/clean \
    --heatmaps_dir data/heatmaps/60_round2 --epochs 40 \
    --checkpoint models/ensemble_60_round2/seed1337.pt
python3 evaluate_unet.py \
    --checkpoint models/ensemble_60_reviewed/seed1337.pt,models/ensemble_60_reviewed/seed7.pt,models/ensemble_60_reviewed/seed2024.pt \
    --images_dir test_frames/background_removed --frames rheed_unused_frame_sep27.png \
    --threshold 0.4 --output_dir test_frames/prediction_sep27_ensemble
```
