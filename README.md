# Selective Search & a Region-Proposal Detection Pipeline

An implementation of the **Selective Search** region-proposal algorithm
(Uijlings et al., *Selective Search for Object Recognition*, IJCV 2013) built on top of
Felzenszwalb–Huttenlocher initial segmentation, together with a complete
**object-detection pipeline** (proposals → CNN features → SVM → NMS → COCO evaluation)
trained and evaluated on a small balloon dataset.

---

## Overview

The project has two parts:

- **Part 1 — Selective Search (`selective_search.py`).** A from-scratch implementation of the
  hierarchical region-merging algorithm. Starting from a Felzenszwalb over-segmentation, regions are
  iteratively merged by a combined similarity (colour, texture, size, fill) until the whole image is
  one region, emitting a bounding box for every region created along the way. Tested on images from
  three domains: Art History, Christian Archaeology, and Classical Archaeology.

- **Part 2 — Detection pipeline (`Individual_solution.py`).** A simplified version of the Uijlings
  et al. detection pipeline. Selective Search generates candidate regions; positives/negatives are
  mined by IoU against ground truth; each region is encoded with a **pretrained ResNet-18** feature
  vector plus geometry features; a **linear SVM** is trained with **hard-negative mining**; and the
  detector is evaluated with **COCO mAP** and **MABO**.

---

## Repository structure

```
.
├── code/
│   ├── selective_search.py      # Part 1: selective search algorithm (8 tasks)
│   ├── main.py                  # Part 1: runs SS on a single image, filters & draws boxes
│   └── Individual_solution.py   # Part 2: full detection pipeline (CLI)
├── data/
│   ├── arthist/                 # Art History test images
│   ├── chrisarch/               # Christian Archaeology test images
│   ├── classarch/               # Classical Archaeology test images
│   └── balloons/                # COCO-format balloon dataset (train/ valid/ test/)
├── results/                     # Output visualizations
├── requirements.txt
└── README.md
```

---

## Installation

```bash
pip install -r requirements.txt
```

Core dependencies: `numpy`, `scikit-image`, `scikit-learn`, `matplotlib`, `pillow`, `joblib`.
Part 2 additionally requires `torch`, `torchvision` (ResNet-18 features) and `pycocotools` (COCO mAP).

```bash
pip install pycocotools        # or pycocotools-windows on Windows
```

---

## Part 1 — Selective Search

### Run

```bash
cd code
python main.py
```

This runs selective search on a single image, filters the resulting proposals, draws the surviving
boxes in red, and saves the visualization to `results/`.

> **Note:** `main.py` currently selects the input image via a hardcoded path near the top of the file
> (one active line, the rest commented out). Edit `image_path` to point at the image you want, or to
> any of the three domain folders, before running.

### Proposal filtering in `main.py`

Generated regions are kept only if they pass three filters:

- duplicate rectangles are discarded,
- regions smaller than `1600` pixels are discarded,
- boxes with an aspect ratio outside `1 / 1.2 … 1.2` are discarded.

Both thresholds are easy to experiment with at the top of the loop.

### What is implemented (the 8 tasks)

| Task | Function | Description |
|------|----------|-------------|
| 1 | `generate_segments` | Felzenszwalb initial segmentation; label map appended as a 4th image channel |
| 2.1–2.4 | `sim_colour`, `sim_texture`, `sim_size`, `sim_fill` | Four similarity measures (histogram intersection for colour/texture) |
| 2.5 | `calc_colour_hist`, `calc_texture_gradient`, `calc_texture_hist`, `extract_regions` | 25-bin HSV colour histograms, LBP texture gradient (P=8, R=1, *uniform*), 10-bin texture histograms, per-region descriptors |
| 3 | `extract_neighbours` | Adjacency list of touching regions (with an x-sorted early-exit optimization) |
| 4 | `merge_regions` | Merge two regions; size-weighted histogram averaging |
| 5 & 6 | (in `selective_search`) | Mark and remove similarities involving the merged regions |
| 7 | (in `selective_search`) | Recompute similarities between the new region and its neighbours |
| 8 | (in `selective_search`) | Emit final box proposals `(x, y, w, h)` from every region |

---

## Part 2 — Detection Pipeline

A single CLI driver with four sub-commands. The expensive proposal step is cached to disk so the
later stages can be re-run quickly.

### 1. Generate & cache proposals

```bash
python Individual_solution.py --cmd propose --data_root data/balloons --split train
python Individual_solution.py --cmd propose --data_root data/balloons --split valid
```

### 2. Train (with hard-negative mining)

```bash
python Individual_solution.py --cmd train --data_root data/balloons \
    --tp 0.5 --tn 0.1 --hnm_rounds 2
```

Trains a `LinearSVC` on ResNet-18 + geometry features and writes the model to
`cache_5_2/svm_resnet18.joblib`.

### 3. Inference on a single image

```bash
python Individual_solution.py --cmd infer_one --data_root data/balloons \
    --image path/to/image.jpg --model cache_5_2/svm_resnet18.joblib
```

Runs proposals → classification → NMS and saves an annotated visualization to the output folder.

### 4. Evaluate (MABO + COCO mAP)

```bash
python Individual_solution.py --cmd eval --data_root data/balloons --split test
```

### Pipeline details

- **Proposals:** multi-scale Selective Search (`scales = 100, 200, 500`), then filtered by minimum
  size, aspect ratio, and absolute/relative area, with a mixed small/medium/large top-K cap to avoid
  keeping only the largest boxes.
- **Sampling:** boxes are labelled by IoU against ground truth — positive above `tp`, negative below
  `tn`, ambiguous boxes in between are ignored. Ground-truth boxes plus jittered copies are added as
  extra positives, and at least one positive per ground-truth box is forced when a decent match exists.
- **Features:** 512-dim global-pooled features from a pretrained **ResNet-18** backbone, concatenated
  with 5 normalized geometry features (width, height, area, aspect ratio, inverse aspect ratio).
- **Classifier:** `LinearSVC` (`C=1.0`, `class_weight="balanced"`) inside a `StandardScaler` pipeline,
  refined over configurable rounds of **hard-negative mining**.
- **Inference:** score thresholding + greedy **NMS**; evaluation mode relaxes the threshold/NMS so the
  COCO evaluator sees the full ranked detection set.
- **Metrics:** **MABO** (Mean Average Best Overlap) for proposal quality, and **COCO mAP** via
  `pycocotools` for end-to-end detection quality.

### Results

Evaluated on the **test** split with `scales = 50,100,200,400,800`, `sigma = 0.8`, `min_size = 10`
(5504 detections scored):

| Metric | Value |
|--------|-------|
| MABO (proposal quality) | **0.720** |
| COCO mAP @[IoU=0.50:0.95] | **0.032** |
| AP @[IoU=0.50] | 0.080 |
| AP @[IoU=0.75] | 0.010 |

Full COCO breakdown:

| Metric | IoU | Area | maxDets | Value |
|--------|-----|------|---------|-------|
| AP | 0.50:0.95 | all | 100 | 0.032 |
| AP | 0.50 | all | 100 | 0.080 |
| AP | 0.75 | all | 100 | 0.010 |
| AP | 0.50:0.95 | medium | 100 | 0.059 |
| AP | 0.50:0.95 | large | 100 | 0.038 |
| AR | 0.50:0.95 | all | 1 | 0.052 |
| AR | 0.50:0.95 | all | 10 | 0.184 |
| AR | 0.50:0.95 | all | 100 | 0.380 |
| AR | 0.50:0.95 | medium | 100 | 0.321 |
| AR | 0.50:0.95 | large | 100 | 0.567 |
---



---

## Acknowledgements

- J. R. R. Uijlings, K. E. A. van de Sande, T. Gevers, A. W. M. Smeulders,
  *Selective Search for Object Recognition*, IJCV 2013.
- P. F. Felzenszwalb, D. P. Huttenlocher, *Efficient Graph-Based Image Segmentation*, IJCV 2004.
- Course skeleton code: Prathmesh R. Madhu (educational use).
- Balloon detection dataset in COCO format.
