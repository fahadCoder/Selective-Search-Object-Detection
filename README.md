# Selective Search & a Region-Proposal Detection Pipeline

An implementation of the **Selective Search** region-proposal algorithm
(Uijlings et al., *Selective Search for Object Recognition*, IJCV 2013) built on top of
Felzenszwalb–Huttenlocher initial segmentation, together with a complete
**object-detection pipeline** (proposals → CNN features → SVM → NMS → COCO evaluation)
trained and evaluated on a small balloon dataset.

This was developed as the Computer Vision course project (Sheet 5, Winter Term 2025/2026, FAU Erlangen-Nürnberg).

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

| Split | MABO | COCO mAP |
|-------|------|----------|
| test  | _TODO_ | _TODO_ |

> Fill in your measured numbers here after running `--cmd eval`.

---

## Discussion

> Draft answers to the exercise questions — edit freely or move to a separate writeup.

**Q5.1.1 — Why Selective Search if Felzenszwalb already segments the image?**
Felzenszwalb produces a single pixel partition at one fixed scale, and objects rarely correspond to
exactly one segment at one scale. Selective Search hierarchically merges those initial segments to
yield a diverse, multi-scale *set of box proposals* likely to contain objects — which is what detection
needs, rather than a single segmentation.

**Q5.1.2 — Proposal-filtering criteria and their effect.**
`main.py` removes duplicates, regions under 1600 px, and boxes whose aspect ratio falls outside
`1/1.2…1.2`. This suppresses tiny noisy regions and very elongated boxes, biasing the output toward
compact, roughly square objects. The aspect-ratio bound is quite strict and would discard legitimately
elongated objects (e.g. standing figures). Useful additions: non-maximum suppression of near-duplicate
boxes, a cap on the number of proposals, a border-touching filter, and score-based ranking.

**Q5.1.3 — From arbitrary shapes to rectangles.**
Each merged region stores its extent, so the box proposal is simply the axis-aligned bounding box of
the region: top-left `(min_x, min_y)` with `width = max_x − min_x` and `height = max_y − min_y`.

**Q5.1.4 — Effect of `k` (Felzenszwalb scale) and number of proposals.**
A larger scale produces larger, fewer initial segments and coarser proposals (risking merging distinct
objects); a smaller scale produces many small segments, more proposals, finer detail, more noise, and
slower runtime. Increasing the number of proposals generally raises recall / MABO (more chances to
cover each ground-truth box) but lowers precision and costs more compute, so there is a trade-off.

**Q5.2.1 — How this differs from Uijlings et al.**
Their pipeline uses SIFT bag-of-words descriptors with a *histogram-intersection-kernel* SVM and an
explicit difficult-negative retraining loop, with multiple colour spaces and complementary similarity
strategies for proposal diversification. This implementation uses pretrained **ResNet-18** features +
geometry with a **linear** SVM, a single similarity combination, a single object class
(balloon vs. background), and a simplified hard-negative-mining loop.

**Q5.2.2 — Effect of the two thresholds, and why two.**
`tp` selects clean positives (high overlap) and `tn` selects clean negatives (low overlap); the gap
between them defines an "ignore" band of ambiguous boxes that partially overlap an object. Training on
those would blur the decision boundary, so excluding them yields a cleaner classifier. A single
threshold would force every borderline box into a class, degrading performance. Raising `tp` gives
fewer but purer positives; lowering `tn` gives purer negatives.

**Q5.2.3 — Increasing training data on the small balloon set.**
Data augmentation (the code already jitters ground-truth boxes; flips, scale, and colour jitter can be
added), promoting high-overlap proposals to extra positives, hard-negative mining to make better use of
abundant background, and transfer learning (already leveraged via the pretrained backbone). Synthetic
compositing of balloons onto new backgrounds is another option.

---

## Acknowledgements

- J. R. R. Uijlings, K. E. A. van de Sande, T. Gevers, A. W. M. Smeulders,
  *Selective Search for Object Recognition*, IJCV 2013.
- P. F. Felzenszwalb, D. P. Huttenlocher, *Efficient Graph-Based Image Segmentation*, IJCV 2004.
- Course skeleton code: Prathmesh R. Madhu (educational use).
- Balloon detection dataset in COCO format.
