from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import joblib
from PIL import Image

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Import  5.1 implementation 
from selective_search import selective_search


# -----------------------------
# Defaults (CLI overrides)
# -----------------------------
DEFAULT_SS_SCALES = [100, 200, 500]
DEFAULT_SS_SIGMA = 0.8
DEFAULT_SS_MIN_SIZE = 20

TP_IOU_DEFAULT = 0.50
TN_IOU_DEFAULT = 0.10

# proposal filtering (keeps runtime sane)
MIN_REGION_PIXELS = 20   #100
MAX_ASPECT_RATIO = 3.0
MAX_PROPOSALS_PER_IMAGE = 2000  # after filtering, keep top-K proposals (mixed strategy)

# training balance
MAX_NEG_PER_IMAGE = 800

# HNM (hard negative mining)
HNM_ROUNDS_DEFAULT = 2
HNM_CANDIDATES_PER_IMAGE = 1200
HNM_KEEP_PER_IMAGE = 150

# inference defaults
NMS_IOU_DEFAULT = 0.30
SCORE_THRESH_DEFAULT = -1e9  #2.0
TOP_DETECTIONS_DEFAULT = 100 #10

# crop padding (helps reduce “partial balloon” bias)
CROP_PAD_FRAC = 0.05  #0.10


# -----------------------------
# COCO helpers
# -----------------------------
def load_coco(ann_path: Path) -> Dict[str, Any]:
    with open(ann_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = coco.get("images", [])
    anns = coco.get("annotations", [])
    cats = coco.get("categories", [])

    coco["_images"] = images
    coco["_anns"] = anns
    coco["_cats"] = cats

    coco["images_by_id"] = {im["id"]: im for im in images}
    ann_by_image = {}
    for a in anns:
        ann_by_image.setdefault(a["image_id"], []).append(a)
    coco["ann_by_image"] = ann_by_image

    #  choose the category_id that is actually used in annotations
    ann_cat_ids = [a["category_id"] for a in anns if "category_id" in a]
    if ann_cat_ids:
        # most common category id
        coco["category_id"] = int(max(set(ann_cat_ids), key=ann_cat_ids.count))
    else:
        coco["category_id"] = cats[0]["id"] if len(cats) else 1

    return coco



def bbox_xywh_to_xyxy(b: List[float]) -> np.ndarray:
    x, y, w, h = b
    return np.array([x, y, x + w, y + h], dtype=np.float32)


def get_gt_boxes_xyxy(coco: Dict[str, Any], image_id: int) -> np.ndarray:
    anns = coco["ann_by_image"].get(image_id, [])
    boxes = []
    for a in anns:
        if "bbox" in a:
            boxes.append(bbox_xywh_to_xyxy(a["bbox"]))
    if not boxes:
        return np.zeros((0, 4), dtype=np.float32)
    return np.stack(boxes, axis=0)


# -----------------------------
# IoU / NMS
# -----------------------------
def iou_matrix_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (N,4), b: (M,4) -> IoU (N,M)"""
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)

    ax1 = a[:, 0:1]; ay1 = a[:, 1:2]; ax2 = a[:, 2:3]; ay2 = a[:, 3:4]
    bx1 = b[:, 0];   by1 = b[:, 1];   bx2 = b[:, 2];   by2 = b[:, 3]

    inter_x1 = np.maximum(ax1, bx1)
    inter_y1 = np.maximum(ay1, by1)
    inter_x2 = np.minimum(ax2, bx2)
    inter_y2 = np.minimum(ay2, by2)

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    area_a = np.maximum(0.0, (ax2 - ax1)) * np.maximum(0.0, (ay2 - ay1))
    area_b = np.maximum(0.0, (bx2 - bx1)) * np.maximum(0.0, (by2 - by1))
    union = area_a + area_b - inter + 1e-12
    return (inter / union).astype(np.float32)


def nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> List[int]:
    if boxes.size == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-12)

        inds = np.where(iou <= iou_thr)[0]
        order = order[inds + 1]
    return keep


# -----------------------------
# Proposal generation (cached)
# -----------------------------
def _mixed_topk_by_area(boxes: np.ndarray, k: int) -> np.ndarray:
    """
    Avoid keeping ONLY huge boxes.
    Keep a mix: large + small + random middle.
    """
    n = boxes.shape[0]
    if n <= k:
        return boxes

    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    idx_sorted = areas.argsort()  # small -> large

    k_large = k // 3
    k_small = k // 3
    k_mid = k - k_large - k_small

    small_idx = idx_sorted[:k_small]
    large_idx = idx_sorted[-k_large:]

    mid_pool = idx_sorted[k_small: max(k_small, n - k_large)]
    rng = np.random.default_rng(0)
    if mid_pool.size > k_mid:
        mid_idx = rng.choice(mid_pool, size=k_mid, replace=False)
    else:
        mid_idx = mid_pool

    keep_idx = np.unique(np.concatenate([small_idx, mid_idx, large_idx]))
    return boxes[keep_idx]

def regions_to_boxes_xyxy(regions: List[Dict[str, Any]], im_w: int, im_h: int) -> np.ndarray:
    seen = set()
    boxes = []

    im_area = float(im_w * im_h)

    for r in regions:
        x, y, w, h = r["rect"]
        if w <= 0 or h <= 0:
            continue

        if r.get("size", 0) < MIN_REGION_PIXELS:
            continue

        # aspect ratio filter
        ar1 = w / float(h)
        ar2 = h / float(w)
        if ar1 > MAX_ASPECT_RATIO or ar2 > MAX_ASPECT_RATIO:
            continue

        # clamp to image bounds
        x1 = max(0, min(im_w - 1, int(x)))
        y1 = max(0, min(im_h - 1, int(y)))
        x2 = max(0, min(im_w, int(x + w)))
        y2 = max(0, min(im_h, int(y + h)))
        if x2 <= x1 or y2 <= y1:
            continue

        #  filter by rectangle area 
        box_area = float((x2 - x1) * (y2 - y1))
        if box_area < 50:   # 200
            continue
        if box_area > 0.90 * im_area:
            continue

        key = (x1, y1, x2, y2)
        if key in seen:
            continue
        seen.add(key)
        boxes.append([x1, y1, x2, y2])

    if not boxes:
        return np.zeros((0, 4), dtype=np.float32)

    boxes = np.array(boxes, dtype=np.float32)

    if boxes.shape[0] > MAX_PROPOSALS_PER_IMAGE:
        boxes = _mixed_topk_by_area(boxes, MAX_PROPOSALS_PER_IMAGE)

    return boxes



def generate_proposals_for_image(img: np.ndarray, scales: List[int], sigma: float, min_size: int) -> np.ndarray:
    H, W = img.shape[0], img.shape[1]
    all_boxes = []

    for sc in scales:
        _, regions = selective_search(img, scale=sc, sigma=sigma, min_size=min_size)
        boxes = regions_to_boxes_xyxy(regions, W, H)
        if boxes.size:
            all_boxes.append(boxes)

    if not all_boxes:
        return np.zeros((0, 4), dtype=np.float32)

    boxes = np.concatenate(all_boxes, axis=0)
    boxes = np.unique(boxes, axis=0)
    return boxes.astype(np.float32)


def proposals_cache_path(outdir: Path, split: str) -> Path:
    return outdir / f"proposals_{split}.joblib"


def build_or_load_proposals(
    data_root: Path,
    split: str,
    outdir: Path,
    scales: List[int],
    sigma: float,
    min_size: int,
    force: bool,
) -> Dict[str, Any]:
    outdir.mkdir(parents=True, exist_ok=True)
    cache_p = proposals_cache_path(outdir, split)
    if cache_p.exists() and not force:
        return joblib.load(cache_p)

    split_dir = data_root / split
    coco = load_coco(split_dir / "_annotations.coco.json")

    proposals = {}
    for im in coco["_images"]:
        fn = im["file_name"]
        img = np.asarray(Image.open(split_dir / fn).convert("RGB"))
        boxes = generate_proposals_for_image(img, scales=scales, sigma=sigma, min_size=min_size)
        proposals[fn] = boxes

    payload = {"coco": coco, "proposals": proposals}
    joblib.dump(payload, cache_p)
    return payload


# -----------------------------
# Positive/negative sampling
# -----------------------------
def label_boxes(boxes: np.ndarray, gt: np.ndarray, tp: float, tn: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      y: (N,) {1 pos, 0 neg, -1 ignore}
      best_iou: (N,)
    """
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.int8), np.zeros((0,), dtype=np.float32)
    if gt.size == 0:
        return np.zeros((boxes.shape[0],), dtype=np.int8), np.zeros((boxes.shape[0],), dtype=np.float32)

    ious = iou_matrix_xyxy(boxes, gt)
    best = ious.max(axis=1)

    y = np.full((boxes.shape[0],), -1, dtype=np.int8)
    y[best >= tp] = 1
    y[best <= tn] = 0
    return y, best.astype(np.float32)


def jitter_boxes_xyxy(gt: np.ndarray, im_w: int, im_h: int, num: int = 6, frac: float = 0.20) -> np.ndarray:
    if gt.size == 0 or num <= 0:
        return np.zeros((0, 4), dtype=np.float32)

    rng = np.random.default_rng(0)
    out = []
    for b in gt:
        x1, y1, x2, y2 = b
        w = x2 - x1
        h = y2 - y1
        for _ in range(num):
            dx = rng.uniform(-frac, frac) * w
            dy = rng.uniform(-frac, frac) * h
            ds = rng.uniform(-frac, frac)

            nx1 = x1 + dx
            ny1 = y1 + dy
            nw = w * (1.0 + ds)
            nh = h * (1.0 + ds)

            nx2 = nx1 + nw
            ny2 = ny1 + nh

            nx1 = float(max(0, min(im_w - 1, nx1)))
            ny1 = float(max(0, min(im_h - 1, ny1)))
            nx2 = float(max(0, min(im_w, nx2)))
            ny2 = float(max(0, min(im_h, ny2)))

            if nx2 > nx1 + 1 and ny2 > ny1 + 1:
                out.append([nx1, ny1, nx2, ny2])

    return np.array(out, dtype=np.float32) if out else np.zeros((0, 4), dtype=np.float32)


# -----------------------------
# Feature extraction (Torchvision CNN) + geometry
# -----------------------------
_RESNET_CACHE = None

def get_resnet18():
    global _RESNET_CACHE
    if _RESNET_CACHE is not None:
        return _RESNET_CACHE

    try:
        import torch
        import torchvision
    except Exception as e:
        raise RuntimeError("Torch/Torchvision not available. Install them first.") from e

    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights = torchvision.models.ResNet18_Weights.DEFAULT
    model = torchvision.models.resnet18(weights=weights)

    backbone = torch.nn.Sequential(*(list(model.children())[:-1])).to(device)
    backbone.eval()
    transform = weights.transforms()

    _RESNET_CACHE = (backbone, transform, device, torch)
    return _RESNET_CACHE


def extract_features_resnet18(img: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.zeros((0, 512), dtype=np.float32)

    backbone, transform, device, torch_mod = get_resnet18()

    H, W = img.shape[0], img.shape[1]
    crops = []
    for x1, y1, x2, y2 in boxes:
        bw = float(x2 - x1)
        bh = float(y2 - y1)
        pad_x = int(CROP_PAD_FRAC * bw)
        pad_y = int(CROP_PAD_FRAC * bh)

        x1i = int(max(0, min(W - 1, x1 - pad_x)))
        y1i = int(max(0, min(H - 1, y1 - pad_y)))
        x2i = int(max(0, min(W, x2 + pad_x)))
        y2i = int(max(0, min(H, y2 + pad_y)))

        crop = img[y1i:y2i, x1i:x2i, :] if (x2i > x1i and y2i > y1i) else img
        pil = Image.fromarray(crop)
        crops.append(transform(pil))

    feats = []
    bs = 64
    with torch_mod.no_grad():
        for i in range(0, len(crops), bs):
            batch = torch_mod.stack(crops[i:i + bs]).to(device)
            out = backbone(batch).squeeze(-1).squeeze(-1)  # (B,512)
            feats.append(out.detach().cpu().numpy().astype(np.float32))

    return np.concatenate(feats, axis=0)


def add_geom_features(feats: np.ndarray, boxes: np.ndarray, im_w: int, im_h: int) -> np.ndarray:
    if feats.size == 0:
        return feats
    w = (boxes[:, 2] - boxes[:, 0]) / float(im_w)
    h = (boxes[:, 3] - boxes[:, 1]) / float(im_h)
    area = w * h
    ar = w / (h + 1e-6)
    geom = np.stack([w, h, area, ar, 1.0 / (ar + 1e-6)], axis=1).astype(np.float32)
    return np.concatenate([feats, geom], axis=1)

def force_one_positive_per_gt(boxes: np.ndarray, gt: np.ndarray, y: np.ndarray, min_iou: float = 0.30) -> np.ndarray:
    if boxes.size == 0 or gt.size == 0:
        return y
    ious = iou_matrix_xyxy(gt, boxes)  # (G,P)
    best_idx = ious.argmax(axis=1)
    best_iou = ious.max(axis=1)
    for idx, biou in zip(best_idx, best_iou):
        if biou >= min_iou:
            y[int(idx)] = 1
    return y
# -----------------------------
# Train data + SVM + Hard Negative Mining
# -----------------------------
def build_training_data(data_root: Path, split: str, outdir: Path, tp: float, tn: float) -> Tuple[np.ndarray, np.ndarray]:
    payload = joblib.load(proposals_cache_path(outdir, split))
    coco = payload["coco"]
    props = payload["proposals"]
    split_dir = data_root / split

    X_list = []
    y_list = []
    rng = np.random.default_rng(0)

    for im in coco["_images"]:
        fn = im["file_name"]
        image_id = im["id"]

        img = np.asarray(Image.open(split_dir / fn).convert("RGB"))
        H, W = img.shape[0], img.shape[1]

        gt = get_gt_boxes_xyxy(coco, image_id)
        boxes = props[fn]

        # add GT + jitters as positives (helps box quality)
        gt_aug = jitter_boxes_xyxy(gt, W, H, num=6, frac=0.20)
        boxes_all = np.concatenate([boxes, gt, gt_aug], axis=0) if gt.size else boxes

        y, best = label_boxes(boxes_all, gt, tp=tp, tn=tn)
        y = force_one_positive_per_gt(boxes_all, gt, y, min_iou=0.30)
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]

        if neg_idx.size > MAX_NEG_PER_IMAGE:
            neg_idx = rng.choice(neg_idx, size=MAX_NEG_PER_IMAGE, replace=False)

        sel = np.concatenate([pos_idx, neg_idx], axis=0)
        if sel.size == 0:
            continue

        sel_boxes = boxes_all[sel]
        feats = extract_features_resnet18(img, sel_boxes)
        feats = add_geom_features(feats, sel_boxes, W, H)

        X_list.append(feats)
        y_list.append(y[sel].astype(np.int32))

    if not X_list:
        raise RuntimeError("No training samples produced. Try lowering tp or increasing proposals.")
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    return X, y


def train_svm(X: np.ndarray, y: np.ndarray) -> Pipeline:
    clf = Pipeline([
        ("scaler", StandardScaler(with_mean=True)),
        ("svm", LinearSVC(C=1.0, class_weight="balanced", max_iter=30000)),
    ])
    clf.fit(X, y)
    return clf


def collect_hard_negatives(
    data_root: Path,
    split: str,
    outdir: Path,
    clf: Pipeline,
    tp: float,
    tn: float,
    keep_per_image: int,
) -> Tuple[np.ndarray, np.ndarray]:
    payload = joblib.load(proposals_cache_path(outdir, split))
    coco = payload["coco"]
    props = payload["proposals"]
    split_dir = data_root / split

    X_hn = []
    y_hn = []

    rng = np.random.default_rng(0)

    for im in coco["_images"]:
        fn = im["file_name"]
        image_id = im["id"]
        img = np.asarray(Image.open(split_dir / fn).convert("RGB"))
        H, W = img.shape[0], img.shape[1]

        gt = get_gt_boxes_xyxy(coco, image_id)
        boxes = props[fn]
        if boxes.size == 0:
            continue

        y, _ = label_boxes(boxes, gt, tp=tp, tn=tn)
        neg_idx = np.where(y == 0)[0]
        if neg_idx.size == 0:
            continue

        # score a subset of negatives to find hard ones
        if neg_idx.size > HNM_CANDIDATES_PER_IMAGE:
            neg_idx = rng.choice(neg_idx, size=HNM_CANDIDATES_PER_IMAGE, replace=False)

        neg_boxes = boxes[neg_idx]
        feats = extract_features_resnet18(img, neg_boxes)
        feats = add_geom_features(feats, neg_boxes, W, H)

        scores = clf.decision_function(feats).astype(np.float32)
        top = scores.argsort()[::-1][:keep_per_image]
        X_hn.append(feats[top])
        y_hn.append(np.zeros((top.size,), dtype=np.int32))

    if not X_hn:
        return np.zeros((0, 517), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    return np.concatenate(X_hn, axis=0), np.concatenate(y_hn, axis=0)


# -----------------------------
# Inference
# -----------------------------

def infer_image(
    img: np.ndarray,
    clf: Pipeline,
    scales: List[int],
    sigma: float,
    min_size: int,
    score_thr: float,
    nms_iou: float,
    top_det: int | None,
) -> Tuple[np.ndarray, np.ndarray]:
    boxes = generate_proposals_for_image(img, scales=scales, sigma=sigma, min_size=min_size)
    if boxes.size == 0:
        return boxes, np.zeros((0,), dtype=np.float32)

    H, W = img.shape[0], img.shape[1]
    feats = extract_features_resnet18(img, boxes)
    feats = add_geom_features(feats, boxes, W, H)

    scores = clf.decision_function(feats).astype(np.float32)

    keep = np.where(scores > score_thr)[0]
    if keep.size == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    b = boxes[keep]
    s = scores[keep]

    #  IMPORTANT: allow disabling NMS during eval
    if nms_iou < 1.0:
        keep2 = nms_xyxy(b, s, iou_thr=nms_iou)
        b = b[keep2]
        s = s[keep2]

    #  allow keeping many boxes for eval
    if top_det is not None and b.shape[0] > top_det:
        idx = s.argsort()[::-1][:top_det]
        b = b[idx]
        s = s[idx]

    return b, s

# -----------------------------
# Metrics
# -----------------------------
def compute_mabo_for_split(data_root: Path, split: str, outdir: Path) -> float:
    payload = joblib.load(proposals_cache_path(outdir, split))
    coco = payload["coco"]
    props = payload["proposals"]

    all_best = []
    for im in coco["_images"]:
        fn = im["file_name"]
        image_id = im["id"]
        gt = get_gt_boxes_xyxy(coco, image_id)
        boxes = props[fn]
        if gt.size == 0:
            continue
        if boxes.size == 0:
            all_best.extend([0.0] * gt.shape[0])
            continue
        ious = iou_matrix_xyxy(gt, boxes)  # (G,P)
        best = ious.max(axis=1)
        all_best.extend(best.tolist())

    return float(np.mean(all_best)) if all_best else 0.0


def eval_coco_map(
    data_root: Path,
    split: str,
    outdir: Path,
    model_path: Path,
    scales: List[int],
    sigma: float,
    min_size: int,
    score_thr: float,
    nms_iou: float,
    top_det: int,
):
    split_dir = data_root / split
    ann_path = split_dir / "_annotations.coco.json"

    clf: Pipeline = joblib.load(model_path)

    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except Exception as e:
        print("pycocotools is not installed, so COCO mAP cannot be computed here.")
        print("Install: pip install pycocotools   (or pycocotools-windows on Windows)")
        print("Error:", e)
        return

    coco_gt = COCO(str(ann_path))

    #  pick category_id actually used in annotations ( 1)
    ann_cat_ids = [a["category_id"] for a in coco_gt.dataset.get("annotations", []) if "category_id" in a]
    if ann_cat_ids:
        cat_id = int(max(set(ann_cat_ids), key=ann_cat_ids.count))
    else:
        cat_ids = coco_gt.getCatIds()
        cat_id = int(cat_ids[0]) if cat_ids else 1

    #  eval must be "relaxed" (no threshold + no NMS + many dets)
    eval_score_thr = -1e9
    eval_nms_iou = 1.0       # disables NMS
    eval_top_det = 1000      # keep many; COCO will use maxDets internally

    imgs = coco_gt.loadImgs(coco_gt.getImgIds())

    detections = []
    for im in imgs:
        fn = im["file_name"]
        img_id = im["id"]
        img = np.asarray(Image.open(split_dir / fn).convert("RGB"))

        boxes, scores = infer_image(
            img, clf,
            scales=scales, sigma=sigma, min_size=min_size,
            score_thr=eval_score_thr, nms_iou=eval_nms_iou, top_det=eval_top_det
        )

        for b, s in zip(boxes, scores):
            x1, y1, x2, y2 = b
            detections.append({
                "image_id": int(img_id),
                "category_id": int(cat_id),
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(s),
            })

    #  debug prints to confirm eval is sane
    print("Eval category_id used for detections:", cat_id)
    print("Total detections written:", len(detections))

    coco_dt = coco_gt.loadRes(detections) if len(detections) else coco_gt.loadRes([])
    ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()



# -----------------------------
# CLI
# -----------------------------
def parse_scales(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--outdir", type=str, default="cache_5_2")
    ap.add_argument("--cmd", type=str, required=True, choices=["propose", "train", "infer_one", "eval"])
    ap.add_argument("--split", type=str, default="train", choices=["train", "valid", "test"])
    ap.add_argument("--force", action="store_true")

    ap.add_argument("--scales", type=str, default="100,200,500")
    ap.add_argument("--sigma", type=float, default=DEFAULT_SS_SIGMA)
    ap.add_argument("--min_size", type=int, default=DEFAULT_SS_MIN_SIZE)

    ap.add_argument("--tp", type=float, default=TP_IOU_DEFAULT)
    ap.add_argument("--tn", type=float, default=TN_IOU_DEFAULT)

    ap.add_argument("--hnm_rounds", type=int, default=HNM_ROUNDS_DEFAULT)
    ap.add_argument("--hnm_keep_per_image", type=int, default=HNM_KEEP_PER_IMAGE)

    ap.add_argument("--score_thr", type=float, default=SCORE_THRESH_DEFAULT)
    ap.add_argument("--nms_iou", type=float, default=NMS_IOU_DEFAULT)
    ap.add_argument("--top_det", type=int, default=TOP_DETECTIONS_DEFAULT)

    ap.add_argument("--image", type=str, default="")
    ap.add_argument("--model", type=str, default="")

    args = ap.parse_args()

    data_root = Path(args.data_root)
    outdir = Path(args.outdir)
    scales = parse_scales(args.scales)

    if args.cmd == "propose":
        build_or_load_proposals(
            data_root=data_root,
            split=args.split,
            outdir=outdir,
            scales=scales,
            sigma=args.sigma,
            min_size=args.min_size,
            force=args.force,
        )
        print(f"Saved proposals cache: {proposals_cache_path(outdir, args.split)}")

    elif args.cmd == "train":
        build_or_load_proposals(data_root, "train", outdir, scales, args.sigma, args.min_size, force=args.force)
        build_or_load_proposals(data_root, "valid", outdir, scales, args.sigma, args.min_size, force=args.force)

        X, y = build_training_data(data_root, "train", outdir, tp=args.tp, tn=args.tn)
        print(f"Training samples: X={X.shape}, pos={(y==1).sum()}, neg={(y==0).sum()}")

        clf = train_svm(X, y)

        # Hard Negative Mining rounds
        for r in range(1, args.hnm_rounds + 1):
            X_hn, y_hn = collect_hard_negatives(
                data_root, "train", outdir, clf, tp=args.tp, tn=args.tn,
                keep_per_image=args.hnm_keep_per_image
            )
            if X_hn.shape[0] == 0:
                print(f"HNM round {r}: no hard negatives found.")
                break

            print(f"HNM round {r}: added hard negatives: {X_hn.shape}")
            X = np.concatenate([X, X_hn], axis=0)
            y = np.concatenate([y, y_hn], axis=0)
            clf = train_svm(X, y)


            # X, y = X2, y2

        outdir.mkdir(parents=True, exist_ok=True)
        model_path = outdir / "svm_resnet18.joblib"
        joblib.dump(clf, model_path)
        print(f"Saved model: {model_path}")

    elif args.cmd == "infer_one":
        model_path = Path(args.model) if args.model else (outdir / "svm_resnet18.joblib")
        clf: Pipeline = joblib.load(model_path)

        img_path = Path(args.image)
        img = np.asarray(Image.open(img_path).convert("RGB"))
        boxes, scores = infer_image(
            img, clf,
            scales=scales, sigma=args.sigma, min_size=args.min_size,
            score_thr=args.score_thr, nms_iou=args.nms_iou, top_det=args.top_det
        )

        print(f"Detections: {boxes.shape[0]}")
        for b, s in zip(boxes[:20], scores[:20]):
            x1, y1, x2, y2 = b
            print(f"score={s:.3f} box=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")

        vis_path = outdir / f"infer_{img_path.stem}.png"
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        ax.imshow(img)
        for b, s in zip(boxes, scores):
            x1, y1, x2, y2 = b
            rect = mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="red", linewidth=2)
            ax.add_patch(rect)
            ax.text(x1, y1, f"{s:.2f}", color="red", fontsize=8)
        ax.axis("off")
        fig.savefig(vis_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        print("Saved visualization to:", vis_path)

    elif args.cmd == "eval":
        build_or_load_proposals(data_root, args.split, outdir, scales, args.sigma, args.min_size, force=args.force)

        mabo = compute_mabo_for_split(data_root, args.split, outdir)
        print(f"MABO (proposals) on {args.split}: {mabo:.4f}")

        model_path = Path(args.model) if args.model else (outdir / "svm_resnet18.joblib")
        eval_coco_map(
            data_root, args.split, outdir, model_path,
            scales=scales, sigma=args.sigma, min_size=args.min_size,
            score_thr=args.score_thr, nms_iou=args.nms_iou, top_det=args.top_det
        )


if __name__ == "__main__":
    main()




