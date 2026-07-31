from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import numpy as np
import torch
from torchvision.ops import nms


def box_iou(box_a: Iterable[float], box_b: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(value) for value in box_a]
    bx1, by1, bx2, by2 = [float(value) for value in box_b]
    intersection_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    intersection_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_w * intersection_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def match_predictions(
    predictions: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
    confidence_threshold: float,
    iou_threshold: float,
) -> dict[str, Any]:
    filtered = [
        pred
        for pred in predictions
        if float(pred["confidence"]) >= confidence_threshold
    ]
    tp = 0
    fp = 0
    tp_confidences: list[float] = []
    matched_gt: set[int] = set()

    for pred in sorted(filtered, key=lambda item: float(item["confidence"]), reverse=True):
        candidates = [
            (index, box_iou(pred["bbox_xyxy"], gt["bbox_xyxy"]))
            for index, gt in enumerate(ground_truth)
            if index not in matched_gt and gt["category"] == pred["category"]
        ]
        best_index, best_iou = max(candidates, key=lambda item: item[1], default=(-1, 0.0))
        if best_index >= 0 and best_iou >= iou_threshold:
            matched_gt.add(best_index)
            tp += 1
            tp_confidences.append(float(pred["confidence"]))
        else:
            fp += 1

    fn = len(ground_truth) - len(matched_gt)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp_confidences": tp_confidences,
        "mean_tp_confidence": (
            float(np.mean(tp_confidences)) if tp_confidences else 0.0
        ),
    }


def evaluate_records(
    predictions_by_image: dict[int, list[dict[str, Any]]],
    ground_truth_by_image: dict[int, list[dict[str, Any]]],
    image_ids: list[int],
    confidence_threshold: float,
    iou_threshold: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    totals = {"tp": 0, "fp": 0, "fn": 0}
    all_tp_confidences: list[float] = []
    details: list[dict[str, Any]] = []
    for image_id in image_ids:
        metrics = match_predictions(
            predictions_by_image.get(image_id, []),
            ground_truth_by_image.get(image_id, []),
            confidence_threshold,
            iou_threshold,
        )
        for key in totals:
            totals[key] += int(metrics[key])
        all_tp_confidences.extend(metrics["tp_confidences"])
        details.append(
            {
                "image_id": image_id,
                **{key: value for key, value in metrics.items() if key != "tp_confidences"},
            }
        )

    precision = totals["tp"] / (totals["tp"] + totals["fp"]) if totals["tp"] + totals["fp"] else 0.0
    recall = totals["tp"] / (totals["tp"] + totals["fn"]) if totals["tp"] + totals["fn"] else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    summary = {
        **totals,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_tp_confidence": (
            float(np.mean(all_tp_confidences)) if all_tp_confidences else 0.0
        ),
        "image_count": len(image_ids),
        "ground_truth_count": totals["tp"] + totals["fn"],
        "prediction_count": totals["tp"] + totals["fp"],
    }
    return summary, details


def canonical_nms(
    predictions: list[dict[str, Any]],
    confidence_threshold: float,
    nms_iou: float,
) -> list[dict[str, Any]]:
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pred in predictions:
        if float(pred["confidence"]) >= confidence_threshold:
            by_category[pred["category"]].append(pred)

    kept: list[dict[str, Any]] = []
    for category_predictions in by_category.values():
        boxes = torch.tensor(
            [pred["bbox_xyxy"] for pred in category_predictions],
            dtype=torch.float32,
        )
        scores = torch.tensor(
            [pred["confidence"] for pred in category_predictions],
            dtype=torch.float32,
        )
        keep_indices = nms(boxes, scores, float(nms_iou)).tolist()
        kept.extend(category_predictions[index] for index in keep_indices)
    return sorted(kept, key=lambda item: float(item["confidence"]), reverse=True)
