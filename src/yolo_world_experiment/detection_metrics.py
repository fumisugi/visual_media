from __future__ import annotations

import contextlib
import io
from typing import Any

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


COCO_STAT_NAMES = (
    "ap",
    "ap50",
    "ap75",
    "ap_small",
    "ap_medium",
    "ap_large",
    "ar_1",
    "ar_10",
    "ar_100",
    "ar_small",
    "ar_medium",
    "ar_large",
)


def _mean_valid(values: np.ndarray) -> float:
    valid = values[values >= 0]
    return float(np.mean(valid)) if valid.size else float("nan")


def _build_coco_ground_truth(
    manifest: dict[str, Any],
    image_ids: list[int],
    category_specs: dict[str, dict[str, Any]],
) -> COCO:
    selected_ids = {int(image_id) for image_id in image_ids}
    categories = [
        {"id": int(spec["coco_id"]), "name": name, "supercategory": "target"}
        for name, spec in category_specs.items()
    ]
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    for entry in manifest["images"]:
        image_id = int(entry["image_id"])
        if image_id not in selected_ids:
            continue
        images.append(
            {
                "id": image_id,
                "file_name": entry["file_name"],
                "width": int(entry["width"]),
                "height": int(entry["height"]),
            }
        )
        for ann in entry["annotations"]:
            x1, y1, x2, y2 = [float(value) for value in ann["bbox_xyxy"]]
            width = max(0.0, x2 - x1)
            height = max(0.0, y2 - y1)
            annotations.append(
                {
                    "id": int(ann["annotation_id"]),
                    "image_id": image_id,
                    "category_id": int(ann["category_id"]),
                    "bbox": [x1, y1, width, height],
                    "area": float(ann.get("area", width * height)),
                    "iscrowd": 0,
                }
            )

    coco = COCO()
    coco.dataset = {
        "info": {"description": "Fixed YOLO-World COCO subset"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    with contextlib.redirect_stdout(io.StringIO()):
        coco.createIndex()
    return coco


def _to_coco_results(
    predictions_by_image: dict[int, list[dict[str, Any]]],
    image_ids: list[int],
    category_specs: dict[str, dict[str, Any]],
    confidence_threshold: float,
) -> list[dict[str, Any]]:
    category_ids = {
        name: int(spec["coco_id"]) for name, spec in category_specs.items()
    }
    results: list[dict[str, Any]] = []
    for image_id in image_ids:
        predictions = predictions_by_image.get(
            int(image_id),
            predictions_by_image.get(str(image_id), []),  # type: ignore[arg-type]
        )
        for prediction in predictions:
            category = str(prediction["category"])
            score = float(prediction["confidence"])
            if category not in category_ids or score < confidence_threshold:
                continue
            x1, y1, x2, y2 = [
                float(value) for value in prediction["bbox_xyxy"]
            ]
            width = max(0.0, x2 - x1)
            height = max(0.0, y2 - y1)
            if width <= 0 or height <= 0:
                continue
            results.append(
                {
                    "image_id": int(image_id),
                    "category_id": category_ids[category],
                    "bbox": [x1, y1, width, height],
                    "score": score,
                }
            )
    return results


def evaluate_coco_detection(
    manifest: dict[str, Any],
    predictions_by_image: dict[int, list[dict[str, Any]]],
    image_ids: list[int],
    category_specs: dict[str, dict[str, Any]],
    confidence_threshold: float = 0.0,
) -> tuple[dict[str, float], list[dict[str, Any]], list[dict[str, Any]]]:
    """Compute standard COCO AP/AR and a macro-averaged IoU=0.50 PR curve.

    The ground truth is the experiment manifest, not every COCO annotation in
    the source image. This keeps the metric universe identical to the existing
    TP/FP/FN evaluation, including its minimum-area selection rule.
    """
    coco_gt = _build_coco_ground_truth(manifest, image_ids, category_specs)
    results = _to_coco_results(
        predictions_by_image,
        image_ids,
        category_specs,
        confidence_threshold,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        if results:
            coco_dt = coco_gt.loadRes(results)
        else:
            coco_dt = COCO()
            coco_dt.dataset = {
                "info": dict(coco_gt.dataset.get("info", {})),
                "licenses": [],
                "images": list(coco_gt.dataset["images"]),
                "annotations": [],
                "categories": list(coco_gt.dataset["categories"]),
            }
            coco_dt.createIndex()
        evaluator = COCOeval(coco_gt, coco_dt, "bbox")
        evaluator.params.imgIds = [int(image_id) for image_id in image_ids]
        evaluator.params.catIds = [
            int(spec["coco_id"]) for spec in category_specs.values()
        ]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

    summary = {
        name: (float(evaluator.stats[index]) if evaluator.stats[index] >= 0 else float("nan"))
        for index, name in enumerate(COCO_STAT_NAMES)
    }
    summary.update(
        {
            "image_count": float(len(image_ids)),
            "ground_truth_count": float(len(coco_gt.dataset["annotations"])),
            "prediction_count": float(len(results)),
        }
    )

    precision = evaluator.eval["precision"]
    recall = evaluator.eval["recall"]
    iou50_index = int(np.argmin(np.abs(evaluator.params.iouThrs - 0.50)))
    iou75_index = int(np.argmin(np.abs(evaluator.params.iouThrs - 0.75)))
    all_area_index = 0
    max_det_index = len(evaluator.params.maxDets) - 1

    category_rows: list[dict[str, Any]] = []
    for category_index, category in enumerate(category_specs):
        category_rows.append(
            {
                "category": category,
                "ap": _mean_valid(
                    precision[:, :, category_index, all_area_index, max_det_index]
                ),
                "ap50": _mean_valid(
                    precision[
                        iou50_index,
                        :,
                        category_index,
                        all_area_index,
                        max_det_index,
                    ]
                ),
                "ap75": _mean_valid(
                    precision[
                        iou75_index,
                        :,
                        category_index,
                        all_area_index,
                        max_det_index,
                    ]
                ),
                "ar_100": _mean_valid(
                    recall[:, category_index, all_area_index, max_det_index]
                ),
            }
        )

    pr_rows: list[dict[str, Any]] = []
    for recall_index, recall_threshold in enumerate(evaluator.params.recThrs):
        values = precision[
            iou50_index,
            recall_index,
            :,
            all_area_index,
            max_det_index,
        ]
        pr_rows.append(
            {
                "recall": float(recall_threshold),
                "precision": _mean_valid(values),
                "iou": 0.50,
            }
        )
    return summary, category_rows, pr_rows


def _metrics_from_counts(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    precision = np.divide(
        tp,
        tp + fp,
        out=np.zeros_like(tp, dtype=float),
        where=(tp + fp) > 0,
    )
    recall = np.divide(
        tp,
        tp + fn,
        out=np.zeros_like(tp, dtype=float),
        where=(tp + fn) > 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision, dtype=float),
        where=(precision + recall) > 0,
    )
    return precision, recall, f1


def bootstrap_metric_intervals(
    per_image_rows: list[dict[str, Any]],
    samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Image-level non-parametric bootstrap CIs for micro P/R/F1."""
    if not per_image_rows:
        raise ValueError("per_image_rows must not be empty")
    counts = np.asarray(
        [[row["tp"], row["fp"], row["fn"]] for row in per_image_rows],
        dtype=float,
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(counts), size=(int(samples), len(counts)))
    totals = counts[indices].sum(axis=1)
    values = _metrics_from_counts(totals[:, 0], totals[:, 1], totals[:, 2])
    point_totals = counts.sum(axis=0)
    points = _metrics_from_counts(
        point_totals[0:1], point_totals[1:2], point_totals[2:3]
    )
    rows = []
    for metric, point, bootstrapped in zip(
        ("precision", "recall", "f1"),
        points,
        values,
    ):
        rows.append(
            {
                "metric": metric,
                "estimate": float(point[0]),
                "ci_low": float(np.quantile(bootstrapped, 0.025)),
                "ci_high": float(np.quantile(bootstrapped, 0.975)),
                "bootstrap_samples": int(samples),
            }
        )
    return rows


def paired_bootstrap_delta(
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Paired image-bootstrap CIs for candidate minus baseline P/R/F1."""
    baseline = {int(row["image_id"]): row for row in baseline_rows}
    candidate = {int(row["image_id"]): row for row in candidate_rows}
    image_ids = sorted(set(baseline) & set(candidate))
    if not image_ids:
        raise ValueError("baseline and candidate have no common image IDs")
    base_counts = np.asarray(
        [[baseline[i]["tp"], baseline[i]["fp"], baseline[i]["fn"]] for i in image_ids],
        dtype=float,
    )
    cand_counts = np.asarray(
        [[candidate[i]["tp"], candidate[i]["fp"], candidate[i]["fn"]] for i in image_ids],
        dtype=float,
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(image_ids), size=(int(samples), len(image_ids)))
    base_totals = base_counts[indices].sum(axis=1)
    cand_totals = cand_counts[indices].sum(axis=1)
    base_metrics = _metrics_from_counts(
        base_totals[:, 0], base_totals[:, 1], base_totals[:, 2]
    )
    cand_metrics = _metrics_from_counts(
        cand_totals[:, 0], cand_totals[:, 1], cand_totals[:, 2]
    )
    base_point = _metrics_from_counts(
        base_counts[:, 0].sum(keepdims=True),
        base_counts[:, 1].sum(keepdims=True),
        base_counts[:, 2].sum(keepdims=True),
    )
    cand_point = _metrics_from_counts(
        cand_counts[:, 0].sum(keepdims=True),
        cand_counts[:, 1].sum(keepdims=True),
        cand_counts[:, 2].sum(keepdims=True),
    )
    rows = []
    for metric, base_values, cand_values, base_value, cand_value in zip(
        ("precision", "recall", "f1"),
        base_metrics,
        cand_metrics,
        base_point,
        cand_point,
    ):
        delta = cand_values - base_values
        rows.append(
            {
                "metric": metric,
                "estimate": float(cand_value[0] - base_value[0]),
                "ci_low": float(np.quantile(delta, 0.025)),
                "ci_high": float(np.quantile(delta, 0.975)),
                "bootstrap_samples": int(samples),
            }
        )
    return rows
