import pytest

from yolo_world_experiment.detection_metrics import (
    bootstrap_metric_intervals,
    evaluate_coco_detection,
    paired_bootstrap_delta,
)


def _manifest():
    return {
        "images": [
            {
                "image_id": 1,
                "file_name": "one.jpg",
                "width": 100,
                "height": 100,
                "split": "test",
                "annotations": [
                    {
                        "annotation_id": 1,
                        "category": "cup",
                        "category_id": 47,
                        "bbox_xyxy": [10, 10, 50, 50],
                        "area": 1600,
                    }
                ],
            }
        ]
    }


def test_coco_metrics_are_one_for_perfect_detection():
    predictions = {
        1: [
            {
                "category": "cup",
                "bbox_xyxy": [10, 10, 50, 50],
                "confidence": 0.9,
            }
        ]
    }
    summary, categories, curve = evaluate_coco_detection(
        _manifest(),
        predictions,
        [1],
        {"cup": {"coco_id": 47}},
    )
    assert summary["ap"] == pytest.approx(1.0)
    assert summary["ap50"] == pytest.approx(1.0)
    assert summary["ap75"] == pytest.approx(1.0)
    assert categories[0]["ap50"] == pytest.approx(1.0)
    assert curve[0]["precision"] == pytest.approx(1.0)


def test_coco_metrics_are_zero_for_missed_detection():
    summary, categories, _ = evaluate_coco_detection(
        _manifest(),
        {1: []},
        [1],
        {"cup": {"coco_id": 47}},
    )
    assert summary["ap"] == pytest.approx(0.0)
    assert summary["ar_100"] == pytest.approx(0.0)
    assert categories[0]["ap50"] == pytest.approx(0.0)


def test_bootstrap_intervals_and_paired_delta_are_deterministic():
    baseline = [
        {"image_id": 1, "tp": 1, "fp": 1, "fn": 0},
        {"image_id": 2, "tp": 0, "fp": 0, "fn": 1},
    ]
    candidate = [
        {"image_id": 1, "tp": 1, "fp": 0, "fn": 0},
        {"image_id": 2, "tp": 1, "fp": 0, "fn": 0},
    ]
    first = bootstrap_metric_intervals(baseline, 100, 7)
    second = bootstrap_metric_intervals(baseline, 100, 7)
    assert first == second
    f1 = next(row for row in first if row["metric"] == "f1")
    assert f1["estimate"] == pytest.approx(0.5)

    deltas = paired_bootstrap_delta(baseline, candidate, 100, 7)
    f1_delta = next(row for row in deltas if row["metric"] == "f1")
    assert f1_delta["estimate"] == pytest.approx(0.5)
    assert f1_delta["ci_low"] >= 0.0
