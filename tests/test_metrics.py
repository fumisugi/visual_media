import pytest

from yolo_world_experiment.metrics import (
    box_iou,
    canonical_nms,
    evaluate_records,
    match_predictions,
)


def test_box_iou_identical_and_disjoint():
    assert box_iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)
    assert box_iou([0, 0, 10, 10], [20, 20, 30, 30]) == pytest.approx(0.0)


def test_greedy_matching_counts_tp_fp_fn_by_category():
    ground_truth = [
        {"category": "cup", "bbox_xyxy": [0, 0, 10, 10]},
        {"category": "car", "bbox_xyxy": [20, 20, 40, 40]},
    ]
    predictions = [
        {"category": "cup", "bbox_xyxy": [0, 0, 10, 10], "confidence": 0.9},
        {"category": "cup", "bbox_xyxy": [0, 0, 9, 9], "confidence": 0.8},
        {"category": "car", "bbox_xyxy": [60, 60, 80, 80], "confidence": 0.7},
    ]
    result = match_predictions(predictions, ground_truth, 0.25, 0.5)
    assert result["tp"] == 1
    assert result["fp"] == 2
    assert result["fn"] == 1
    assert result["precision"] == pytest.approx(1 / 3)
    assert result["recall"] == pytest.approx(1 / 2)
    assert result["f1"] == pytest.approx(0.4)


def test_confidence_threshold_excludes_low_score_prediction():
    result = match_predictions(
        [{"category": "cup", "bbox_xyxy": [0, 0, 10, 10], "confidence": 0.2}],
        [{"category": "cup", "bbox_xyxy": [0, 0, 10, 10]}],
        0.25,
        0.5,
    )
    assert result["tp"] == 0
    assert result["fp"] == 0
    assert result["fn"] == 1


def test_canonical_nms_merges_synonym_boxes_but_not_categories():
    predictions = [
        {"category": "cup", "prompt": "cup", "bbox_xyxy": [0, 0, 10, 10], "confidence": 0.9},
        {"category": "cup", "prompt": "mug", "bbox_xyxy": [0, 0, 9, 9], "confidence": 0.8},
        {"category": "car", "prompt": "car", "bbox_xyxy": [0, 0, 10, 10], "confidence": 0.7},
    ]
    kept = canonical_nms(predictions, 0.25, 0.5)
    assert len(kept) == 2
    assert {item["category"] for item in kept} == {"cup", "car"}


def test_evaluate_records_micro_aggregates_across_images():
    predictions = {
        1: [{"category": "cup", "bbox_xyxy": [0, 0, 10, 10], "confidence": 0.9}],
        2: [{"category": "car", "bbox_xyxy": [0, 0, 5, 5], "confidence": 0.8}],
    }
    ground_truth = {
        1: [{"category": "cup", "bbox_xyxy": [0, 0, 10, 10]}],
        2: [{"category": "car", "bbox_xyxy": [20, 20, 30, 30]}],
    }
    summary, details = evaluate_records(predictions, ground_truth, [1, 2], 0.25, 0.5)
    assert summary["tp"] == 1
    assert summary["fp"] == 1
    assert summary["fn"] == 1
    assert summary["f1"] == pytest.approx(0.5)
    assert len(details) == 2
