import pytest

from yolo_world_experiment.clear_improvement import (
    build_support_calibrated_predictions,
    fuse_prediction_sources,
    optimize_category_thresholds,
    unflip_predictions,
)


def test_unflip_predictions_restores_horizontal_coordinates():
    restored = unflip_predictions(
        [
            {
                "category": "cup",
                "confidence": 0.9,
                "bbox_xyxy": [10, 20, 30, 40],
            }
        ],
        100,
    )
    assert restored[0]["bbox_xyxy"] == [70.0, 20.0, 90.0, 40.0]


def test_fuse_prediction_sources_averages_aligned_boxes():
    fused = fuse_prediction_sources(
        [
            {
                "category": "cup",
                "confidence": 0.8,
                "bbox_xyxy": [0, 0, 10, 10],
                "inference_source": "original",
            },
            {
                "category": "cup",
                "confidence": 0.8,
                "bbox_xyxy": [2, 0, 12, 10],
                "inference_source": "flip",
            },
        ],
        0.5,
    )
    assert len(fused) == 1
    assert fused[0]["bbox_xyxy"] == pytest.approx([1, 0, 11, 10])
    assert fused[0]["fusion_support"] == 2


def test_optimize_category_thresholds_can_choose_different_values():
    predictions = {
        1: [
            {
                "category": "cup",
                "confidence": 0.2,
                "bbox_xyxy": [0, 0, 10, 10],
            },
            {
                "category": "car",
                "confidence": 0.8,
                "bbox_xyxy": [20, 20, 30, 30],
            },
            {
                "category": "car",
                "confidence": 0.3,
                "bbox_xyxy": [40, 40, 50, 50],
            },
        ]
    }
    ground_truth = {
        1: [
            {"category": "cup", "bbox_xyxy": [0, 0, 10, 10]},
            {"category": "car", "bbox_xyxy": [20, 20, 30, 30]},
        ]
    }
    thresholds, summary = optimize_category_thresholds(
        predictions,
        ground_truth,
        [1],
        ["cup", "car"],
        [0.1, 0.5],
        0.5,
    )
    assert thresholds == {"cup": 0.1, "car": 0.5}
    assert summary["f1"] == pytest.approx(1.0)


def test_support_calibration_preserves_baseline_box_and_adds_bonus():
    baseline = {
        1: [
            {
                "category": "cup",
                "confidence": 0.2,
                "bbox_xyxy": [0, 0, 10, 10],
            }
        ]
    }
    supporting = {
        "canonical_768_original": {
            1: [
                {
                    "category": "cup",
                    "confidence": 0.3,
                    "bbox_xyxy": [0, 0, 10, 10],
                }
            ]
        }
    }
    output = build_support_calibrated_predictions(
        baseline,
        supporting,
        [1],
        ["cup"],
        support_iou=0.5,
        support_bonus=0.1,
        supplement_weight=0.5,
        supplement_min_support=1,
        supplement_cluster_iou=0.5,
    )
    assert len(output[1]) == 1
    assert output[1][0]["bbox_xyxy"] == [0, 0, 10, 10]
    assert output[1][0]["confidence"] == pytest.approx(0.3)
    assert output[1][0]["support_count"] == 1
