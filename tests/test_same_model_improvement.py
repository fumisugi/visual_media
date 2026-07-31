import numpy as np
import pytest
from PIL import Image

from yolo_world_experiment.model import (
    apply_corruption,
    apply_unsharp_mask,
    apply_wiener_deconvolution,
)
from yolo_world_experiment.same_model_improvement import (
    apply_blur_aware_preprocessing,
    estimate_blur_thresholds,
    laplacian_variance,
    paired_bootstrap_mean_condition_f1_delta,
)


def _checkerboard(size: int = 64) -> Image.Image:
    yy, xx = np.indices((size, size))
    values = (((xx // 4) + (yy // 4)) % 2 * 255).astype(np.uint8)
    rgb = np.repeat(values[:, :, None], 3, axis=2)
    return Image.fromarray(rgb, mode="RGB")


def test_sharpness_score_decreases_under_gaussian_blur():
    sharp = _checkerboard()
    blurred = apply_corruption(sharp, "gaussian_blur", 2.0)
    assert laplacian_variance(sharp) > laplacian_variance(blurred)


def test_blur_thresholds_separate_synthetic_score_ranges():
    thresholds = estimate_blur_thresholds(
        {
            "clean": [100.0, 120.0],
            "blur_sigma_2": [8.0, 10.0],
            "blur_sigma_4": [1.0, 2.0],
        }
    )
    assert 10.0 < thresholds["clean_threshold"] < 100.0
    assert 2.0 < thresholds["severe_threshold"] < 8.0
    assert thresholds["clean_vs_blur_balanced_accuracy"] == pytest.approx(1.0)
    assert thresholds["sigma_2_vs_4_balanced_accuracy"] == pytest.approx(1.0)


def test_blur_aware_preprocessing_leaves_clean_image_unchanged():
    image = _checkerboard()
    output, metadata = apply_blur_aware_preprocessing(
        image,
        {
            "name": "unsharp",
            "method": "unsharp",
            "radius_factor": 1.0,
            "amount": 2.0,
        },
        {"clean_threshold": 10.0, "severe_threshold": 3.0},
    )
    assert metadata["detected_state"] == "clean"
    assert not metadata["changed"]
    assert np.array_equal(np.asarray(image), np.asarray(output))


def test_identity_policy_can_leave_detected_blur_unchanged():
    image = apply_corruption(_checkerboard(), "gaussian_blur", 2.0)
    output, metadata = apply_blur_aware_preprocessing(
        image,
        {"name": "identity", "method": "identity"},
        {"clean_threshold": 1e9, "severe_threshold": -1.0},
    )
    assert metadata["detected_state"] == "blur_sigma_2"
    assert not metadata["changed"]
    assert np.array_equal(np.asarray(image), np.asarray(output))


def test_classical_restoration_preserves_shape_and_mode():
    image = apply_corruption(_checkerboard(), "gaussian_blur", 2.0)
    unsharp = apply_unsharp_mask(image, radius=2.0, amount=2.0)
    wiener = apply_wiener_deconvolution(
        image,
        blur_sigma=2.0,
        regularization=0.01,
        blend=0.75,
    )
    for output in (unsharp, wiener):
        assert output.mode == "RGB"
        assert output.size == image.size


def test_paired_bootstrap_mean_condition_delta_uses_both_conditions():
    baseline = {
        "mild": [
            {"image_id": 1, "tp": 0, "fp": 0, "fn": 1},
            {"image_id": 2, "tp": 0, "fp": 0, "fn": 1},
        ],
        "severe": [
            {"image_id": 1, "tp": 0, "fp": 0, "fn": 1},
            {"image_id": 2, "tp": 0, "fp": 0, "fn": 1},
        ],
    }
    candidate = {
        condition: [
            {"image_id": 1, "tp": 1, "fp": 0, "fn": 0},
            {"image_id": 2, "tp": 1, "fp": 0, "fn": 0},
        ]
        for condition in baseline
    }
    result = paired_bootstrap_mean_condition_f1_delta(
        baseline,
        candidate,
        samples=100,
        seed=7,
    )
    assert result["estimate"] == pytest.approx(1.0)
    assert result["ci_low"] == pytest.approx(1.0)
