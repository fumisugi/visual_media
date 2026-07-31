from __future__ import annotations

import pytest
import torch

from yolo_world_experiment.model import weighted_spherical_prompt_mean
from yolo_world_experiment.prompt_prototype_improvement import (
    build_canonical_anchored_prompt_groups,
)


def test_weighted_spherical_prompt_mean_is_unit_normalized() -> None:
    embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    result = weighted_spherical_prompt_mean(
        embeddings, torch.tensor([1.0, 1.0])
    )
    assert torch.linalg.vector_norm(result).item() == pytest.approx(1.0)
    assert result.tolist() == pytest.approx(
        [2**-0.5, 2**-0.5]
    )


def test_weighted_spherical_prompt_mean_respects_anchor_weight() -> None:
    embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    result = weighted_spherical_prompt_mean(
        embeddings, torch.tensor([0.75, 0.25])
    )
    assert result[0].item() > result[1].item()


def test_weighted_spherical_prompt_mean_rejects_bad_weights() -> None:
    with pytest.raises(ValueError):
        weighted_spherical_prompt_mean(
            torch.ones((2, 3)), torch.tensor([1.0])
        )
    with pytest.raises(ValueError):
        weighted_spherical_prompt_mean(
            torch.ones((2, 3)), torch.tensor([0.0, 0.0])
        )


def test_canonical_anchored_groups_leave_canonical_unchanged() -> None:
    groups, weights = build_canonical_anchored_prompt_groups(
        ["car", "cup"],
        ["car", "mug"],
        0.75,
    )
    assert groups == [["car"], ["mug", "cup"]]
    assert weights == [[1.0], [0.25, 0.75]]
