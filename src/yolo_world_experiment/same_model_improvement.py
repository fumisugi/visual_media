from __future__ import annotations

import json
import hashlib
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from .coco import ground_truth_by_image
from .config import category_names, prompt_set, resolve_path
from .detection_metrics import (
    evaluate_coco_detection,
    paired_bootstrap_delta,
)
from .metrics import evaluate_records
from .model import (
    YoloWorldPredictor,
    apply_corruption,
    apply_unsharp_mask,
    apply_wiener_deconvolution,
)
from .prompt_study import _write_prediction_file


def laplacian_variance(image: Image.Image) -> float:
    """Return a deterministic no-reference sharpness score."""
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    laplacian = -4.0 * gray.copy()
    laplacian[1:, :] += gray[:-1, :]
    laplacian[:-1, :] += gray[1:, :]
    laplacian[:, 1:] += gray[:, :-1]
    laplacian[:, :-1] += gray[:, 1:]
    if min(gray.shape) > 2:
        laplacian = laplacian[1:-1, 1:-1]
    return float(np.var(laplacian))


def _best_low_class_threshold(
    low_values: list[float],
    high_values: list[float],
) -> tuple[float, float]:
    """Choose a threshold for classifying low-valued observations."""
    values = sorted(set(float(value) for value in low_values + high_values))
    candidates = [
        values[0] - 1e-6,
        *[
            (left + right) / 2.0
            for left, right in zip(values[:-1], values[1:])
        ],
        values[-1] + 1e-6,
    ]
    best_threshold = candidates[0]
    best_accuracy = -1.0
    for threshold in candidates:
        low_accuracy = np.mean(
            [float(value) <= threshold for value in low_values]
        )
        high_accuracy = np.mean(
            [float(value) > threshold for value in high_values]
        )
        balanced_accuracy = float((low_accuracy + high_accuracy) / 2.0)
        if (
            balanced_accuracy > best_accuracy
            or (
                np.isclose(balanced_accuracy, best_accuracy)
                and threshold < best_threshold
            )
        ):
            best_threshold = float(threshold)
            best_accuracy = balanced_accuracy
    return best_threshold, best_accuracy


def estimate_blur_thresholds(
    scores_by_condition: dict[str, list[float]],
) -> dict[str, float]:
    blurred = [
        *scores_by_condition["blur_sigma_2"],
        *scores_by_condition["blur_sigma_4"],
    ]
    clean_threshold, clean_accuracy = _best_low_class_threshold(
        blurred,
        scores_by_condition["clean"],
    )
    severe_threshold, severity_accuracy = _best_low_class_threshold(
        scores_by_condition["blur_sigma_4"],
        scores_by_condition["blur_sigma_2"],
    )
    return {
        "clean_threshold": clean_threshold,
        "severe_threshold": severe_threshold,
        "clean_vs_blur_balanced_accuracy": clean_accuracy,
        "sigma_2_vs_4_balanced_accuracy": severity_accuracy,
    }


def apply_blur_aware_preprocessing(
    image: Image.Image,
    candidate: dict[str, Any],
    thresholds: dict[str, float],
) -> tuple[Image.Image, dict[str, Any]]:
    """Apply a frozen classical correction only when blur is detected."""
    score = laplacian_variance(image)
    if score > float(thresholds["clean_threshold"]):
        return image.copy(), {
            "sharpness_score": score,
            "detected_state": "clean",
            "changed": False,
        }
    estimated_sigma = (
        4.0
        if score <= float(thresholds["severe_threshold"])
        else 2.0
    )
    method = str(candidate["method"])
    if method == "identity":
        output = image.copy()
    elif method == "unsharp":
        output = apply_unsharp_mask(
            image,
            radius=estimated_sigma * float(candidate["radius_factor"]),
            amount=float(candidate["amount"]),
        )
    elif method == "wiener":
        output = apply_wiener_deconvolution(
            image,
            blur_sigma=estimated_sigma,
            regularization=float(candidate["regularization"]),
            blend=float(candidate["blend"]),
        )
    else:
        raise ValueError(f"Unsupported blur correction method: {method}")
    return output, {
        "sharpness_score": score,
        "detected_state": f"blur_sigma_{int(estimated_sigma)}",
        "changed": method != "identity",
    }


def paired_bootstrap_mean_condition_f1_delta(
    baseline_by_condition: dict[str, list[dict[str, Any]]],
    candidate_by_condition: dict[str, list[dict[str, Any]]],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    """Bootstrap the paired delta in mean F1 across fixed conditions."""

    def f1_from_counts(counts: np.ndarray) -> np.ndarray:
        precision = np.divide(
            counts[:, 0],
            counts[:, 0] + counts[:, 1],
            out=np.zeros(len(counts), dtype=float),
            where=(counts[:, 0] + counts[:, 1]) > 0,
        )
        recall = np.divide(
            counts[:, 0],
            counts[:, 0] + counts[:, 2],
            out=np.zeros(len(counts), dtype=float),
            where=(counts[:, 0] + counts[:, 2]) > 0,
        )
        return np.divide(
            2.0 * precision * recall,
            precision + recall,
            out=np.zeros(len(counts), dtype=float),
            where=(precision + recall) > 0,
        )

    conditions = sorted(baseline_by_condition)
    if conditions != sorted(candidate_by_condition):
        raise ValueError("Baseline and candidate conditions differ")
    baseline_maps = {
        condition: {
            int(row["image_id"]): row
            for row in baseline_by_condition[condition]
        }
        for condition in conditions
    }
    candidate_maps = {
        condition: {
            int(row["image_id"]): row
            for row in candidate_by_condition[condition]
        }
        for condition in conditions
    }
    image_ids = sorted(
        set.intersection(
            *(
                set(baseline_maps[condition])
                & set(candidate_maps[condition])
                for condition in conditions
            )
        )
    )
    if not image_ids:
        raise ValueError("No paired images across conditions")
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, len(image_ids), size=(int(samples), len(image_ids))
    )
    baseline_f1 = []
    candidate_f1 = []
    baseline_point = []
    candidate_point = []
    for condition in conditions:
        base_counts = np.asarray(
            [
                [
                    baseline_maps[condition][image_id][key]
                    for key in ("tp", "fp", "fn")
                ]
                for image_id in image_ids
            ],
            dtype=float,
        )
        candidate_counts = np.asarray(
            [
                [
                    candidate_maps[condition][image_id][key]
                    for key in ("tp", "fp", "fn")
                ]
                for image_id in image_ids
            ],
            dtype=float,
        )
        baseline_f1.append(
            f1_from_counts(base_counts[indices].sum(axis=1))
        )
        candidate_f1.append(
            f1_from_counts(candidate_counts[indices].sum(axis=1))
        )
        baseline_point.append(
            float(f1_from_counts(base_counts.sum(axis=0, keepdims=True))[0])
        )
        candidate_point.append(
            float(
                f1_from_counts(
                    candidate_counts.sum(axis=0, keepdims=True)
                )[0]
            )
        )
    bootstrap_delta = np.mean(candidate_f1, axis=0) - np.mean(
        baseline_f1, axis=0
    )
    estimate = float(np.mean(candidate_point) - np.mean(baseline_point))
    return {
        "metric": "mean_condition_f1",
        "estimate": estimate,
        "ci_low": float(np.quantile(bootstrap_delta, 0.025)),
        "ci_high": float(np.quantile(bootstrap_delta, 0.975)),
        "bootstrap_samples": int(samples),
        "image_count": len(image_ids),
        "conditions": conditions,
    }


class SameModelImprovementStudy:
    """Failure-aware improvements with the Small detector held fixed."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        if str(config["model"]["name"]) != "yolov8s-worldv2.pt":
            raise ValueError(
                "Same-model improvement requires yolov8s-worldv2.pt"
            )
        self.categories = category_names(config)
        self.images_dir = resolve_path(config, config["paths"]["coco_images"])
        self.results_dir = resolve_path(
            config, config["paths"]["same_model_results_dir"]
        )
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.predictor = YoloWorldPredictor(config, self.categories)
        self.predictor.set_prompts(prompt_set(config, 0))
        self.confidence = float(config["evaluation"]["confidence"])
        self.match_iou = float(config["evaluation"]["match_iou"])

    def _read_manifest(self, key: str) -> dict[str, Any]:
        path = resolve_path(self.config, self.config["paths"][key])
        return json.loads(path.read_text(encoding="utf-8"))

    def _development_manifests(self) -> list[dict[str, Any]]:
        return [
            self._read_manifest("subset_manifest"),
            self._read_manifest("holdout_manifest"),
            self._read_manifest("final_holdout_manifest"),
        ]

    def _development_manifest(self) -> dict[str, Any]:
        manifests = self._development_manifests()
        return {
            "source": "COCO 2017 val; three previously inspected folds",
            "images": [
                image
                for manifest in manifests
                for image in manifest["images"]
            ],
        }

    def _estimate_development_blur_thresholds(self) -> dict[str, float]:
        scores = {
            "clean": [],
            "blur_sigma_2": [],
            "blur_sigma_4": [],
        }
        for entry in self._development_manifest()["images"]:
            image = Image.open(
                self.images_dir / entry["file_name"]
            ).convert("RGB")
            scores["clean"].append(laplacian_variance(image))
            for sigma in (2.0, 4.0):
                blurred = apply_corruption(
                    image, "gaussian_blur", sigma
                )
                scores[f"blur_sigma_{int(sigma)}"].append(
                    laplacian_variance(blurred)
                )
        thresholds = estimate_blur_thresholds(scores)
        quantiles = {
            condition: {
                str(quantile): float(np.quantile(values, quantile))
                for quantile in (0.0, 0.1, 0.5, 0.9, 1.0)
            }
            for condition, values in scores.items()
        }
        payload = {**thresholds, "score_quantiles": quantiles}
        (self.results_dir / "blur_detector_parameters.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        return thresholds

    def _screening_entries(self) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        manifest = self._read_manifest("subset_manifest")
        entries = [
            entry
            for entry in manifest["images"]
            if entry["split"] == "validation"
        ]
        return manifest, entries

    def _predict_blur_condition(
        self,
        entries: list[dict[str, Any]],
        sigma: float,
        candidate: dict[str, Any] | None,
        thresholds: dict[str, float],
    ) -> tuple[
        dict[int, list[dict[str, Any]]],
        list[dict[str, Any]],
    ]:
        predictions: dict[int, list[dict[str, Any]]] = {}
        detector_rows: list[dict[str, Any]] = []
        for entry in entries:
            image_id = int(entry["image_id"])
            image = Image.open(
                self.images_dir / entry["file_name"]
            ).convert("RGB")
            transformed = apply_corruption(
                image, "gaussian_blur", float(sigma)
            )
            if candidate is not None:
                transformed, detector = apply_blur_aware_preprocessing(
                    transformed, candidate, thresholds
                )
                detector_rows.append(
                    {
                        "candidate": candidate["name"],
                        "true_condition": f"blur_sigma_{int(sigma)}",
                        "image_id": image_id,
                        **detector,
                    }
                )
            predictions[image_id] = self.predictor.predict(transformed)
        return predictions, detector_rows

    def run_screening(self) -> dict[str, Any]:
        manifest, entries = self._screening_entries()
        image_ids = [int(entry["image_id"]) for entry in entries]
        ground_truth = ground_truth_by_image(manifest)
        thresholds = self._estimate_development_blur_thresholds()
        candidates = [
            dict(item)
            for item in self.config["same_model_improvement"][
                "blur_candidates"
            ]
        ]
        sigmas = [
            float(value)
            for value in self.config["same_model_improvement"][
                "blur_sigmas"
            ]
        ]

        records: dict[str, dict[int, list[dict[str, Any]]]] = {}
        detector_rows: list[dict[str, Any]] = []
        baseline_by_sigma: dict[float, dict[str, Any]] = {}
        rows: list[dict[str, Any]] = []
        for sigma in sigmas:
            condition = f"baseline_blur_sigma_{int(sigma)}"
            predictions, _ = self._predict_blur_condition(
                entries, sigma, None, thresholds
            )
            records[condition] = predictions
            summary, _ = evaluate_records(
                predictions,
                ground_truth,
                image_ids,
                self.confidence,
                self.match_iou,
            )
            baseline_by_sigma[sigma] = summary
            rows.append(
                {
                    "candidate": "baseline",
                    "condition": f"blur_sigma_{int(sigma)}",
                    **summary,
                }
            )

        candidate_aggregate: list[dict[str, Any]] = []
        for candidate in candidates:
            condition_summaries = []
            for sigma in sigmas:
                condition = (
                    f"{candidate['name']}_blur_sigma_{int(sigma)}"
                )
                predictions, condition_detector_rows = (
                    self._predict_blur_condition(
                        entries, sigma, candidate, thresholds
                    )
                )
                records[condition] = predictions
                detector_rows.extend(condition_detector_rows)
                summary, _ = evaluate_records(
                    predictions,
                    ground_truth,
                    image_ids,
                    self.confidence,
                    self.match_iou,
                )
                condition_summaries.append(summary)
                rows.append(
                    {
                        "candidate": candidate["name"],
                        "condition": f"blur_sigma_{int(sigma)}",
                        **summary,
                    }
                )
            mean_f1 = float(
                np.mean([summary["f1"] for summary in condition_summaries])
            )
            baseline_mean_f1 = float(
                np.mean(
                    [baseline_by_sigma[sigma]["f1"] for sigma in sigmas]
                )
            )
            candidate_aggregate.append(
                {
                    "candidate": candidate["name"],
                    "mean_blur_f1": mean_f1,
                    "baseline_mean_blur_f1": baseline_mean_f1,
                    "mean_blur_f1_gain": mean_f1 - baseline_mean_f1,
                    "method": candidate["method"],
                }
            )

        aggregate = pd.DataFrame(candidate_aggregate).sort_values(
            ["mean_blur_f1_gain", "mean_blur_f1"],
            ascending=False,
        )
        keep_count = int(
            self.config["same_model_improvement"][
                "screening_candidates_to_keep"
            ]
        )
        selected_names = aggregate.head(keep_count)["candidate"].tolist()
        selected_candidates = [
            candidate
            for candidate in candidates
            if candidate["name"] in selected_names
        ]
        pd.DataFrame(rows).to_csv(
            self.results_dir / "blur_screening_metrics.csv", index=False
        )
        aggregate.to_csv(
            self.results_dir / "blur_screening_ranking.csv", index=False
        )
        pd.DataFrame(detector_rows).to_csv(
            self.results_dir / "blur_screening_detector_rows.csv",
            index=False,
        )
        _write_prediction_file(
            self.results_dir / "blur_screening_predictions.json",
            records,
        )
        payload = {
            "model": self.config["model"]["name"],
            "prompts": prompt_set(self.config, 0),
            "image_size": int(self.config["model"]["image_size"]),
            "confidence_threshold": self.confidence,
            "screening_scope": "original validation split; 30 images",
            "blur_detector": thresholds,
            "selected_candidates": selected_candidates,
        }
        (self.results_dir / "blur_screening_selection.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        return payload

    def run_development(self) -> dict[str, Any]:
        selection_path = (
            self.results_dir / "blur_screening_selection.json"
        )
        if not selection_path.exists():
            raise FileNotFoundError(
                "Run same-model-blur-screen before development"
            )
        screening = json.loads(selection_path.read_text(encoding="utf-8"))
        candidates = [
            dict(candidate)
            for candidate in screening["selected_candidates"]
        ]
        candidates = [
            {"name": "identity", "method": "identity"},
            *candidates,
        ]
        thresholds = {
            key: float(screening["blur_detector"][key])
            for key in ("clean_threshold", "severe_threshold")
        }
        manifest = self._development_manifest()
        entries = manifest["images"]
        image_ids = [int(entry["image_id"]) for entry in entries]
        ground_truth = ground_truth_by_image(manifest)
        sigmas = [
            float(value)
            for value in self.config["same_model_improvement"][
                "blur_sigmas"
            ]
        ]

        records: dict[str, dict[int, list[dict[str, Any]]]] = {}
        detector_state: dict[
            tuple[float, int], str
        ] = {}
        baseline_predictions: dict[
            float, dict[int, list[dict[str, Any]]]
        ] = {}
        baseline_summaries: dict[float, dict[str, Any]] = {}
        baseline_details: dict[str, list[dict[str, Any]]] = {}
        metric_rows: list[dict[str, Any]] = []
        for sigma in sigmas:
            predictions, _ = self._predict_blur_condition(
                entries, sigma, None, thresholds
            )
            baseline_predictions[sigma] = predictions
            records[f"baseline_blur_sigma_{int(sigma)}"] = predictions
            summary, details = evaluate_records(
                predictions,
                ground_truth,
                image_ids,
                self.confidence,
                self.match_iou,
            )
            baseline_summaries[sigma] = summary
            baseline_details[f"blur_sigma_{int(sigma)}"] = details
            metric_rows.append(
                {
                    "scope": "development_300",
                    "policy": "baseline",
                    "fold": "all",
                    "condition": f"blur_sigma_{int(sigma)}",
                    **summary,
                }
            )

        predictions_by_candidate: dict[
            str, dict[float, dict[int, list[dict[str, Any]]]]
        ] = {}
        for candidate_index, candidate in enumerate(candidates):
            predictions_by_candidate[candidate["name"]] = {}
            for sigma in sigmas:
                predictions, detector_rows = self._predict_blur_condition(
                    entries, sigma, candidate, thresholds
                )
                predictions_by_candidate[candidate["name"]][sigma] = (
                    predictions
                )
                records[
                    f"{candidate['name']}_blur_sigma_{int(sigma)}"
                ] = predictions
                if candidate_index == 0:
                    for row in detector_rows:
                        detector_state[
                            (sigma, int(row["image_id"]))
                        ] = str(row["detected_state"])

        manifest_keys = [
            "subset_manifest",
            "holdout_manifest",
            "final_holdout_manifest",
        ]
        fold_ids = {
            key: {
                int(entry["image_id"])
                for entry in self._read_manifest(key)["images"]
            }
            for key in manifest_keys
        }
        policy_rows: list[dict[str, Any]] = []
        policy_payloads: dict[str, dict[str, Any]] = {}
        details_by_policy: dict[
            str, dict[str, list[dict[str, Any]]]
        ] = {}
        for mild_candidate, severe_candidate in product(
            candidates, candidates
        ):
            policy_name = (
                f"mild={mild_candidate['name']}__"
                f"severe={severe_candidate['name']}"
            )
            condition_predictions: dict[
                float, dict[int, list[dict[str, Any]]]
            ] = {}
            condition_summaries: dict[float, dict[str, Any]] = {}
            condition_details: dict[str, list[dict[str, Any]]] = {}
            for sigma in sigmas:
                predictions = {}
                for image_id in image_ids:
                    state = detector_state[(sigma, image_id)]
                    source = (
                        severe_candidate
                        if state == "blur_sigma_4"
                        else mild_candidate
                    )
                    predictions[image_id] = predictions_by_candidate[
                        source["name"]
                    ][sigma][image_id]
                condition_predictions[sigma] = predictions
                summary, details = evaluate_records(
                    predictions,
                    ground_truth,
                    image_ids,
                    self.confidence,
                    self.match_iou,
                )
                condition_summaries[sigma] = summary
                condition_details[f"blur_sigma_{int(sigma)}"] = details
                metric_rows.append(
                    {
                        "scope": "development_300",
                        "policy": policy_name,
                        "fold": "all",
                        "condition": f"blur_sigma_{int(sigma)}",
                        **summary,
                    }
                )

            fold_gains: list[float] = []
            for fold, ids in fold_ids.items():
                fold_mean_f1 = []
                fold_baseline_mean_f1 = []
                fold_image_ids = sorted(ids)
                for sigma in sigmas:
                    candidate_summary, _ = evaluate_records(
                        condition_predictions[sigma],
                        ground_truth,
                        fold_image_ids,
                        self.confidence,
                        self.match_iou,
                    )
                    baseline_summary, _ = evaluate_records(
                        baseline_predictions[sigma],
                        ground_truth,
                        fold_image_ids,
                        self.confidence,
                        self.match_iou,
                    )
                    fold_mean_f1.append(float(candidate_summary["f1"]))
                    fold_baseline_mean_f1.append(
                        float(baseline_summary["f1"])
                    )
                    metric_rows.append(
                        {
                            "scope": "development_300",
                            "policy": policy_name,
                            "fold": fold,
                            "condition": f"blur_sigma_{int(sigma)}",
                            **candidate_summary,
                        }
                    )
                fold_gains.append(
                    float(
                        np.mean(fold_mean_f1)
                        - np.mean(fold_baseline_mean_f1)
                    )
                )
            bootstrap = paired_bootstrap_mean_condition_f1_delta(
                baseline_details,
                condition_details,
                samples=int(
                    self.config["evaluation"]["bootstrap_samples"]
                ),
                seed=int(self.config["seed"]) + len(policy_rows) + 101,
            )
            mean_f1 = float(
                np.mean(
                    [summary["f1"] for summary in condition_summaries.values()]
                )
            )
            baseline_mean_f1 = float(
                np.mean(
                    [
                        baseline_summaries[sigma]["f1"]
                        for sigma in sigmas
                    ]
                )
            )
            row = {
                "policy": policy_name,
                "mild_candidate": mild_candidate["name"],
                "severe_candidate": severe_candidate["name"],
                "mean_blur_f1": mean_f1,
                "baseline_mean_blur_f1": baseline_mean_f1,
                "mean_blur_f1_gain": mean_f1 - baseline_mean_f1,
                "minimum_fold_gain": float(min(fold_gains)),
                "mean_fold_gain": float(np.mean(fold_gains)),
                "f1_delta_ci_low": float(bootstrap["ci_low"]),
                "f1_delta_ci_high": float(bootstrap["ci_high"]),
            }
            policy_rows.append(row)
            policy_payloads[policy_name] = {
                "mild_candidate": mild_candidate,
                "severe_candidate": severe_candidate,
                "metrics": row,
                "bootstrap": bootstrap,
            }
            details_by_policy[policy_name] = condition_details

        ranking = pd.DataFrame(policy_rows).sort_values(
            [
                "minimum_fold_gain",
                "mean_blur_f1_gain",
                "f1_delta_ci_low",
            ],
            ascending=False,
        )
        selected_name = str(ranking.iloc[0]["policy"])
        selected = policy_payloads[selected_name]
        selected["method"] = "automatic_blur_detection_and_classical_restoration"
        selected["model"] = self.config["model"]["name"]
        selected["prompts"] = prompt_set(self.config, 0)
        selected["image_size"] = int(self.config["model"]["image_size"])
        selected["confidence_threshold"] = self.confidence
        selected["blur_detector"] = thresholds
        selected["selection_scope"] = "development_300_only"
        selected["clean_input_policy"] = "identity"
        selected["clean_f1_delta_by_construction"] = 0.0
        selected["success_criteria"] = {
            "minimum_blur_mean_f1_gain": float(
                self.config["same_model_improvement"][
                    "minimum_blur_mean_f1_gain"
                ]
            ),
            "minimum_ci_low": 0.0,
            "maximum_clean_f1_drop": float(
                self.config["same_model_improvement"][
                    "maximum_clean_f1_drop"
                ]
            ),
        }
        selected["development_gate_passed"] = bool(
            float(selected["metrics"]["mean_blur_f1_gain"])
            >= selected["success_criteria"]["minimum_blur_mean_f1_gain"]
            and float(selected["bootstrap"]["ci_low"]) > 0.0
            and float(selected["metrics"]["minimum_fold_gain"]) > 0.0
        )
        selected_path = self.results_dir / "selected_same_model_method.json"
        selected_path.write_text(
            json.dumps(selected, indent=2) + "\n", encoding="utf-8"
        )
        selected["selected_parameters_sha256"] = hashlib.sha256(
            selected_path.read_bytes()
        ).hexdigest()

        pd.DataFrame(metric_rows).to_csv(
            self.results_dir / "blur_development_metrics.csv",
            index=False,
        )
        ranking.to_csv(
            self.results_dir / "blur_policy_ranking.csv", index=False
        )
        selected_details = []
        for condition, details in details_by_policy[selected_name].items():
            selected_details.extend(
                {"condition": condition, **row} for row in details
            )
        pd.DataFrame(selected_details).to_csv(
            self.results_dir / "blur_development_selected_per_image.csv",
            index=False,
        )
        _write_prediction_file(
            self.results_dir / "blur_development_source_predictions.json",
            records,
        )
        return selected

    def _apply_selected_policy(
        self,
        image: Image.Image,
        selected: dict[str, Any],
    ) -> tuple[Image.Image, dict[str, Any]]:
        thresholds = selected["blur_detector"]
        score = laplacian_variance(image)
        if score > float(thresholds["clean_threshold"]):
            candidate = selected["mild_candidate"]
        elif score <= float(thresholds["severe_threshold"]):
            candidate = selected["severe_candidate"]
        else:
            candidate = selected["mild_candidate"]
        output, metadata = apply_blur_aware_preprocessing(
            image, candidate, thresholds
        )
        return output, {
            **metadata,
            "applied_candidate": (
                candidate["name"] if metadata["changed"] else "identity"
            ),
        }

    def run_final_holdout(self) -> dict[str, Any]:
        verdict_path = self.results_dir / "same_model_final_verdict.json"
        if verdict_path.exists():
            raise RuntimeError(
                "The same-model holdout has already been evaluated; "
                "refusing a second run"
            )
        selected_path = self.results_dir / "selected_same_model_method.json"
        if not selected_path.exists():
            raise FileNotFoundError("No frozen same-model method")
        selected_bytes = selected_path.read_bytes()
        selected = json.loads(selected_bytes)
        if not bool(selected.get("development_gate_passed")):
            raise RuntimeError("Frozen method failed the development gate")
        if selected.get("model") != "yolov8s-worldv2.pt":
            raise RuntimeError("Frozen method changes the detector model")

        manifest_path = resolve_path(
            self.config,
            self.config["paths"]["same_model_holdout_manifest"],
        )
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
        entries = manifest["images"]
        image_ids = [int(entry["image_id"]) for entry in entries]
        ground_truth = ground_truth_by_image(manifest)
        conditions = [
            ("clean", None),
            *[
                (f"blur_sigma_{int(float(sigma))}", float(sigma))
                for sigma in self.config["same_model_improvement"][
                    "blur_sigmas"
                ]
            ],
        ]

        baseline_records: dict[
            str, dict[int, list[dict[str, Any]]]
        ] = {}
        candidate_records: dict[
            str, dict[int, list[dict[str, Any]]]
        ] = {}
        detector_rows: list[dict[str, Any]] = []
        for condition, sigma in conditions:
            baseline_records[condition] = {}
            candidate_records[condition] = {}
            for entry in entries:
                image_id = int(entry["image_id"])
                image = Image.open(
                    self.images_dir / entry["file_name"]
                ).convert("RGB")
                transformed = (
                    image
                    if sigma is None
                    else apply_corruption(
                        image, "gaussian_blur", float(sigma)
                    )
                )
                baseline_predictions = self.predictor.predict(transformed)
                corrected, detector = self._apply_selected_policy(
                    transformed, selected
                )
                baseline_records[condition][image_id] = (
                    baseline_predictions
                )
                if not detector["changed"]:
                    candidate_records[condition][image_id] = (
                        baseline_predictions
                    )
                else:
                    candidate_records[condition][image_id] = (
                        self.predictor.predict(corrected)
                    )
                detector_rows.append(
                    {
                        "condition": condition,
                        "image_id": image_id,
                        **detector,
                    }
                )

        summary_rows: list[dict[str, Any]] = []
        detail_rows: list[dict[str, Any]] = []
        baseline_details: dict[str, list[dict[str, Any]]] = {}
        candidate_details: dict[str, list[dict[str, Any]]] = {}
        for condition, _ in conditions:
            for method, records, detail_store in (
                ("baseline", baseline_records, baseline_details),
                (
                    "blur_aware_wiener",
                    candidate_records,
                    candidate_details,
                ),
            ):
                summary, details = evaluate_records(
                    records[condition],
                    ground_truth,
                    image_ids,
                    self.confidence,
                    self.match_iou,
                )
                ap, _, _ = evaluate_coco_detection(
                    manifest,
                    records[condition],
                    image_ids,
                    self.config["categories"],
                    confidence_threshold=0.0,
                )
                detail_store[condition] = details
                summary_rows.append(
                    {
                        "scope": "same_model_final_holdout",
                        "method": method,
                        "condition": condition,
                        **summary,
                        "ap": ap["ap"],
                        "ap50": ap["ap50"],
                        "ap75": ap["ap75"],
                    }
                )
                detail_rows.extend(
                    {
                        "scope": "same_model_final_holdout",
                        "method": method,
                        "condition": condition,
                        **row,
                    }
                    for row in details
                )

        blur_conditions = [
            condition for condition, sigma in conditions if sigma is not None
        ]
        blur_bootstrap = paired_bootstrap_mean_condition_f1_delta(
            {
                condition: baseline_details[condition]
                for condition in blur_conditions
            },
            {
                condition: candidate_details[condition]
                for condition in blur_conditions
            },
            samples=int(self.config["evaluation"]["bootstrap_samples"]),
            seed=int(self.config["selection"]["same_model_holdout_seed"])
            + 109,
        )
        clean_deltas = paired_bootstrap_delta(
            baseline_details["clean"],
            candidate_details["clean"],
            int(self.config["evaluation"]["bootstrap_samples"]),
            int(self.config["selection"]["same_model_holdout_seed"]) + 113,
        )
        clean_f1_delta = next(
            row for row in clean_deltas if row["metric"] == "f1"
        )
        summary_frame = pd.DataFrame(summary_rows)
        summary_index = summary_frame.set_index(["method", "condition"])
        baseline_blur_mean = float(
            np.mean(
                [
                    summary_index.loc[("baseline", condition), "f1"]
                    for condition in blur_conditions
                ]
            )
        )
        candidate_blur_mean = float(
            np.mean(
                [
                    summary_index.loc[
                        ("blur_aware_wiener", condition), "f1"
                    ]
                    for condition in blur_conditions
                ]
            )
        )
        minimum_gain = float(
            self.config["same_model_improvement"][
                "minimum_blur_mean_f1_gain"
            ]
        )
        maximum_clean_drop = float(
            self.config["same_model_improvement"][
                "maximum_clean_f1_drop"
            ]
        )
        passed = bool(
            candidate_blur_mean - baseline_blur_mean >= minimum_gain
            and float(blur_bootstrap["ci_low"]) > 0.0
            and float(clean_f1_delta["estimate"]) >= -maximum_clean_drop
        )

        summary_frame.to_csv(
            self.results_dir / "same_model_final_summary.csv",
            index=False,
        )
        pd.DataFrame(detail_rows).to_csv(
            self.results_dir / "same_model_final_per_image.csv",
            index=False,
        )
        pd.DataFrame(
            [
                {"comparison": "blur_mean", **blur_bootstrap},
                *[
                    {"comparison": "clean", **row}
                    for row in clean_deltas
                ],
            ]
        ).to_csv(
            self.results_dir / "same_model_final_bootstrap.csv",
            index=False,
        )
        pd.DataFrame(detector_rows).to_csv(
            self.results_dir / "same_model_final_blur_detector.csv",
            index=False,
        )
        _write_prediction_file(
            self.results_dir / "same_model_final_predictions.json",
            {
                **{
                    f"baseline_{condition}": predictions
                    for condition, predictions in baseline_records.items()
                },
                **{
                    f"blur_aware_wiener_{condition}": predictions
                    for condition, predictions in candidate_records.items()
                },
            },
        )
        verdict = {
            "passed": passed,
            "selected_method": selected["method"],
            "model": selected["model"],
            "model_change_used": False,
            "training_used": False,
            "super_resolution_used": False,
            "selection_scope": selected["selection_scope"],
            "selection_was_frozen_before_holdout": True,
            "holdout_evaluated_once": True,
            "baseline_blur_mean_f1": baseline_blur_mean,
            "candidate_blur_mean_f1": candidate_blur_mean,
            "blur_mean_f1_delta": blur_bootstrap,
            "clean_f1_delta": clean_f1_delta,
            "success_criteria": {
                "minimum_blur_mean_f1_gain": minimum_gain,
                "minimum_blur_delta_ci_low": 0.0,
                "maximum_clean_f1_drop": maximum_clean_drop,
            },
            "selected_parameters_sha256": hashlib.sha256(
                selected_bytes
            ).hexdigest(),
            "holdout_manifest_sha256": hashlib.sha256(
                manifest_bytes
            ).hexdigest(),
        }
        verdict_path.write_text(
            json.dumps(verdict, indent=2) + "\n", encoding="utf-8"
        )
        return verdict
