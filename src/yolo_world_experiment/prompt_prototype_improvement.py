from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

from .coco import ground_truth_by_image
from .config import category_names, resolve_path
from .detection_metrics import evaluate_coco_detection, paired_bootstrap_delta
from .metrics import evaluate_records
from .model import YoloWorldPredictor
from .prompt_study import _read_prediction_file, _write_prediction_file
from .same_model_improvement import (
    paired_bootstrap_mean_condition_f1_delta,
)


PROMPT_CONDITION_INDICES = {
    "canonical": 0,
    "synonym": 1,
    "hypernym": 2,
}


def prompts_for_condition(
    config: dict[str, Any],
    condition: str,
) -> list[str]:
    if condition not in PROMPT_CONDITION_INDICES:
        raise ValueError(f"Unsupported prompt condition: {condition}")
    index = PROMPT_CONDITION_INDICES[condition]
    return [
        str(specification["prompts"][index])
        for specification in config["categories"].values()
    ]


def build_canonical_anchored_prompt_groups(
    canonical_prompts: list[str],
    input_prompts: list[str],
    anchor_weight: float,
) -> tuple[list[list[str]], list[list[float]]]:
    """Build one input-plus-canonical text prototype per output class."""
    if len(canonical_prompts) != len(input_prompts):
        raise ValueError("canonical and input prompt lists must align")
    if not 0.0 <= float(anchor_weight) <= 1.0:
        raise ValueError("anchor_weight must be in [0, 1]")
    groups: list[list[str]] = []
    weights: list[list[float]] = []
    for canonical, input_prompt in zip(canonical_prompts, input_prompts):
        if canonical == input_prompt:
            groups.append([canonical])
            weights.append([1.0])
        else:
            groups.append([input_prompt, canonical])
            weights.append([1.0 - float(anchor_weight), float(anchor_weight)])
    return groups, weights


def _load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _combined_manifest(
    manifests: list[tuple[str, dict[str, Any]]],
) -> dict[str, Any]:
    images = [
        {**image, "development_fold": fold}
        for fold, manifest in manifests
        for image in manifest["images"]
    ]
    image_ids = [int(image["image_id"]) for image in images]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("Development prompt manifests must be disjoint")
    return {
        "source": "Four disjoint COCO 2017 val development manifests",
        "categories": list(manifests[0][1]["categories"]),
        "images": images,
    }


def _method_key(anchor_weight: float | None, condition: str) -> str:
    if anchor_weight is None:
        return f"baseline__{condition}"
    return f"anchor_{float(anchor_weight):.2f}__{condition}"


def _mean_metric(
    rows: list[dict[str, Any]],
    conditions: list[str],
    metric: str,
) -> float:
    selected = [
        float(row[metric])
        for row in rows
        if str(row["condition"]) in conditions
    ]
    if len(selected) != len(conditions):
        raise ValueError(
            f"Expected one {metric} row for each target condition"
        )
    return float(sum(selected) / len(selected))


class PromptPrototypeImprovementStudy:
    """Canonical-anchored text prototypes with the detector held fixed."""

    development_manifest_keys = (
        "subset_manifest",
        "holdout_manifest",
        "final_holdout_manifest",
        "same_model_holdout_manifest",
    )

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        if str(config["model"]["name"]) != "yolov8s-worldv2.pt":
            raise ValueError(
                "Prompt prototype improvement requires yolov8s-worldv2.pt"
            )
        self.categories = category_names(config)
        self.results_dir = resolve_path(
            config, config["paths"]["prompt_prototype_results_dir"]
        )
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir = resolve_path(config, config["paths"]["coco_images"])
        self.confidence = float(config["evaluation"]["confidence"])
        self.raw_confidence = float(config["model"]["raw_confidence"])
        self.match_iou = float(config["evaluation"]["match_iou"])
        self.bootstrap_samples = int(
            config["evaluation"]["bootstrap_samples"]
        )
        self.target_conditions = [
            str(value)
            for value in config["prompt_prototype_improvement"][
                "target_conditions"
            ]
        ]
        self.conditions = list(PROMPT_CONDITION_INDICES)

    def _development_manifests(
        self,
    ) -> list[tuple[str, dict[str, Any]]]:
        return [
            (
                key,
                _load_manifest(
                    resolve_path(self.config, self.config["paths"][key])
                ),
            )
            for key in self.development_manifest_keys
        ]

    def _predict_records(
        self,
        *,
        manifest: dict[str, Any],
        anchor_weights: list[float],
        cache_path: Path,
        metadata_path: Path,
    ) -> dict[str, dict[int, list[dict[str, Any]]]]:
        image_ids = [
            int(entry["image_id"]) for entry in manifest["images"]
        ]
        records: dict[str, dict[int, list[dict[str, Any]]]] = {}
        if cache_path.exists() and metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if (
                metadata.get("model") == self.config["model"]["name"]
                and metadata.get("image_ids") == image_ids
                and metadata.get("anchor_weights") == anchor_weights
            ):
                records = _read_prediction_file(cache_path)

        required_keys = [
            _method_key(None, condition) for condition in self.conditions
        ] + [
            _method_key(weight, condition)
            for weight in anchor_weights
            for condition in self.target_conditions
        ]
        missing = [
            key
            for key in required_keys
            if key not in records
            or set(records[key]) != set(image_ids)
        ]
        if not missing:
            return records

        predictor = YoloWorldPredictor(self.config, self.categories)
        canonical = prompts_for_condition(self.config, "canonical")
        entries = {
            int(entry["image_id"]): entry
            for entry in manifest["images"]
        }
        for key in missing:
            prefix, condition = key.split("__", maxsplit=1)
            input_prompts = prompts_for_condition(self.config, condition)
            if prefix == "baseline":
                predictor.set_prompts(input_prompts)
                method = "raw_prompt"
                anchor_weight: float | None = None
            else:
                anchor_weight = float(prefix.removeprefix("anchor_"))
                prompt_groups, prompt_weights = (
                    build_canonical_anchored_prompt_groups(
                        canonical,
                        input_prompts,
                        anchor_weight,
                    )
                )
                predictor.set_prompt_prototypes(
                    prompt_groups,
                    prompt_weights,
                )
                method = f"canonical_anchor_{anchor_weight:.2f}"
            print(
                f"[prompt-prototype] inference: {method}/{condition} "
                f"({len(image_ids)} images)"
            )
            by_image: dict[int, list[dict[str, Any]]] = {}
            for image_id in image_ids:
                entry = entries[image_id]
                image = Image.open(
                    self.images_dir / entry["file_name"]
                ).convert("RGB")
                by_image[image_id] = [
                    {
                        **prediction,
                        "prompt_condition": condition,
                        "prototype_method": method,
                        "canonical_anchor_weight": anchor_weight,
                    }
                    for prediction in predictor.predict(image)
                ]
            records[key] = by_image
            _write_prediction_file(cache_path, records)
            metadata_path.write_text(
                json.dumps(
                    {
                        "model": self.config["model"]["name"],
                        "image_ids": image_ids,
                        "anchor_weights": anchor_weights,
                        "conditions": self.conditions,
                        "target_conditions": self.target_conditions,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        return records

    def _evaluate(
        self,
        *,
        manifest: dict[str, Any],
        predictions: dict[int, list[dict[str, Any]]],
        image_ids: list[int],
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        gt = ground_truth_by_image(manifest)
        summary, details = evaluate_records(
            predictions,
            gt,
            image_ids,
            self.confidence,
            self.match_iou,
        )
        ap, _, _ = evaluate_coco_detection(
            manifest,
            predictions,
            image_ids,
            self.config["categories"],
            confidence_threshold=self.raw_confidence,
        )
        return {
            **summary,
            "ap": float(ap["ap"]),
            "ap50": float(ap["ap50"]),
            "ap75": float(ap["ap75"]),
        }, details

    def run_screening(self) -> dict[str, Any]:
        first_fold, first_manifest = self._development_manifests()[0]
        anchor_weights = [
            float(value)
            for value in self.config["prompt_prototype_improvement"][
                "anchor_weights"
            ]
        ]
        records = self._predict_records(
            manifest=first_manifest,
            anchor_weights=anchor_weights,
            cache_path=(
                self.results_dir / "prompt_prototype_screening_predictions.json"
            ),
            metadata_path=(
                self.results_dir / "prompt_prototype_screening_metadata.json"
            ),
        )
        image_ids = [
            int(entry["image_id"]) for entry in first_manifest["images"]
        ]
        summary_rows: list[dict[str, Any]] = []
        details_by_key: dict[str, list[dict[str, Any]]] = {}
        for condition in self.conditions:
            key = _method_key(None, condition)
            summary, details = self._evaluate(
                manifest=first_manifest,
                predictions=records[key],
                image_ids=image_ids,
            )
            summary_rows.append(
                {
                    "scope": "screening_100",
                    "fold": first_fold,
                    "method": "raw_prompt",
                    "anchor_weight": None,
                    "condition": condition,
                    **summary,
                }
            )
            details_by_key[key] = details
        for anchor_weight in anchor_weights:
            for condition in self.conditions:
                if condition == "canonical":
                    key = _method_key(None, condition)
                else:
                    key = _method_key(anchor_weight, condition)
                summary, details = self._evaluate(
                    manifest=first_manifest,
                    predictions=records[key],
                    image_ids=image_ids,
                )
                method = f"canonical_anchor_{anchor_weight:.2f}"
                summary_rows.append(
                    {
                        "scope": "screening_100",
                        "fold": first_fold,
                        "method": method,
                        "anchor_weight": anchor_weight,
                        "condition": condition,
                        **summary,
                    }
                )
                details_by_key[
                    f"{method}__{condition}"
                ] = details

        baseline_rows = [
            row for row in summary_rows if row["method"] == "raw_prompt"
        ]
        baseline_mean_f1 = _mean_metric(
            baseline_rows, self.target_conditions, "f1"
        )
        baseline_mean_ap = _mean_metric(
            baseline_rows, self.target_conditions, "ap"
        )
        ranking: list[dict[str, Any]] = []
        for anchor_weight in anchor_weights:
            method = f"canonical_anchor_{anchor_weight:.2f}"
            candidate_rows = [
                row for row in summary_rows if row["method"] == method
            ]
            candidate_mean_f1 = _mean_metric(
                candidate_rows, self.target_conditions, "f1"
            )
            candidate_mean_ap = _mean_metric(
                candidate_rows, self.target_conditions, "ap"
            )
            bootstrap = paired_bootstrap_mean_condition_f1_delta(
                {
                    condition: details_by_key[
                        _method_key(None, condition)
                    ]
                    for condition in self.target_conditions
                },
                {
                    condition: details_by_key[
                        f"{method}__{condition}"
                    ]
                    for condition in self.target_conditions
                },
                samples=self.bootstrap_samples,
                seed=int(self.config["seed"]) + int(anchor_weight * 1000),
            )
            ranking.append(
                {
                    "anchor_weight": anchor_weight,
                    "baseline_target_mean_f1": baseline_mean_f1,
                    "candidate_target_mean_f1": candidate_mean_f1,
                    "target_mean_f1_gain": candidate_mean_f1
                    - baseline_mean_f1,
                    "f1_delta_ci_low": float(bootstrap["ci_low"]),
                    "f1_delta_ci_high": float(bootstrap["ci_high"]),
                    "baseline_target_mean_ap": baseline_mean_ap,
                    "candidate_target_mean_ap": candidate_mean_ap,
                    "target_mean_ap_delta": candidate_mean_ap
                    - baseline_mean_ap,
                }
            )
        ranking.sort(
            key=lambda row: (
                float(row["target_mean_f1_gain"]),
                float(row["target_mean_ap_delta"]),
                -float(row["anchor_weight"]),
            ),
            reverse=True,
        )
        keep = int(
            self.config["prompt_prototype_improvement"][
                "screening_candidates_to_keep"
            ]
        )
        selected_weights = [
            float(row["anchor_weight"]) for row in ranking[:keep]
        ]
        pd.DataFrame(summary_rows).to_csv(
            self.results_dir / "prompt_prototype_screening_summary.csv",
            index=False,
        )
        pd.DataFrame(ranking).to_csv(
            self.results_dir / "prompt_prototype_screening_ranking.csv",
            index=False,
        )
        selection = {
            "selection_scope": "first_known_100_only",
            "model": self.config["model"]["name"],
            "selected_anchor_weights": selected_weights,
            "ranking": ranking,
        }
        (
            self.results_dir / "prompt_prototype_screening_selection.json"
        ).write_text(
            json.dumps(selection, indent=2) + "\n",
            encoding="utf-8",
        )
        return selection

    def run_development(self) -> dict[str, Any]:
        screening_path = (
            self.results_dir / "prompt_prototype_screening_selection.json"
        )
        if not screening_path.exists():
            raise FileNotFoundError(
                "Run prompt prototype screening before development"
            )
        screening = json.loads(screening_path.read_text(encoding="utf-8"))
        anchor_weights = [
            float(value)
            for value in screening["selected_anchor_weights"]
        ]
        manifests = self._development_manifests()
        combined = _combined_manifest(manifests)
        image_ids = [
            int(entry["image_id"]) for entry in combined["images"]
        ]
        fold_ids = {
            fold: [
                int(entry["image_id"])
                for entry in manifest["images"]
            ]
            for fold, manifest in manifests
        }
        records = self._predict_records(
            manifest=combined,
            anchor_weights=anchor_weights,
            cache_path=(
                self.results_dir / "prompt_prototype_development_predictions.json"
            ),
            metadata_path=(
                self.results_dir / "prompt_prototype_development_metadata.json"
            ),
        )

        summary_rows: list[dict[str, Any]] = []
        detail_rows: list[dict[str, Any]] = []
        detail_lookup: dict[tuple[str, str], list[dict[str, Any]]] = {}
        methods: list[tuple[str, float | None]] = [
            ("raw_prompt", None),
            *[
                (f"canonical_anchor_{weight:.2f}", weight)
                for weight in anchor_weights
            ],
        ]
        for scope, ids in [("development_400", image_ids), *fold_ids.items()]:
            for method, anchor_weight in methods:
                for condition in self.conditions:
                    key = (
                        _method_key(None, condition)
                        if anchor_weight is None or condition == "canonical"
                        else _method_key(anchor_weight, condition)
                    )
                    summary, details = self._evaluate(
                        manifest=combined,
                        predictions=records[key],
                        image_ids=ids,
                    )
                    summary_rows.append(
                        {
                            "scope": scope,
                            "method": method,
                            "anchor_weight": anchor_weight,
                            "condition": condition,
                            **summary,
                        }
                    )
                    if scope == "development_400":
                        tagged = [
                            {
                                "scope": scope,
                                "method": method,
                                "anchor_weight": anchor_weight,
                                "condition": condition,
                                **row,
                            }
                            for row in details
                        ]
                        detail_rows.extend(tagged)
                        detail_lookup[(method, condition)] = details

        overall_rows = [
            row
            for row in summary_rows
            if row["scope"] == "development_400"
        ]
        baseline_rows = [
            row for row in overall_rows if row["method"] == "raw_prompt"
        ]
        baseline_mean_f1 = _mean_metric(
            baseline_rows, self.target_conditions, "f1"
        )
        baseline_mean_ap = _mean_metric(
            baseline_rows, self.target_conditions, "ap"
        )
        ranking: list[dict[str, Any]] = []
        for anchor_weight in anchor_weights:
            method = f"canonical_anchor_{anchor_weight:.2f}"
            candidate_rows = [
                row for row in overall_rows if row["method"] == method
            ]
            candidate_mean_f1 = _mean_metric(
                candidate_rows, self.target_conditions, "f1"
            )
            candidate_mean_ap = _mean_metric(
                candidate_rows, self.target_conditions, "ap"
            )
            bootstrap = paired_bootstrap_mean_condition_f1_delta(
                {
                    condition: detail_lookup[("raw_prompt", condition)]
                    for condition in self.target_conditions
                },
                {
                    condition: detail_lookup[(method, condition)]
                    for condition in self.target_conditions
                },
                samples=self.bootstrap_samples,
                seed=int(self.config["seed"]) + 3000
                + int(anchor_weight * 1000),
            )
            fold_gains: dict[str, float] = {}
            for fold in fold_ids:
                fold_baseline = [
                    row
                    for row in summary_rows
                    if row["scope"] == fold
                    and row["method"] == "raw_prompt"
                ]
                fold_candidate = [
                    row
                    for row in summary_rows
                    if row["scope"] == fold
                    and row["method"] == method
                ]
                fold_gains[fold] = _mean_metric(
                    fold_candidate, self.target_conditions, "f1"
                ) - _mean_metric(
                    fold_baseline, self.target_conditions, "f1"
                )
            ranking.append(
                {
                    "anchor_weight": anchor_weight,
                    "baseline_target_mean_f1": baseline_mean_f1,
                    "candidate_target_mean_f1": candidate_mean_f1,
                    "target_mean_f1_gain": candidate_mean_f1
                    - baseline_mean_f1,
                    "f1_delta_ci_low": float(bootstrap["ci_low"]),
                    "f1_delta_ci_high": float(bootstrap["ci_high"]),
                    "baseline_target_mean_ap": baseline_mean_ap,
                    "candidate_target_mean_ap": candidate_mean_ap,
                    "target_mean_ap_delta": candidate_mean_ap
                    - baseline_mean_ap,
                    "minimum_fold_gain": min(fold_gains.values()),
                    "fold_gains": json.dumps(
                        fold_gains, sort_keys=True
                    ),
                }
            )
        ranking.sort(
            key=lambda row: (
                float(row["target_mean_f1_gain"]),
                float(row["minimum_fold_gain"]),
                float(row["target_mean_ap_delta"]),
                -float(row["anchor_weight"]),
            ),
            reverse=True,
        )
        selected_row = ranking[0]
        minimum_gain = float(
            self.config["prompt_prototype_improvement"][
                "minimum_mean_f1_gain"
            ]
        )
        minimum_fold_gain = float(
            self.config["prompt_prototype_improvement"][
                "minimum_fold_gain"
            ]
        )
        maximum_map_drop = float(
            self.config["prompt_prototype_improvement"][
                "maximum_mean_map_drop"
            ]
        )
        gate_passed = (
            float(selected_row["target_mean_f1_gain"]) >= minimum_gain
            and float(selected_row["f1_delta_ci_low"]) > 0.0
            and float(selected_row["minimum_fold_gain"])
            > minimum_fold_gain
            and float(selected_row["target_mean_ap_delta"])
            >= -maximum_map_drop
        )
        pd.DataFrame(summary_rows).to_csv(
            self.results_dir / "prompt_prototype_development_summary.csv",
            index=False,
        )
        pd.DataFrame(detail_rows).to_csv(
            self.results_dir / "prompt_prototype_development_per_image.csv",
            index=False,
        )
        pd.DataFrame(ranking).to_csv(
            self.results_dir / "prompt_prototype_development_ranking.csv",
            index=False,
        )
        selected = {
            "method": "canonical_anchored_text_prototype",
            "model": self.config["model"]["name"],
            "model_change_used": False,
            "training_used": False,
            "prompt_box_union_used": False,
            "selection_scope": "development_400_only",
            "development_gate_passed": gate_passed,
            "anchor_weight": float(selected_row["anchor_weight"]),
            "target_conditions": self.target_conditions,
            "canonical_condition_is_unchanged": True,
            "metrics": selected_row,
            "success_criteria": {
                "minimum_mean_f1_gain": minimum_gain,
                "minimum_f1_delta_ci_low": 0.0,
                "minimum_fold_gain": minimum_fold_gain,
                "maximum_canonical_f1_drop": float(
                    self.config["prompt_prototype_improvement"][
                        "maximum_canonical_f1_drop"
                    ]
                ),
                "maximum_mean_map_drop": maximum_map_drop,
            },
            "development_manifest_sha256": {
                key: hashlib.sha256(
                    resolve_path(
                        self.config, self.config["paths"][key]
                    ).read_bytes()
                ).hexdigest()
                for key in self.development_manifest_keys
            },
        }
        (
            self.results_dir / "selected_prompt_prototype_method.json"
        ).write_text(
            json.dumps(selected, indent=2) + "\n",
            encoding="utf-8",
        )
        return selected

    def run_final_holdout(self) -> dict[str, Any]:
        verdict_path = (
            self.results_dir / "prompt_prototype_final_verdict.json"
        )
        if verdict_path.exists():
            raise RuntimeError(
                "Prompt prototype holdout has already been evaluated; "
                "refusing a second run"
            )
        selected_path = (
            self.results_dir / "selected_prompt_prototype_method.json"
        )
        selected = json.loads(selected_path.read_text(encoding="utf-8"))
        if not bool(selected["development_gate_passed"]):
            raise RuntimeError(
                "Prompt prototype method did not pass development"
            )
        holdout_path = resolve_path(
            self.config,
            self.config["paths"]["prompt_prototype_holdout_manifest"],
        )
        manifest = _load_manifest(holdout_path)
        image_ids = [
            int(entry["image_id"]) for entry in manifest["images"]
        ]
        anchor_weight = float(selected["anchor_weight"])
        records = self._predict_records(
            manifest=manifest,
            anchor_weights=[anchor_weight],
            cache_path=(
                self.results_dir / "prompt_prototype_final_predictions.json"
            ),
            metadata_path=(
                self.results_dir / "prompt_prototype_final_metadata.json"
            ),
        )
        summary_rows: list[dict[str, Any]] = []
        detail_rows: list[dict[str, Any]] = []
        detail_lookup: dict[tuple[str, str], list[dict[str, Any]]] = {}
        methods: list[tuple[str, float | None]] = [
            ("raw_prompt", None),
            (f"canonical_anchor_{anchor_weight:.2f}", anchor_weight),
        ]
        for method, method_weight in methods:
            for condition in self.conditions:
                key = (
                    _method_key(None, condition)
                    if method_weight is None or condition == "canonical"
                    else _method_key(method_weight, condition)
                )
                summary, details = self._evaluate(
                    manifest=manifest,
                    predictions=records[key],
                    image_ids=image_ids,
                )
                summary_rows.append(
                    {
                        "scope": "prompt_prototype_final_holdout",
                        "method": method,
                        "anchor_weight": method_weight,
                        "condition": condition,
                        **summary,
                    }
                )
                detail_lookup[(method, condition)] = details
                detail_rows.extend(
                    {
                        "scope": "prompt_prototype_final_holdout",
                        "method": method,
                        "anchor_weight": method_weight,
                        "condition": condition,
                        **row,
                    }
                    for row in details
                )
        baseline_rows = [
            row for row in summary_rows if row["method"] == "raw_prompt"
        ]
        candidate_method = f"canonical_anchor_{anchor_weight:.2f}"
        candidate_rows = [
            row
            for row in summary_rows
            if row["method"] == candidate_method
        ]
        baseline_mean_f1 = _mean_metric(
            baseline_rows, self.target_conditions, "f1"
        )
        candidate_mean_f1 = _mean_metric(
            candidate_rows, self.target_conditions, "f1"
        )
        baseline_mean_ap = _mean_metric(
            baseline_rows, self.target_conditions, "ap"
        )
        candidate_mean_ap = _mean_metric(
            candidate_rows, self.target_conditions, "ap"
        )
        target_bootstrap = paired_bootstrap_mean_condition_f1_delta(
            {
                condition: detail_lookup[("raw_prompt", condition)]
                for condition in self.target_conditions
            },
            {
                condition: detail_lookup[(candidate_method, condition)]
                for condition in self.target_conditions
            },
            samples=self.bootstrap_samples,
            seed=int(self.config["selection"][
                "prompt_prototype_holdout_seed"
            ]),
        )
        canonical_bootstrap = next(
            row
            for row in paired_bootstrap_delta(
                detail_lookup[("raw_prompt", "canonical")],
                detail_lookup[(candidate_method, "canonical")],
                self.bootstrap_samples,
                int(
                    self.config["selection"][
                        "prompt_prototype_holdout_seed"
                    ]
                )
                + 101,
            )
            if row["metric"] == "f1"
        )
        minimum_gain = float(
            selected["success_criteria"]["minimum_mean_f1_gain"]
        )
        maximum_canonical_drop = float(
            selected["success_criteria"]["maximum_canonical_f1_drop"]
        )
        maximum_map_drop = float(
            selected["success_criteria"]["maximum_mean_map_drop"]
        )
        passed = (
            candidate_mean_f1 - baseline_mean_f1 >= minimum_gain
            and float(target_bootstrap["ci_low"]) > 0.0
            and float(canonical_bootstrap["estimate"])
            >= -maximum_canonical_drop
            and candidate_mean_ap - baseline_mean_ap >= -maximum_map_drop
        )
        pd.DataFrame(summary_rows).to_csv(
            self.results_dir / "prompt_prototype_final_summary.csv",
            index=False,
        )
        pd.DataFrame(detail_rows).to_csv(
            self.results_dir / "prompt_prototype_final_per_image.csv",
            index=False,
        )
        pd.DataFrame(
            [
                {"comparison": "target_prompt_mean", **target_bootstrap},
                {"comparison": "canonical_guardrail", **canonical_bootstrap},
            ]
        ).to_csv(
            self.results_dir / "prompt_prototype_final_bootstrap.csv",
            index=False,
        )
        verdict = {
            "passed": passed,
            "method": selected["method"],
            "model": selected["model"],
            "model_change_used": False,
            "training_used": False,
            "prompt_box_union_used": False,
            "selection_scope": selected["selection_scope"],
            "selection_was_frozen_before_holdout": True,
            "holdout_evaluated_once": True,
            "anchor_weight": anchor_weight,
            "target_conditions": self.target_conditions,
            "baseline_target_mean_f1": baseline_mean_f1,
            "candidate_target_mean_f1": candidate_mean_f1,
            "target_mean_f1_delta": target_bootstrap,
            "baseline_target_mean_ap": baseline_mean_ap,
            "candidate_target_mean_ap": candidate_mean_ap,
            "target_mean_ap_delta": candidate_mean_ap - baseline_mean_ap,
            "canonical_f1_delta": canonical_bootstrap,
            "success_criteria": selected["success_criteria"],
            "selected_parameters_sha256": hashlib.sha256(
                selected_path.read_bytes()
            ).hexdigest(),
            "holdout_manifest_sha256": hashlib.sha256(
                holdout_path.read_bytes()
            ).hexdigest(),
        }
        verdict_path.write_text(
            json.dumps(verdict, indent=2) + "\n",
            encoding="utf-8",
        )
        return verdict
