from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .coco import ground_truth_by_image, load_manifest
from .config import resolve_path
from .detection_metrics import (
    bootstrap_metric_intervals,
    evaluate_coco_detection,
    paired_bootstrap_delta,
)
from .metrics import evaluate_records
from .prompt_study import _read_prediction_file


RAW_CONDITIONS = (
    "original",
    "brightness_0.25",
    "blur_sigma_4",
    "prompt_synonym",
    "prompt_hypernym",
)


def _fixed_threshold(condition: str, config: dict[str, Any]) -> float:
    if condition.startswith("strategy/"):
        return 0.0
    return float(config["evaluation"]["confidence"])


def run_extended_evaluation(config: dict[str, Any]) -> dict[str, Path]:
    results_dir = resolve_path(config, config["paths"]["results_dir"])
    manifest = load_manifest(config)
    ground_truth = ground_truth_by_image(manifest)
    raw = _read_prediction_file(results_dir / "raw_predictions.json")
    prompts = _read_prediction_file(results_dir / "prompt_predictions.json")
    strategies = _read_prediction_file(
        results_dir / "prompt_strategy_predictions.json"
    )
    strategy_ap_candidates = _read_prediction_file(
        results_dir / "prompt_strategy_ap_predictions.json"
    )
    selected = json.loads(
        (results_dir / "selected_parameters.json").read_text(encoding="utf-8")
    )
    gamma = float(selected["gamma_correction"]["gamma"])
    gamma_condition = f"lowlight_gamma_{gamma:.2f}"

    conditions: dict[
        str,
        tuple[
            str,
            dict[int, list[dict[str, Any]]],
            dict[int, list[dict[str, Any]]],
        ],
    ] = {}
    for condition in (*RAW_CONDITIONS, gamma_condition):
        conditions[f"raw/{condition}"] = (
            "raw",
            raw[condition],
            raw[condition],
        )
    for variant, predictions in prompts.items():
        conditions[f"variant/{variant}"] = (
            "prompt_variant",
            predictions,
            predictions,
        )
    for strategy, predictions in strategies.items():
        conditions[f"strategy/{strategy}"] = (
            "prompt_strategy",
            strategy_ap_candidates[strategy],
            predictions,
        )

    split_ids = {
        scope: [
            int(item["image_id"])
            for item in manifest["images"]
            if item["split"] == scope
        ]
        for scope in ("validation", "test")
    }
    summary_rows: list[dict[str, Any]] = []
    category_rows: list[dict[str, Any]] = []
    pr_rows: list[dict[str, Any]] = []
    interval_rows: list[dict[str, Any]] = []
    details_by_condition: dict[
        tuple[str, str], list[dict[str, Any]]
    ] = {}

    for scope, image_ids in split_ids.items():
        for condition, (
            family,
            ap_predictions,
            fixed_predictions,
        ) in conditions.items():
            coco_summary, coco_categories, curve = evaluate_coco_detection(
                manifest,
                ap_predictions,
                image_ids,
                config["categories"],
                confidence_threshold=0.0,
            )
            threshold = _fixed_threshold(condition, config)
            fixed_summary, fixed_details = evaluate_records(
                fixed_predictions,
                ground_truth,
                image_ids,
                threshold,
                float(config["evaluation"]["match_iou"]),
            )
            details_by_condition[(scope, condition)] = fixed_details
            summary_rows.append(
                {
                    "scope": scope,
                    "family": family,
                    "condition": condition,
                    "fixed_confidence_threshold": threshold,
                    **coco_summary,
                    "precision_at_fixed_threshold": fixed_summary["precision"],
                    "recall_at_fixed_threshold": fixed_summary["recall"],
                    "f1_at_fixed_threshold": fixed_summary["f1"],
                    "tp_at_fixed_threshold": fixed_summary["tp"],
                    "fp_at_fixed_threshold": fixed_summary["fp"],
                    "fn_at_fixed_threshold": fixed_summary["fn"],
                }
            )
            category_rows.extend(
                {
                    "scope": scope,
                    "family": family,
                    "condition": condition,
                    **row,
                }
                for row in coco_categories
            )
            pr_rows.extend(
                {
                    "scope": scope,
                    "family": family,
                    "condition": condition,
                    **row,
                }
                for row in curve
            )
            if scope == "test":
                interval_rows.extend(
                    {
                        "scope": scope,
                        "family": family,
                        "condition": condition,
                        **row,
                    }
                    for row in bootstrap_metric_intervals(
                        fixed_details,
                        int(config["evaluation"]["bootstrap_samples"]),
                        int(config["seed"]),
                    )
                )

    delta_rows: list[dict[str, Any]] = []
    baseline_condition = "strategy/canonical_baseline"
    baseline_details = details_by_condition[("test", baseline_condition)]
    for strategy in strategies:
        condition = f"strategy/{strategy}"
        if condition == baseline_condition:
            continue
        delta_rows.extend(
            {
                "scope": "test",
                "baseline": baseline_condition,
                "candidate": condition,
                **row,
            }
            for row in paired_bootstrap_delta(
                baseline_details,
                details_by_condition[("test", condition)],
                int(config["evaluation"]["bootstrap_samples"]),
                int(config["seed"]) + 1,
            )
        )

    paths = {
        "summary": results_dir / "detection_ap_metrics.csv",
        "category": results_dir / "detection_ap_by_category.csv",
        "pr": results_dir / "detection_pr_curves.csv",
        "intervals": results_dir / "bootstrap_intervals.csv",
        "deltas": results_dir / "bootstrap_deltas_vs_baseline.csv",
    }
    pd.DataFrame(summary_rows).to_csv(paths["summary"], index=False)
    pd.DataFrame(category_rows).to_csv(paths["category"], index=False)
    pd.DataFrame(pr_rows).to_csv(paths["pr"], index=False)
    pd.DataFrame(interval_rows).to_csv(paths["intervals"], index=False)
    pd.DataFrame(delta_rows).to_csv(paths["deltas"], index=False)
    return paths
