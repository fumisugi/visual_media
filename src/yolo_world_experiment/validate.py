from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from .config import resolve_path


def _close(actual: float, expected: float, tolerance: float = 1e-9) -> bool:
    return math.isclose(float(actual), float(expected), rel_tol=tolerance, abs_tol=tolerance)


def _check(
    checks: list[dict[str, Any]],
    name: str,
    passed: bool,
    detail: str,
) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def validate_results(config: dict[str, Any]) -> Path:
    results_dir = resolve_path(config, config["paths"]["results_dir"])
    manifest_path = resolve_path(config, config["paths"]["subset_manifest"])
    summary_path = results_dir / "summary_metrics.csv"
    detail_path = results_dir / "per_image_metrics.csv"
    ensemble_tuning_path = results_dir / "ensemble_tuning.csv"
    gamma_tuning_path = results_dir / "gamma_tuning.csv"
    selected_path = results_dir / "selected_parameters.json"
    prompt_metrics_path = results_dir / "prompt_strategy_metrics.csv"
    prompt_details_path = results_dir / "prompt_strategy_per_image.csv"
    prompt_selected_path = results_dir / "prompt_selected_parameters.json"
    detection_ap_path = results_dir / "detection_ap_metrics.csv"
    bootstrap_path = results_dir / "bootstrap_intervals.csv"
    bootstrap_delta_path = results_dir / "bootstrap_deltas_vs_baseline.csv"
    holdout_manifest_path = resolve_path(
        config, config["paths"]["holdout_manifest"]
    )
    final_manifest_path = resolve_path(
        config, config["paths"]["final_holdout_manifest"]
    )
    improvement_dir = resolve_path(
        config, config["paths"]["improvement_results_dir"]
    )
    model_selected_path = improvement_dir / "selected_model_scale.json"
    model_final_summary_path = (
        improvement_dir / "model_final_holdout_summary.csv"
    )
    model_final_detail_path = (
        improvement_dir / "model_final_holdout_per_image.csv"
    )
    model_final_delta_path = (
        improvement_dir / "model_final_holdout_bootstrap_deltas.csv"
    )
    model_final_verdict_path = (
        improvement_dir / "model_final_holdout_verdict.json"
    )
    same_model_manifest_path = resolve_path(
        config, config["paths"]["same_model_holdout_manifest"]
    )
    same_model_dir = resolve_path(
        config, config["paths"]["same_model_results_dir"]
    )
    same_model_selected_path = (
        same_model_dir / "selected_same_model_method.json"
    )
    same_model_summary_path = (
        same_model_dir / "same_model_final_summary.csv"
    )
    same_model_detail_path = (
        same_model_dir / "same_model_final_per_image.csv"
    )
    same_model_bootstrap_path = (
        same_model_dir / "same_model_final_bootstrap.csv"
    )
    same_model_detector_path = (
        same_model_dir / "same_model_final_blur_detector.csv"
    )
    same_model_verdict_path = (
        same_model_dir / "same_model_final_verdict.json"
    )
    prompt_prototype_manifest_path = resolve_path(
        config, config["paths"]["prompt_prototype_holdout_manifest"]
    )
    prompt_prototype_dir = resolve_path(
        config, config["paths"]["prompt_prototype_results_dir"]
    )
    prompt_prototype_selected_path = (
        prompt_prototype_dir / "selected_prompt_prototype_method.json"
    )
    prompt_prototype_summary_path = (
        prompt_prototype_dir / "prompt_prototype_final_summary.csv"
    )
    prompt_prototype_detail_path = (
        prompt_prototype_dir / "prompt_prototype_final_per_image.csv"
    )
    prompt_prototype_bootstrap_path = (
        prompt_prototype_dir / "prompt_prototype_final_bootstrap.csv"
    )
    prompt_prototype_verdict_path = (
        prompt_prototype_dir / "prompt_prototype_final_verdict.json"
    )

    required = [
        manifest_path,
        summary_path,
        detail_path,
        ensemble_tuning_path,
        gamma_tuning_path,
        selected_path,
        prompt_metrics_path,
        prompt_details_path,
        prompt_selected_path,
        detection_ap_path,
        bootstrap_path,
        bootstrap_delta_path,
        holdout_manifest_path,
        final_manifest_path,
        model_selected_path,
        model_final_summary_path,
        model_final_detail_path,
        model_final_delta_path,
        model_final_verdict_path,
        same_model_manifest_path,
        same_model_selected_path,
        same_model_summary_path,
        same_model_detail_path,
        same_model_bootstrap_path,
        same_model_detector_path,
        same_model_verdict_path,
        prompt_prototype_manifest_path,
        prompt_prototype_selected_path,
        prompt_prototype_summary_path,
        prompt_prototype_detail_path,
        prompt_prototype_bootstrap_path,
        prompt_prototype_verdict_path,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Required result files are missing: {missing}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = pd.read_csv(summary_path)
    details = pd.read_csv(detail_path)
    ensemble_tuning = pd.read_csv(ensemble_tuning_path)
    gamma_tuning = pd.read_csv(gamma_tuning_path)
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    prompt_metrics = pd.read_csv(prompt_metrics_path)
    prompt_details = pd.read_csv(prompt_details_path)
    prompt_selected = json.loads(
        prompt_selected_path.read_text(encoding="utf-8")
    )
    detection_ap = pd.read_csv(detection_ap_path)
    bootstrap = pd.read_csv(bootstrap_path)
    bootstrap_delta = pd.read_csv(bootstrap_delta_path)
    holdout_manifest = json.loads(
        holdout_manifest_path.read_text(encoding="utf-8")
    )
    final_manifest = json.loads(
        final_manifest_path.read_text(encoding="utf-8")
    )
    model_selected_bytes = model_selected_path.read_bytes()
    model_final_summary = pd.read_csv(model_final_summary_path)
    model_final_details = pd.read_csv(model_final_detail_path)
    model_final_deltas = pd.read_csv(model_final_delta_path)
    model_final_verdict = json.loads(
        model_final_verdict_path.read_text(encoding="utf-8")
    )
    same_model_manifest = json.loads(
        same_model_manifest_path.read_text(encoding="utf-8")
    )
    same_model_selected_bytes = same_model_selected_path.read_bytes()
    same_model_summary = pd.read_csv(same_model_summary_path)
    same_model_details = pd.read_csv(same_model_detail_path)
    same_model_bootstrap = pd.read_csv(same_model_bootstrap_path)
    same_model_detector = pd.read_csv(same_model_detector_path)
    same_model_verdict = json.loads(
        same_model_verdict_path.read_text(encoding="utf-8")
    )
    prompt_prototype_manifest = json.loads(
        prompt_prototype_manifest_path.read_text(encoding="utf-8")
    )
    prompt_prototype_selected_bytes = (
        prompt_prototype_selected_path.read_bytes()
    )
    prompt_prototype_selected = json.loads(
        prompt_prototype_selected_bytes.decode("utf-8")
    )
    prompt_prototype_summary = pd.read_csv(
        prompt_prototype_summary_path
    )
    prompt_prototype_details = pd.read_csv(
        prompt_prototype_detail_path
    )
    prompt_prototype_bootstrap = pd.read_csv(
        prompt_prototype_bootstrap_path
    )
    prompt_prototype_verdict = json.loads(
        prompt_prototype_verdict_path.read_text(encoding="utf-8")
    )
    checks: list[dict[str, Any]] = []

    images = manifest["images"]
    validation_ids = {
        int(image["image_id"]) for image in images if image["split"] == "validation"
    }
    test_ids = {int(image["image_id"]) for image in images if image["split"] == "test"}
    pilot_ids = {int(image["image_id"]) for image in images if image["pilot"]}
    expected_selection = config["selection"]

    _check(
        checks,
        "manifest_total_images",
        len(images) == int(expected_selection["total_images"]),
        f"observed={len(images)}, expected={expected_selection['total_images']}",
    )
    _check(
        checks,
        "validation_test_split",
        len(validation_ids) == int(expected_selection["validation_images"])
        and len(test_ids)
        == int(expected_selection["total_images"])
        - int(expected_selection["validation_images"]),
        f"validation={len(validation_ids)}, test={len(test_ids)}",
    )
    _check(
        checks,
        "split_disjoint",
        validation_ids.isdisjoint(test_ids),
        f"overlap={sorted(validation_ids & test_ids)}",
    )
    _check(
        checks,
        "pilot_is_validation_subset",
        len(pilot_ids) == int(expected_selection["pilot_images"])
        and pilot_ids <= validation_ids,
        f"pilot={len(pilot_ids)}, outside_validation={sorted(pilot_ids - validation_ids)}",
    )

    primary_counts = pd.Series(
        [image["primary_category"] for image in images]
    ).value_counts()
    quota_ok = all(
        int(primary_counts.get(category, 0)) == int(spec["quota"])
        for category, spec in config["categories"].items()
    )
    _check(
        checks,
        "primary_category_quotas",
        quota_ok,
        ", ".join(
            f"{category}={int(primary_counts.get(category, 0))}"
            for category in config["categories"]
        ),
    )
    split_primary_counts = {
        split: pd.Series(
            [
                image["primary_category"]
                for image in images
                if image["split"] == split
            ]
        ).value_counts()
        for split in ("validation", "test")
    }
    split_balance_ok = all(
        max(
            int(counts.get(category, 0))
            for category in config["categories"]
        )
        - min(
            int(counts.get(category, 0))
            for category in config["categories"]
        )
        <= 1
        for counts in split_primary_counts.values()
    )
    _check(
        checks,
        "primary_categories_balanced_by_split",
        split_balance_ok,
        "; ".join(
            f"{split}="
            + ",".join(
                f"{category}:{int(counts.get(category, 0))}"
                for category in config["categories"]
            )
            for split, counts in split_primary_counts.items()
        ),
    )

    expected_conditions = {
        *(item["name"] for item in config["corruptions"]),
        "prompt_canonical",
        "prompt_synonym",
        "prompt_hypernym",
        "baseline_tuned",
        "prompt_ensemble",
        "lowlight_baseline",
        "gamma_correction",
    }
    observed_conditions = set(summary["condition"])
    _check(
        checks,
        "required_conditions_present",
        expected_conditions <= observed_conditions,
        f"missing={sorted(expected_conditions - observed_conditions)}",
    )

    aggregation_failures: list[str] = []
    formula_failures: list[str] = []
    for row in summary.itertuples(index=False):
        group = details[
            (details["scope"] == row.scope)
            & (details["family"] == row.family)
            & (details["condition"] == row.condition)
            & (details["confidence_threshold"] == row.confidence_threshold)
        ]
        if group.empty:
            aggregation_failures.append(f"{row.family}/{row.condition}: no detail rows")
            continue
        observed = {
            key: int(group[key].sum())
            for key in ("tp", "fp", "fn")
        }
        expected = {"tp": int(row.tp), "fp": int(row.fp), "fn": int(row.fn)}
        if observed != expected or len(group) != int(row.image_count):
            aggregation_failures.append(
                f"{row.family}/{row.condition}: detail={observed}, summary={expected}, "
                f"rows={len(group)}"
            )

        precision = row.tp / (row.tp + row.fp) if row.tp + row.fp else 0.0
        recall = row.tp / (row.tp + row.fn) if row.tp + row.fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        if not (
            _close(row.precision, precision)
            and _close(row.recall, recall)
            and _close(row.f1, f1)
            and int(row.ground_truth_count) == int(row.tp + row.fn)
            and int(row.prediction_count) == int(row.tp + row.fp)
        ):
            formula_failures.append(f"{row.family}/{row.condition}")

    _check(
        checks,
        "summary_matches_per_image_rows",
        not aggregation_failures,
        "ok" if not aggregation_failures else "; ".join(aggregation_failures),
    )
    _check(
        checks,
        "metric_formulas",
        not formula_failures,
        "ok" if not formula_failures else f"failed={formula_failures}",
    )

    selected_ensemble = selected["ensemble"]
    best_ensemble = ensemble_tuning.iloc[0]
    ensemble_selection_ok = (
        _close(selected_ensemble["confidence_threshold"], best_ensemble["confidence_threshold"])
        and _close(selected_ensemble["nms_iou"], best_ensemble["nms_iou"])
    )
    _check(
        checks,
        "ensemble_selected_on_validation",
        ensemble_selection_ok,
        "selected="
        f"({selected_ensemble['confidence_threshold']}, {selected_ensemble['nms_iou']}), "
        f"validation_best=({best_ensemble['confidence_threshold']}, {best_ensemble['nms_iou']})",
    )

    selected_gamma = float(selected["gamma_correction"]["gamma"])
    best_gamma = float(gamma_tuning.iloc[0]["gamma"])
    _check(
        checks,
        "gamma_selected_on_validation",
        _close(selected_gamma, best_gamma),
        f"selected={selected_gamma}, validation_best={best_gamma}",
    )

    test_image_counts_ok = (
        summary["image_count"].astype(int)
        == len(test_ids)
    ).all()
    _check(
        checks,
        "test_scope_uses_configured_images",
        bool(test_image_counts_ok),
        f"expected={len(test_ids)}, observed="
        f"{sorted(int(value) for value in summary['image_count'].unique())}",
    )

    key_rows = summary.set_index("condition")
    original = key_rows.loc["original"]
    lowlight = key_rows.loc["lowlight_baseline"]
    gamma = key_rows.loc["gamma_correction"]
    blur4 = key_rows.loc["blur_sigma_4"]
    _check(
        checks,
        "failure_cases_are_quantitatively_visible",
        float(original.f1 - blur4.f1) >= 0.15
        and float(original.f1 - key_rows.loc["prompt_hypernym"].f1) >= 0.15,
        "image_blur_delta="
        f"{float(original.f1 - blur4.f1):.4f}, prompt_delta="
        f"{float(original.f1 - key_rows.loc['prompt_hypernym'].f1):.4f}",
    )
    _check(
        checks,
        "gamma_improves_lowlight_test",
        float(gamma.f1) > float(lowlight.f1),
        f"baseline_f1={lowlight.f1:.4f}, gamma_f1={gamma.f1:.4f}",
    )

    strategy_aggregation_failures: list[str] = []
    for row in prompt_metrics.itertuples(index=False):
        group = prompt_details[
            (prompt_details["scope"] == row.scope)
            & (prompt_details["strategy"] == row.strategy)
        ]
        observed = {
            key: int(group[key].sum()) for key in ("tp", "fp", "fn")
        }
        expected = {
            "tp": int(row.tp),
            "fp": int(row.fp),
            "fn": int(row.fn),
        }
        expected_count = (
            len(validation_ids) if row.scope == "validation" else len(test_ids)
        )
        if observed != expected or len(group) != expected_count:
            strategy_aggregation_failures.append(
                f"{row.scope}/{row.strategy}: observed={observed}, "
                f"expected={expected}, rows={len(group)}"
            )
    _check(
        checks,
        "prompt_strategy_summary_matches_per_image",
        not strategy_aggregation_failures,
        (
            "ok"
            if not strategy_aggregation_failures
            else "; ".join(strategy_aggregation_failures)
        ),
    )

    validation_strategies = prompt_metrics[
        prompt_metrics["scope"] == "validation"
    ].sort_values(
        ["f1", "recall", "precision"],
        ascending=[False, False, False],
    )
    validation_best_strategy = str(
        validation_strategies.iloc[0]["strategy"]
    )
    _check(
        checks,
        "primary_prompt_strategy_selected_on_validation",
        prompt_selected["primary_strategy"] == validation_best_strategy,
        f"selected={prompt_selected['primary_strategy']}, "
        f"validation_best={validation_best_strategy}",
    )

    required_prompt_strategies = {
        "canonical_baseline",
        "naive_three_prompt_nms",
        "category_best_fixed",
        "category_best_calibrated",
        "category_subset_nms",
        "reliability_weighted_fusion",
    }
    observed_prompt_strategies = set(prompt_metrics["strategy"])
    _check(
        checks,
        "prompt_strategy_suite_present",
        required_prompt_strategies <= observed_prompt_strategies,
        f"missing={sorted(required_prompt_strategies - observed_prompt_strategies)}",
    )

    core_ap_columns = ["ap", "ap50", "ap75", "ar_1", "ar_10", "ar_100"]
    ap_ranges_ok = all(
        detection_ap[column].between(0.0, 1.0, inclusive="both").all()
        for column in core_ap_columns
    )
    baseline_ap_rows = detection_ap[
        detection_ap["condition"] == "strategy/canonical_baseline"
    ]
    _check(
        checks,
        "coco_ap_ar_metrics_valid",
        ap_ranges_ok
        and set(baseline_ap_rows["scope"]) == {"validation", "test"},
        "core AP/AR values are in [0, 1] and baseline covers both splits",
    )

    expected_bootstrap_samples = int(
        config["evaluation"]["bootstrap_samples"]
    )
    bootstrap_ok = (
        (bootstrap["ci_low"] <= bootstrap["estimate"]).all()
        and (bootstrap["estimate"] <= bootstrap["ci_high"]).all()
        and (
            bootstrap["bootstrap_samples"].astype(int)
            == expected_bootstrap_samples
        ).all()
        and (bootstrap_delta["ci_low"] <= bootstrap_delta["estimate"]).all()
        and (
            bootstrap_delta["estimate"] <= bootstrap_delta["ci_high"]
        ).all()
    )
    _check(
        checks,
        "bootstrap_intervals_are_well_formed",
        bool(bootstrap_ok),
        f"samples={expected_bootstrap_samples}, "
        f"interval_rows={len(bootstrap)}, delta_rows={len(bootstrap_delta)}",
    )

    manifests = [
        manifest,
        holdout_manifest,
        final_manifest,
        same_model_manifest,
        prompt_prototype_manifest,
    ]
    manifest_id_sets = [
        {int(image["image_id"]) for image in item["images"]}
        for item in manifests
    ]
    manifest_sizes_ok = all(
        len(item["images"]) == int(config["selection"]["holdout_images"])
        for item in manifests
    )
    mutually_disjoint = all(
        manifest_id_sets[left].isdisjoint(manifest_id_sets[right])
        for left in range(len(manifest_id_sets))
        for right in range(left + 1, len(manifest_id_sets))
    )
    _check(
        checks,
        "five_manifests_are_complete_and_disjoint",
        manifest_sizes_ok and mutually_disjoint,
        "sizes="
        f"{[len(item['images']) for item in manifests]}, "
        f"pairwise_overlaps="
        f"{[len(manifest_id_sets[left] & manifest_id_sets[right]) for left in range(len(manifest_id_sets)) for right in range(left + 1, len(manifest_id_sets))]}",
    )

    manifest_balance: list[dict[str, int]] = []
    for item in manifests:
        counts = pd.Series(
            [image["primary_category"] for image in item["images"]]
        ).value_counts()
        manifest_balance.append(
            {
                category: int(counts.get(category, 0))
                for category in config["categories"]
            }
        )
    expected_quota = int(config["selection"]["holdout_images"]) // len(
        config["categories"]
    )
    _check(
        checks,
        "development_manifests_are_category_balanced",
        all(
            all(count == expected_quota for count in counts.values())
            for counts in manifest_balance[:3]
        ),
        f"development_counts={manifest_balance[:3]}",
    )
    expected_same_model_quotas = {
        category: int(count)
        for category, count in config["same_model_improvement"][
            "final_holdout_quotas"
        ].items()
    }
    _check(
        checks,
        "same_model_holdout_uses_frozen_quotas",
        manifest_balance[3] == expected_same_model_quotas,
        f"observed={manifest_balance[3]}, expected={expected_same_model_quotas}",
    )
    expected_prompt_prototype_quotas = {
        category: int(count)
        for category, count in config["prompt_prototype_improvement"][
            "final_holdout_quotas"
        ].items()
    }
    _check(
        checks,
        "prompt_prototype_holdout_uses_frozen_quotas",
        manifest_balance[4] == expected_prompt_prototype_quotas,
        f"observed={manifest_balance[4]}, "
        f"expected={expected_prompt_prototype_quotas}",
    )

    final_expected_conditions = {
        "small_canonical_baseline",
        "small_threshold_tuned_ablation",
        "medium_model_improvement",
    }
    _check(
        checks,
        "model_scale_reference_conditions_present",
        set(model_final_summary["condition"]) == final_expected_conditions,
        f"observed={sorted(model_final_summary['condition'].tolist())}",
    )

    final_aggregation_failures: list[str] = []
    final_formula_failures: list[str] = []
    for row in model_final_summary.itertuples(index=False):
        group = model_final_details[
            model_final_details["condition"] == row.condition
        ]
        observed = {
            key: int(group[key].sum()) for key in ("tp", "fp", "fn")
        }
        expected = {
            "tp": int(row.tp),
            "fp": int(row.fp),
            "fn": int(row.fn),
        }
        if observed != expected or len(group) != int(row.image_count):
            final_aggregation_failures.append(
                f"{row.condition}: observed={observed}, "
                f"expected={expected}, rows={len(group)}"
            )
        precision = row.tp / (row.tp + row.fp) if row.tp + row.fp else 0.0
        recall = row.tp / (row.tp + row.fn) if row.tp + row.fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        if not (
            _close(row.precision, precision)
            and _close(row.recall, recall)
            and _close(row.f1, f1)
            and int(row.ground_truth_count) == int(row.tp + row.fn)
            and int(row.prediction_count) == int(row.tp + row.fp)
            and int(row.image_count) == len(final_manifest["images"])
        ):
            final_formula_failures.append(str(row.condition))
    _check(
        checks,
        "model_scale_reference_summary_matches_per_image",
        not final_aggregation_failures,
        (
            "ok"
            if not final_aggregation_failures
            else "; ".join(final_aggregation_failures)
        ),
    )
    _check(
        checks,
        "model_scale_reference_metric_formulas",
        not final_formula_failures,
        (
            "ok"
            if not final_formula_failures
            else f"failed={final_formula_failures}"
        ),
    )

    final_delta_well_formed = (
        (model_final_deltas["ci_low"] <= model_final_deltas["estimate"]).all()
        and (
            model_final_deltas["estimate"] <= model_final_deltas["ci_high"]
        ).all()
        and (
            model_final_deltas["bootstrap_samples"].astype(int)
            == expected_bootstrap_samples
        ).all()
    )
    _check(
        checks,
        "model_scale_reference_bootstrap_is_well_formed",
        bool(final_delta_well_formed),
        f"rows={len(model_final_deltas)}, samples={expected_bootstrap_samples}",
    )

    selected_hash = hashlib.sha256(model_selected_bytes).hexdigest()
    final_manifest_hash = hashlib.sha256(
        final_manifest_path.read_bytes()
    ).hexdigest()
    frozen_hashes_ok = (
        model_final_verdict["selected_parameters_sha256"] == selected_hash
        and model_final_verdict["holdout_manifest_sha256"]
        == final_manifest_hash
        and bool(model_final_verdict["selection_was_frozen_before_holdout"])
    )
    _check(
        checks,
        "model_scale_reference_selection_was_frozen",
        frozen_hashes_ok,
        f"selected_sha256={selected_hash}, manifest_sha256={final_manifest_hash}",
    )

    same_expected_pairs = {
        (method, condition)
        for method in ("baseline", "blur_aware_wiener")
        for condition in ("clean", "blur_sigma_2", "blur_sigma_4")
    }
    same_observed_pairs = set(
        zip(same_model_summary["method"], same_model_summary["condition"])
    )
    _check(
        checks,
        "same_model_final_conditions_present",
        same_observed_pairs == same_expected_pairs,
        f"observed={sorted(same_observed_pairs)}",
    )

    same_aggregation_failures: list[str] = []
    same_formula_failures: list[str] = []
    for row in same_model_summary.itertuples(index=False):
        group = same_model_details[
            (same_model_details["method"] == row.method)
            & (same_model_details["condition"] == row.condition)
        ]
        observed = {
            key: int(group[key].sum()) for key in ("tp", "fp", "fn")
        }
        expected = {
            "tp": int(row.tp),
            "fp": int(row.fp),
            "fn": int(row.fn),
        }
        if observed != expected or len(group) != int(row.image_count):
            same_aggregation_failures.append(
                f"{row.method}/{row.condition}: observed={observed}, "
                f"expected={expected}, rows={len(group)}"
            )
        precision = row.tp / (row.tp + row.fp) if row.tp + row.fp else 0.0
        recall = row.tp / (row.tp + row.fn) if row.tp + row.fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        if not (
            _close(row.precision, precision)
            and _close(row.recall, recall)
            and _close(row.f1, f1)
            and int(row.ground_truth_count) == int(row.tp + row.fn)
            and int(row.prediction_count) == int(row.tp + row.fp)
            and int(row.image_count) == len(same_model_manifest["images"])
            and 0.0 <= float(row.ap) <= 1.0
            and 0.0 <= float(row.ap50) <= 1.0
            and 0.0 <= float(row.ap75) <= 1.0
        ):
            same_formula_failures.append(f"{row.method}/{row.condition}")
    _check(
        checks,
        "same_model_summary_matches_per_image",
        not same_aggregation_failures,
        (
            "ok"
            if not same_aggregation_failures
            else "; ".join(same_aggregation_failures)
        ),
    )
    _check(
        checks,
        "same_model_metric_formulas_and_ap_ranges",
        not same_formula_failures,
        (
            "ok"
            if not same_formula_failures
            else f"failed={same_formula_failures}"
        ),
    )

    same_bootstrap_ok = (
        (same_model_bootstrap["ci_low"] <= same_model_bootstrap["estimate"]).all()
        and (
            same_model_bootstrap["estimate"]
            <= same_model_bootstrap["ci_high"]
        ).all()
        and (
            same_model_bootstrap["bootstrap_samples"].astype(int)
            == expected_bootstrap_samples
        ).all()
    )
    _check(
        checks,
        "same_model_bootstrap_is_well_formed",
        bool(same_bootstrap_ok),
        f"rows={len(same_model_bootstrap)}, samples={expected_bootstrap_samples}",
    )

    same_selected_hash = hashlib.sha256(
        same_model_selected_bytes
    ).hexdigest()
    same_manifest_hash = hashlib.sha256(
        same_model_manifest_path.read_bytes()
    ).hexdigest()
    same_frozen_ok = (
        same_model_verdict["selected_parameters_sha256"]
        == same_selected_hash
        and same_model_verdict["holdout_manifest_sha256"]
        == same_manifest_hash
        and bool(same_model_verdict["selection_was_frozen_before_holdout"])
        and bool(same_model_verdict["holdout_evaluated_once"])
    )
    _check(
        checks,
        "same_model_selection_and_manifest_are_frozen",
        same_frozen_ok,
        f"selected_sha256={same_selected_hash}, "
        f"manifest_sha256={same_manifest_hash}",
    )

    fixed_detector_ok = (
        same_model_verdict["model"] == config["model"]["name"]
        and not bool(same_model_verdict["model_change_used"])
        and not bool(same_model_verdict["training_used"])
        and not bool(same_model_verdict["super_resolution_used"])
    )
    _check(
        checks,
        "same_detector_is_fixed_without_training_or_super_resolution",
        fixed_detector_ok,
        f"model={same_model_verdict['model']}, "
        f"model_change={same_model_verdict['model_change_used']}, "
        f"training={same_model_verdict['training_used']}, "
        f"super_resolution={same_model_verdict['super_resolution_used']}",
    )

    same_rows = same_model_summary.set_index(["method", "condition"])
    baseline_blur_mean = (
        float(same_rows.loc[("baseline", "blur_sigma_2")].f1)
        + float(same_rows.loc[("baseline", "blur_sigma_4")].f1)
    ) / 2.0
    candidate_blur_mean = (
        float(same_rows.loc[("blur_aware_wiener", "blur_sigma_2")].f1)
        + float(same_rows.loc[("blur_aware_wiener", "blur_sigma_4")].f1)
    ) / 2.0
    clean_delta = (
        float(same_rows.loc[("blur_aware_wiener", "clean")].f1)
        - float(same_rows.loc[("baseline", "clean")].f1)
    )
    blur_delta_row = same_model_bootstrap[
        (same_model_bootstrap["comparison"] == "blur_mean")
        & (same_model_bootstrap["metric"] == "mean_condition_f1")
    ].iloc[0]
    clean_delta_row = same_model_bootstrap[
        (same_model_bootstrap["comparison"] == "clean")
        & (same_model_bootstrap["metric"] == "f1")
    ].iloc[0]
    recomputed_deltas_ok = (
        _close(
            baseline_blur_mean,
            same_model_verdict["baseline_blur_mean_f1"],
        )
        and _close(
            candidate_blur_mean,
            same_model_verdict["candidate_blur_mean_f1"],
        )
        and _close(
            candidate_blur_mean - baseline_blur_mean,
            blur_delta_row["estimate"],
        )
        and _close(clean_delta, clean_delta_row["estimate"])
    )
    _check(
        checks,
        "same_model_verdict_matches_recomputed_metrics",
        recomputed_deltas_ok,
        f"blur_delta={candidate_blur_mean - baseline_blur_mean:.6f}, "
        f"clean_delta={clean_delta:.6f}",
    )

    minimum_gain = float(
        config["same_model_improvement"]["minimum_blur_mean_f1_gain"]
    )
    maximum_clean_drop = float(
        config["same_model_improvement"]["maximum_clean_f1_drop"]
    )
    same_gate_ok = (
        bool(same_model_verdict["passed"])
        and candidate_blur_mean - baseline_blur_mean >= minimum_gain
        and float(blur_delta_row["ci_low"]) > 0.0
        and clean_delta >= -maximum_clean_drop
    )
    _check(
        checks,
        "same_model_improvement_passes_strict_gate",
        same_gate_ok,
        f"blur_f1_delta={candidate_blur_mean - baseline_blur_mean:.4f}, "
        f"ci_low={float(blur_delta_row['ci_low']):.4f}, "
        f"clean_f1_delta={clean_delta:.4f}",
    )

    expected_detector_rows = 3 * len(same_model_manifest["images"])
    exact_state_count = int(
        (
            same_model_detector["condition"]
            == same_model_detector["detected_state"]
        ).sum()
    )
    blur_rows = same_model_detector[
        same_model_detector["condition"].isin(
            ["blur_sigma_2", "blur_sigma_4"]
        )
    ]
    blur_false_negatives = int(
        (blur_rows["detected_state"] == "clean").sum()
    )
    detector_ok = (
        len(same_model_detector) == expected_detector_rows
        and exact_state_count / expected_detector_rows >= 0.98
        and blur_false_negatives == 0
    )
    _check(
        checks,
        "same_model_blur_detector_behavior_is_recorded",
        detector_ok,
        f"exact={exact_state_count}/{expected_detector_rows}, "
        f"blur_false_negatives={blur_false_negatives}",
    )

    prompt_anchor_weight = float(
        prompt_prototype_verdict["anchor_weight"]
    )
    prompt_candidate_method = (
        f"canonical_anchor_{prompt_anchor_weight:.2f}"
    )
    prompt_expected_pairs = {
        (method, condition)
        for method in ("raw_prompt", prompt_candidate_method)
        for condition in ("canonical", "synonym", "hypernym")
    }
    prompt_observed_pairs = set(
        zip(
            prompt_prototype_summary["method"],
            prompt_prototype_summary["condition"],
        )
    )
    _check(
        checks,
        "prompt_prototype_final_conditions_present",
        prompt_observed_pairs == prompt_expected_pairs,
        f"observed={sorted(prompt_observed_pairs)}",
    )

    prompt_aggregation_failures: list[str] = []
    prompt_formula_failures: list[str] = []
    prompt_ground_truth_count = sum(
        len(image["annotations"])
        for image in prompt_prototype_manifest["images"]
    )
    for row in prompt_prototype_summary.itertuples(index=False):
        group = prompt_prototype_details[
            (prompt_prototype_details["method"] == row.method)
            & (prompt_prototype_details["condition"] == row.condition)
        ]
        observed = {
            key: int(group[key].sum()) for key in ("tp", "fp", "fn")
        }
        expected = {
            "tp": int(row.tp),
            "fp": int(row.fp),
            "fn": int(row.fn),
        }
        if observed != expected or len(group) != int(row.image_count):
            prompt_aggregation_failures.append(
                f"{row.method}/{row.condition}: observed={observed}, "
                f"expected={expected}, rows={len(group)}"
            )
        precision = row.tp / (row.tp + row.fp) if row.tp + row.fp else 0.0
        recall = row.tp / (row.tp + row.fn) if row.tp + row.fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        if not (
            _close(row.precision, precision)
            and _close(row.recall, recall)
            and _close(row.f1, f1)
            and int(row.ground_truth_count) == int(row.tp + row.fn)
            and int(row.prediction_count) == int(row.tp + row.fp)
            and int(row.ground_truth_count) == prompt_ground_truth_count
            and int(row.image_count)
            == len(prompt_prototype_manifest["images"])
            and 0.0 <= float(row.ap) <= 1.0
            and 0.0 <= float(row.ap50) <= 1.0
            and 0.0 <= float(row.ap75) <= 1.0
        ):
            prompt_formula_failures.append(
                f"{row.method}/{row.condition}"
            )
    _check(
        checks,
        "prompt_prototype_summary_matches_per_image",
        not prompt_aggregation_failures,
        (
            "ok"
            if not prompt_aggregation_failures
            else "; ".join(prompt_aggregation_failures)
        ),
    )
    _check(
        checks,
        "prompt_prototype_metric_formulas_and_ap_ranges",
        not prompt_formula_failures,
        (
            "ok"
            if not prompt_formula_failures
            else f"failed={prompt_formula_failures}"
        ),
    )

    prompt_bootstrap_ok = (
        (prompt_prototype_bootstrap["ci_low"]
         <= prompt_prototype_bootstrap["estimate"]).all()
        and (
            prompt_prototype_bootstrap["estimate"]
            <= prompt_prototype_bootstrap["ci_high"]
        ).all()
        and (
            prompt_prototype_bootstrap["bootstrap_samples"].astype(int)
            == expected_bootstrap_samples
        ).all()
        and set(prompt_prototype_bootstrap["comparison"])
        == {"target_prompt_mean", "canonical_guardrail"}
    )
    _check(
        checks,
        "prompt_prototype_bootstrap_is_well_formed",
        bool(prompt_bootstrap_ok),
        f"rows={len(prompt_prototype_bootstrap)}, "
        f"samples={expected_bootstrap_samples}",
    )

    prompt_selected_hash = hashlib.sha256(
        prompt_prototype_selected_bytes
    ).hexdigest()
    prompt_manifest_hash = hashlib.sha256(
        prompt_prototype_manifest_path.read_bytes()
    ).hexdigest()
    prompt_frozen_ok = (
        prompt_prototype_verdict["selected_parameters_sha256"]
        == prompt_selected_hash
        and prompt_prototype_verdict["holdout_manifest_sha256"]
        == prompt_manifest_hash
        and bool(
            prompt_prototype_verdict[
                "selection_was_frozen_before_holdout"
            ]
        )
        and bool(prompt_prototype_verdict["holdout_evaluated_once"])
        and prompt_prototype_verdict["selection_scope"]
        == "development_400_only"
    )
    _check(
        checks,
        "prompt_prototype_selection_and_manifest_are_frozen",
        prompt_frozen_ok,
        f"selected_sha256={prompt_selected_hash}, "
        f"manifest_sha256={prompt_manifest_hash}",
    )

    development_manifest_paths = {
        "subset_manifest": manifest_path,
        "holdout_manifest": holdout_manifest_path,
        "final_holdout_manifest": final_manifest_path,
        "same_model_holdout_manifest": same_model_manifest_path,
    }
    current_development_hashes = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in development_manifest_paths.items()
    }
    prompt_development_ok = (
        prompt_prototype_selected["development_manifest_sha256"]
        == current_development_hashes
        and bool(prompt_prototype_selected["development_gate_passed"])
        and prompt_prototype_selected["selection_scope"]
        == "development_400_only"
        and _close(
            prompt_anchor_weight,
            prompt_prototype_selected["anchor_weight"],
        )
        and prompt_anchor_weight
        in {
            float(value)
            for value in config["prompt_prototype_improvement"][
                "anchor_weights"
            ]
        }
    )
    _check(
        checks,
        "prompt_prototype_development_inputs_and_choice_are_fixed",
        prompt_development_ok,
        f"anchor_weight={prompt_anchor_weight}, "
        f"development_hashes={current_development_hashes}",
    )

    prompt_model_ok = (
        prompt_prototype_verdict["model"] == config["model"]["name"]
        and prompt_prototype_selected["model"] == config["model"]["name"]
        and not bool(prompt_prototype_verdict["model_change_used"])
        and not bool(prompt_prototype_verdict["training_used"])
        and not bool(
            prompt_prototype_verdict["prompt_box_union_used"]
        )
        and not bool(prompt_prototype_selected["model_change_used"])
        and not bool(prompt_prototype_selected["training_used"])
        and not bool(prompt_prototype_selected["prompt_box_union_used"])
    )
    _check(
        checks,
        "prompt_prototype_uses_fixed_model_without_training_or_box_union",
        prompt_model_ok,
        f"model={prompt_prototype_verdict['model']}, "
        f"model_change={prompt_prototype_verdict['model_change_used']}, "
        f"training={prompt_prototype_verdict['training_used']}, "
        f"box_union={prompt_prototype_verdict['prompt_box_union_used']}",
    )

    prompt_rows = prompt_prototype_summary.set_index(
        ["method", "condition"]
    )
    target_conditions = list(
        prompt_prototype_verdict["target_conditions"]
    )
    baseline_target_mean_f1 = sum(
        float(prompt_rows.loc[("raw_prompt", condition)].f1)
        for condition in target_conditions
    ) / len(target_conditions)
    candidate_target_mean_f1 = sum(
        float(
            prompt_rows.loc[(prompt_candidate_method, condition)].f1
        )
        for condition in target_conditions
    ) / len(target_conditions)
    baseline_target_mean_ap = sum(
        float(prompt_rows.loc[("raw_prompt", condition)].ap)
        for condition in target_conditions
    ) / len(target_conditions)
    candidate_target_mean_ap = sum(
        float(
            prompt_rows.loc[(prompt_candidate_method, condition)].ap
        )
        for condition in target_conditions
    ) / len(target_conditions)
    canonical_f1_delta = (
        float(
            prompt_rows.loc[
                (prompt_candidate_method, "canonical")
            ].f1
        )
        - float(prompt_rows.loc[("raw_prompt", "canonical")].f1)
    )
    prompt_target_delta_row = prompt_prototype_bootstrap[
        (prompt_prototype_bootstrap["comparison"] == "target_prompt_mean")
        & (
            prompt_prototype_bootstrap["metric"]
            == "mean_condition_f1"
        )
    ].iloc[0]
    prompt_canonical_delta_row = prompt_prototype_bootstrap[
        (
            prompt_prototype_bootstrap["comparison"]
            == "canonical_guardrail"
        )
        & (prompt_prototype_bootstrap["metric"] == "f1")
    ].iloc[0]
    prompt_recomputed_ok = (
        _close(
            baseline_target_mean_f1,
            prompt_prototype_verdict["baseline_target_mean_f1"],
        )
        and _close(
            candidate_target_mean_f1,
            prompt_prototype_verdict["candidate_target_mean_f1"],
        )
        and _close(
            candidate_target_mean_f1 - baseline_target_mean_f1,
            prompt_target_delta_row["estimate"],
        )
        and _close(
            baseline_target_mean_ap,
            prompt_prototype_verdict["baseline_target_mean_ap"],
        )
        and _close(
            candidate_target_mean_ap,
            prompt_prototype_verdict["candidate_target_mean_ap"],
        )
        and _close(
            candidate_target_mean_ap - baseline_target_mean_ap,
            prompt_prototype_verdict["target_mean_ap_delta"],
        )
        and _close(
            canonical_f1_delta,
            prompt_canonical_delta_row["estimate"],
        )
    )
    _check(
        checks,
        "prompt_prototype_verdict_matches_recomputed_metrics",
        prompt_recomputed_ok,
        f"target_f1_delta="
        f"{candidate_target_mean_f1 - baseline_target_mean_f1:.6f}, "
        f"target_map_delta="
        f"{candidate_target_mean_ap - baseline_target_mean_ap:.6f}, "
        f"canonical_f1_delta={canonical_f1_delta:.6f}",
    )

    raw_canonical = (
        prompt_prototype_details[
            (prompt_prototype_details["method"] == "raw_prompt")
            & (prompt_prototype_details["condition"] == "canonical")
        ]
        .sort_values("image_id")
        .reset_index(drop=True)
    )
    candidate_canonical = (
        prompt_prototype_details[
            (
                prompt_prototype_details["method"]
                == prompt_candidate_method
            )
            & (prompt_prototype_details["condition"] == "canonical")
        ]
        .sort_values("image_id")
        .reset_index(drop=True)
    )
    canonical_columns = [
        "image_id",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
        "mean_tp_confidence",
    ]
    canonical_unchanged = (
        len(raw_canonical) == len(candidate_canonical)
        and raw_canonical[canonical_columns].equals(
            candidate_canonical[canonical_columns]
        )
    )
    _check(
        checks,
        "prompt_prototype_canonical_predictions_are_unchanged",
        canonical_unchanged,
        f"rows={len(raw_canonical)}, "
        f"candidate_rows={len(candidate_canonical)}",
    )

    prompt_minimum_gain = float(
        config["prompt_prototype_improvement"][
            "minimum_mean_f1_gain"
        ]
    )
    prompt_maximum_canonical_drop = float(
        config["prompt_prototype_improvement"][
            "maximum_canonical_f1_drop"
        ]
    )
    prompt_maximum_map_drop = float(
        config["prompt_prototype_improvement"][
            "maximum_mean_map_drop"
        ]
    )
    prompt_minimum_fold_gain = float(
        config["prompt_prototype_improvement"]["minimum_fold_gain"]
    )
    prompt_gate_ok = (
        bool(prompt_prototype_verdict["passed"])
        and (
            candidate_target_mean_f1 - baseline_target_mean_f1
            >= prompt_minimum_gain
        )
        and float(prompt_target_delta_row["ci_low"]) > 0.0
        and canonical_f1_delta >= -prompt_maximum_canonical_drop
        and (
            candidate_target_mean_ap - baseline_target_mean_ap
            >= -prompt_maximum_map_drop
        )
        and float(
            prompt_prototype_selected["metrics"]["minimum_fold_gain"]
        )
        >= prompt_minimum_fold_gain
    )
    _check(
        checks,
        "prompt_prototype_improvement_passes_strict_gate",
        prompt_gate_ok,
        f"target_f1_delta="
        f"{candidate_target_mean_f1 - baseline_target_mean_f1:.4f}, "
        f"ci_low={float(prompt_target_delta_row['ci_low']):.4f}, "
        f"target_map_delta="
        f"{candidate_target_mean_ap - baseline_target_mean_ap:.4f}, "
        f"canonical_f1_delta={canonical_f1_delta:.4f}",
    )

    passed = all(check["passed"] for check in checks)
    payload = {
        "status": "pass" if passed else "fail",
        "check_count": len(checks),
        "passed_count": sum(check["passed"] for check in checks),
        "checks": checks,
    }
    json_path = results_dir / "validation_results.json"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    report_lines = [
        "# Validation Report",
        "",
        f"Overall status: **{payload['status'].upper()}** "
        f"({payload['passed_count']}/{payload['check_count']} checks passed)",
        "",
        "| Check | Status | Detail |",
        "|---|---:|---|",
    ]
    for check in checks:
        detail = str(check["detail"]).replace("|", "\\|")
        status = "PASS" if check["passed"] else "FAIL"
        report_lines.append(f"| {check['name']} | {status} | {detail} |")
    report_lines.extend(
        [
            "",
            "The checks cover subset integrity, split isolation, aggregate consistency, "
            "metric formulas, validation-only parameter selection, fixed test scope, "
            "failure-case strength, low-light improvement, five-way manifest isolation, "
            "frozen final-holdout inputs, fixed-detector constraints, and the strict "
            "blur and prompt improvement gates.",
            "",
        ]
    )
    report_path = results_dir / "VALIDATION_REPORT.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    if not passed:
        failed = [check["name"] for check in checks if not check["passed"]]
        raise RuntimeError(f"Result validation failed: {failed}")
    return report_path
