from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

from .config import resolve_path


REPORT_BASENAME = "YOLO_World_Failure_Case_Report"


def _escape_latex(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in text)


def _format_number(value: Any, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def _format_signed_number(value: Any, digits: int = 3) -> str:
    return f"{float(value):+.{digits}f}"


def _macro(name: str, value: Any) -> str:
    return rf"\providecommand{{\{name}}}{{{value}}}"


def create_latex_metrics(config: dict[str, Any]) -> Path:
    project_root = Path(config["_project_root"])
    results_dir = resolve_path(config, config["paths"]["results_dir"])
    report_dir = project_root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(results_dir / "summary_metrics.csv")
    summary_test = summary[summary["scope"] == "test"].set_index("condition")
    strategies = pd.read_csv(results_dir / "prompt_strategy_metrics.csv")
    strategy_validation = strategies[
        strategies["scope"] == "validation"
    ].set_index("strategy")
    strategy_test = strategies[strategies["scope"] == "test"].set_index(
        "strategy"
    )
    prompt_categories = pd.read_csv(
        results_dir / "prompt_variant_by_category.csv"
    )
    prompt_category_test = prompt_categories[
        prompt_categories["scope"] == "test"
    ].set_index(["variant", "category"])
    detection_ap = pd.read_csv(results_dir / "detection_ap_metrics.csv")
    ap_test = detection_ap[detection_ap["scope"] == "test"].set_index(
        "condition"
    )
    intervals = pd.read_csv(results_dir / "bootstrap_intervals.csv")
    baseline_intervals = intervals[
        (intervals["condition"] == "strategy/canonical_baseline")
    ].set_index("metric")
    deltas = pd.read_csv(results_dir / "bootstrap_deltas_vs_baseline.csv")
    f1_deltas = deltas[deltas["metric"] == "f1"].set_index("candidate")
    prompt_selected = json.loads(
        (results_dir / "prompt_selected_parameters.json").read_text(
            encoding="utf-8"
        )
    )
    environment = json.loads(
        (results_dir / "environment_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    manifest = json.loads(
        resolve_path(config, config["paths"]["subset_manifest"]).read_text(
            encoding="utf-8"
        )
    )
    improvement_dir = resolve_path(
        config, config["paths"]["improvement_results_dir"]
    )
    final_summary = pd.read_csv(
        improvement_dir / "model_final_holdout_summary.csv"
    ).set_index("condition")
    final_verdict = json.loads(
        (improvement_dir / "model_final_holdout_verdict.json").read_text(
            encoding="utf-8"
        )
    )
    selected_scale = json.loads(
        (improvement_dir / "selected_model_scale.json").read_text(
            encoding="utf-8"
        )
    )
    same_model_dir = resolve_path(
        config, config["paths"]["same_model_results_dir"]
    )
    same_model_summary = pd.read_csv(
        same_model_dir / "same_model_final_summary.csv"
    ).set_index(["method", "condition"])
    same_model_verdict = json.loads(
        (same_model_dir / "same_model_final_verdict.json").read_text(
            encoding="utf-8"
        )
    )
    selected_same_model = json.loads(
        (same_model_dir / "selected_same_model_method.json").read_text(
            encoding="utf-8"
        )
    )
    same_model_detector = pd.read_csv(
        same_model_dir / "same_model_final_blur_detector.csv"
    )
    prompt_prototype_dir = resolve_path(
        config, config["paths"]["prompt_prototype_results_dir"]
    )
    prompt_prototype_summary = pd.read_csv(
        prompt_prototype_dir / "prompt_prototype_final_summary.csv"
    ).set_index(["method", "condition"])
    prompt_prototype_verdict = json.loads(
        (
            prompt_prototype_dir / "prompt_prototype_final_verdict.json"
        ).read_text(encoding="utf-8")
    )
    selected_prompt_prototype = json.loads(
        (
            prompt_prototype_dir / "selected_prompt_prototype_method.json"
        ).read_text(encoding="utf-8")
    )
    prompt_prototype_manifest = json.loads(
        resolve_path(
            config, config["paths"]["prompt_prototype_holdout_manifest"]
        ).read_text(encoding="utf-8")
    )
    prompt_prototype_development_image_count = sum(
        len(
            json.loads(
                resolve_path(config, config["paths"][key]).read_text(
                    encoding="utf-8"
                )
            )["images"]
        )
        for key in (
            "subset_manifest",
            "holdout_manifest",
            "final_holdout_manifest",
            "same_model_holdout_manifest",
        )
    )

    primary = str(prompt_selected["primary_strategy"])
    baseline = summary_test.loc["original"]
    lowlight = summary_test.loc["lowlight_baseline"]
    gamma = summary_test.loc["gamma_correction"]
    blur = summary_test.loc["blur_sigma_4"]
    primary_test = strategy_test.loc[primary]
    primary_ap = ap_test.loc[f"strategy/{primary}"]
    baseline_ap = ap_test.loc["strategy/canonical_baseline"]
    primary_delta = f1_deltas.loc[f"strategy/{primary}"]
    naive_delta = f1_deltas.loc["strategy/naive_three_prompt_nms"]
    fixed_delta = f1_deltas.loc["strategy/category_best_fixed"]
    final_small = final_summary.loc["small_canonical_baseline"]
    final_small_tuned = final_summary.loc[
        "small_threshold_tuned_ablation"
    ]
    final_medium = final_summary.loc["medium_model_improvement"]
    final_delta = final_verdict["f1_delta"]
    final_delta_tuned = final_verdict["f1_delta_vs_small_tuned"]
    same_base_clean = same_model_summary.loc[("baseline", "clean")]
    same_improved_clean = same_model_summary.loc[
        ("blur_aware_wiener", "clean")
    ]
    same_base_blur2 = same_model_summary.loc[
        ("baseline", "blur_sigma_2")
    ]
    same_improved_blur2 = same_model_summary.loc[
        ("blur_aware_wiener", "blur_sigma_2")
    ]
    same_base_blur4 = same_model_summary.loc[
        ("baseline", "blur_sigma_4")
    ]
    same_improved_blur4 = same_model_summary.loc[
        ("blur_aware_wiener", "blur_sigma_4")
    ]
    prompt_candidate_method = (
        f"canonical_anchor_{float(prompt_prototype_verdict['anchor_weight']):.2f}"
    )
    prompt_final_baseline = {
        condition: prompt_prototype_summary.loc[
            ("raw_prompt", condition)
        ]
        for condition in ("canonical", "synonym", "hypernym")
    }
    prompt_final_improved = {
        condition: prompt_prototype_summary.loc[
            (prompt_candidate_method, condition)
        ]
        for condition in ("canonical", "synonym", "hypernym")
    }
    clean_correct_count = int(
        (
            (same_model_detector["condition"] == "clean")
            & (same_model_detector["detected_state"] == "clean")
        ).sum()
    )
    total_detector_correct = int(
        (
            (
                (same_model_detector["condition"] == "clean")
                & (same_model_detector["detected_state"] == "clean")
            )
            | (
                (same_model_detector["condition"] == "blur_sigma_2")
                & (
                    same_model_detector["detected_state"]
                    == "blur_sigma_2"
                )
            )
            | (
                (same_model_detector["condition"] == "blur_sigma_4")
                & (
                    same_model_detector["detected_state"]
                    == "blur_sigma_4"
                )
            )
        ).sum()
    )

    validation_entries = [
        item for item in manifest["images"] if item["split"] == "validation"
    ]
    test_entries = [
        item for item in manifest["images"] if item["split"] == "test"
    ]
    validation_gt = sum(len(item["annotations"]) for item in validation_entries)
    test_gt = sum(len(item["annotations"]) for item in test_entries)
    validation_category_gt = {
        category: sum(
            annotation["category"] == category
            for item in validation_entries
            for annotation in item["annotations"]
        )
        for category in config["categories"]
    }
    test_small_gt = sum(
        1
        for item in test_entries
        for annotation in item["annotations"]
        if float(annotation["area"]) < 32**2
    )
    gamma_tuning = pd.read_csv(results_dir / "gamma_tuning.csv")
    best_gamma_f1 = float(gamma_tuning["f1"].max())
    gamma_validation_tie_count = int(
        (gamma_tuning["f1"].astype(float) - best_gamma_f1).abs().le(1e-12).sum()
    )
    subset_selection = prompt_selected["category_subset_nms"]
    subset_max_variant_count = max(
        len(parameters["variants"])
        for parameters in subset_selection.values()
    )

    lines = [
        "% Generated from experiment CSV/JSON files. Do not edit by hand.",
        _macro("TotalImageCount", len(manifest["images"])),
        _macro("ValidationImageCount", len(validation_entries)),
        _macro("TestImageCount", len(test_entries)),
        _macro("ValidationGTCount", validation_gt),
        _macro("TestGTCount", test_gt),
        _macro("ValidationCategoryGTMin", min(validation_category_gt.values())),
        _macro("ValidationCategoryGTMax", max(validation_category_gt.values())),
        _macro("TestSmallGTCount", test_small_gt),
        _macro(
            "ManifestPath",
            _escape_latex(config["paths"]["subset_manifest"]),
        ),
        _macro("GammaValidationTieCount", gamma_validation_tie_count),
        _macro("SubsetMaxVariantCount", subset_max_variant_count),
        _macro(
            "DevelopmentImageCount",
            int(selected_scale["dev_metrics"]["image_count"]),
        ),
        _macro(
            "SameModelDevelopmentImageCount",
            int(selected_same_model["bootstrap"]["image_count"]),
        ),
        _macro(
            "SameModelHoldoutImageCount",
            int(same_base_clean["image_count"]),
        ),
        _macro(
            "SameModelHoldoutGTCount",
            int(same_base_clean["ground_truth_count"]),
        ),
        _macro("SameModelName", _escape_latex(selected_same_model["model"])),
        _macro(
            "SameModelBaselineCleanFOne",
            _format_number(same_base_clean["f1"]),
        ),
        _macro(
            "SameModelImprovedCleanFOne",
            _format_number(same_improved_clean["f1"]),
        ),
        _macro(
            "SameModelCleanFOneDelta",
            _format_signed_number(
                same_model_verdict["clean_f1_delta"]["estimate"]
            ),
        ),
        _macro(
            "SameModelBaselineBlurTwoFOne",
            _format_number(same_base_blur2["f1"]),
        ),
        _macro(
            "SameModelImprovedBlurTwoFOne",
            _format_number(same_improved_blur2["f1"]),
        ),
        _macro(
            "SameModelBaselineBlurTwoAP",
            _format_number(same_base_blur2["ap"]),
        ),
        _macro(
            "SameModelImprovedBlurTwoAP",
            _format_number(same_improved_blur2["ap"]),
        ),
        _macro(
            "SameModelBaselineBlurFourFOne",
            _format_number(same_base_blur4["f1"]),
        ),
        _macro(
            "SameModelImprovedBlurFourFOne",
            _format_number(same_improved_blur4["f1"]),
        ),
        _macro(
            "SameModelBaselineBlurFourAP",
            _format_number(same_base_blur4["ap"]),
        ),
        _macro(
            "SameModelImprovedBlurFourAP",
            _format_number(same_improved_blur4["ap"]),
        ),
        _macro(
            "SameModelBaselineBlurFourRecall",
            _format_number(same_base_blur4["recall"]),
        ),
        _macro(
            "SameModelImprovedBlurFourRecall",
            _format_number(same_improved_blur4["recall"]),
        ),
        _macro(
            "SameModelBaselineBlurMeanFOne",
            _format_number(
                same_model_verdict["baseline_blur_mean_f1"]
            ),
        ),
        _macro(
            "SameModelImprovedBlurMeanFOne",
            _format_number(
                same_model_verdict["candidate_blur_mean_f1"]
            ),
        ),
        _macro(
            "SameModelBlurMeanFOneDelta",
            _format_signed_number(
                same_model_verdict["blur_mean_f1_delta"]["estimate"]
            ),
        ),
        _macro(
            "SameModelBlurMeanFOneDeltaLow",
            _format_signed_number(
                same_model_verdict["blur_mean_f1_delta"]["ci_low"]
            ),
        ),
        _macro(
            "SameModelBlurMeanFOneDeltaHigh",
            _format_signed_number(
                same_model_verdict["blur_mean_f1_delta"]["ci_high"]
            ),
        ),
        _macro(
            "SameModelDevelopmentFOneDelta",
            _format_signed_number(
                selected_same_model["metrics"]["mean_blur_f1_gain"]
            ),
        ),
        _macro(
            "SameModelDevelopmentFOneDeltaLow",
            _format_signed_number(
                selected_same_model["bootstrap"]["ci_low"]
            ),
        ),
        _macro(
            "SameModelDevelopmentFOneDeltaHigh",
            _format_signed_number(
                selected_same_model["bootstrap"]["ci_high"]
            ),
        ),
        _macro(
            "SameModelMinimumFoldGain",
            _format_signed_number(
                selected_same_model["metrics"]["minimum_fold_gain"]
            ),
        ),
        _macro(
            "WienerRegularization",
            _format_number(
                selected_same_model["mild_candidate"]["regularization"],
                3,
            ),
        ),
        _macro(
            "WienerBlend",
            _format_number(
                selected_same_model["mild_candidate"]["blend"], 2
            ),
        ),
        _macro(
            "BlurCleanThreshold",
            _format_number(
                selected_same_model["blur_detector"]["clean_threshold"],
                2,
            ),
        ),
        _macro(
            "BlurSeverityThreshold",
            _format_number(
                selected_same_model["blur_detector"][
                    "severe_threshold"
                ],
                2,
            ),
        ),
        _macro("SameModelDetectorCorrect", total_detector_correct),
        _macro("SameModelDetectorTotal", len(same_model_detector)),
        _macro("SameModelCleanCorrect", clean_correct_count),
        _macro(
            "PromptPrototypeDevelopmentImageCount",
            prompt_prototype_development_image_count,
        ),
        _macro(
            "PromptPrototypeHoldoutImageCount",
            len(prompt_prototype_manifest["images"]),
        ),
        _macro(
            "PromptPrototypeHoldoutGTCount",
            sum(
                len(item["annotations"])
                for item in prompt_prototype_manifest["images"]
            ),
        ),
        _macro(
            "PromptPrototypeAnchorWeight",
            _format_number(prompt_prototype_verdict["anchor_weight"], 2),
        ),
        _macro(
            "PromptPrototypeBaselineCanonicalFOne",
            _format_number(prompt_final_baseline["canonical"]["f1"]),
        ),
        _macro(
            "PromptPrototypeImprovedCanonicalFOne",
            _format_number(prompt_final_improved["canonical"]["f1"]),
        ),
        _macro(
            "PromptPrototypeBaselineSynonymFOne",
            _format_number(prompt_final_baseline["synonym"]["f1"]),
        ),
        _macro(
            "PromptPrototypeImprovedSynonymFOne",
            _format_number(prompt_final_improved["synonym"]["f1"]),
        ),
        _macro(
            "PromptPrototypeBaselineHypernymFOne",
            _format_number(prompt_final_baseline["hypernym"]["f1"]),
        ),
        _macro(
            "PromptPrototypeImprovedHypernymFOne",
            _format_number(prompt_final_improved["hypernym"]["f1"]),
        ),
        _macro(
            "PromptPrototypeBaselineTargetMeanFOne",
            _format_number(
                prompt_prototype_verdict["baseline_target_mean_f1"]
            ),
        ),
        _macro(
            "PromptPrototypeImprovedTargetMeanFOne",
            _format_number(
                prompt_prototype_verdict["candidate_target_mean_f1"]
            ),
        ),
        _macro(
            "PromptPrototypeTargetMeanFOneDelta",
            _format_signed_number(
                prompt_prototype_verdict["target_mean_f1_delta"][
                    "estimate"
                ]
            ),
        ),
        _macro(
            "PromptPrototypeTargetMeanFOneDeltaLow",
            _format_signed_number(
                prompt_prototype_verdict["target_mean_f1_delta"]["ci_low"]
            ),
        ),
        _macro(
            "PromptPrototypeTargetMeanFOneDeltaHigh",
            _format_signed_number(
                prompt_prototype_verdict["target_mean_f1_delta"]["ci_high"]
            ),
        ),
        _macro(
            "PromptPrototypeBaselineTargetMeanAP",
            _format_number(
                prompt_prototype_verdict["baseline_target_mean_ap"]
            ),
        ),
        _macro(
            "PromptPrototypeImprovedTargetMeanAP",
            _format_number(
                prompt_prototype_verdict["candidate_target_mean_ap"]
            ),
        ),
        _macro(
            "PromptPrototypeTargetMeanAPDelta",
            _format_signed_number(
                prompt_prototype_verdict["target_mean_ap_delta"]
            ),
        ),
        _macro(
            "PromptPrototypeDevelopmentFOneDelta",
            _format_signed_number(
                selected_prompt_prototype["metrics"][
                    "target_mean_f1_gain"
                ]
            ),
        ),
        _macro(
            "PromptPrototypeDevelopmentFOneDeltaLow",
            _format_signed_number(
                selected_prompt_prototype["metrics"][
                    "f1_delta_ci_low"
                ]
            ),
        ),
        _macro(
            "PromptPrototypeDevelopmentFOneDeltaHigh",
            _format_signed_number(
                selected_prompt_prototype["metrics"][
                    "f1_delta_ci_high"
                ]
            ),
        ),
        _macro(
            "PromptPrototypeMinimumFoldGain",
            _format_signed_number(
                selected_prompt_prototype["metrics"][
                    "minimum_fold_gain"
                ]
            ),
        ),
        _macro("FinalHoldoutImageCount", int(final_small["image_count"])),
        _macro(
            "FinalHoldoutGTCount", int(final_small["ground_truth_count"])
        ),
        _macro(
            "SmallFinalPrecision",
            _format_number(final_small["precision"]),
        ),
        _macro("SmallFinalRecall", _format_number(final_small["recall"])),
        _macro("SmallFinalFOne", _format_number(final_small["f1"])),
        _macro("SmallFinalTP", int(final_small["tp"])),
        _macro("SmallFinalFP", int(final_small["fp"])),
        _macro("SmallFinalFN", int(final_small["fn"])),
        _macro("SmallFinalAP", _format_number(final_small["ap"])),
        _macro(
            "SmallFinalAPFifty", _format_number(final_small["ap50"])
        ),
        _macro(
            "SmallFinalAPSeventyFive",
            _format_number(final_small["ap75"]),
        ),
        _macro(
            "SmallTunedFinalPrecision",
            _format_number(final_small_tuned["precision"]),
        ),
        _macro(
            "SmallTunedFinalRecall",
            _format_number(final_small_tuned["recall"]),
        ),
        _macro(
            "SmallTunedFinalFOne",
            _format_number(final_small_tuned["f1"]),
        ),
        _macro(
            "MediumFinalPrecision",
            _format_number(final_medium["precision"]),
        ),
        _macro(
            "MediumFinalRecall", _format_number(final_medium["recall"])
        ),
        _macro("MediumFinalFOne", _format_number(final_medium["f1"])),
        _macro("MediumFinalTP", int(final_medium["tp"])),
        _macro("MediumFinalFP", int(final_medium["fp"])),
        _macro("MediumFinalFN", int(final_medium["fn"])),
        _macro("MediumFinalAP", _format_number(final_medium["ap"])),
        _macro(
            "MediumFinalAPFifty", _format_number(final_medium["ap50"])
        ),
        _macro(
            "MediumFinalAPSeventyFive",
            _format_number(final_medium["ap75"]),
        ),
        _macro(
            "ModelFOneDelta",
            _format_signed_number(final_delta["estimate"]),
        ),
        _macro(
            "ModelFOneDeltaLow",
            _format_signed_number(final_delta["ci_low"]),
        ),
        _macro(
            "ModelFOneDeltaHigh",
            _format_signed_number(final_delta["ci_high"]),
        ),
        _macro(
            "ModelFOneDeltaVsSmallTuned",
            _format_signed_number(final_delta_tuned["estimate"]),
        ),
        _macro(
            "ModelFOneDeltaVsSmallTunedLow",
            _format_signed_number(final_delta_tuned["ci_low"]),
        ),
        _macro(
            "ModelFOneDeltaVsSmallTunedHigh",
            _format_signed_number(final_delta_tuned["ci_high"]),
        ),
        _macro(
            "ModelMAPDelta",
            _format_signed_number(final_verdict["map_delta"]),
        ),
        _macro(
            "MediumModelName", _escape_latex(selected_scale["model"])
        ),
        _macro("MediumImageSize", int(selected_scale["image_size"])),
        _macro(
            "MediumCarThreshold",
            _format_number(selected_scale["thresholds"]["car"], 2),
        ),
        _macro(
            "MediumCouchThreshold",
            _format_number(selected_scale["thresholds"]["couch"], 2),
        ),
        _macro(
            "MediumAirplaneThreshold",
            _format_number(selected_scale["thresholds"]["airplane"], 2),
        ),
        _macro(
            "MediumCupThreshold",
            _format_number(selected_scale["thresholds"]["cup"], 2),
        ),
        _macro("BaselinePrecision", _format_number(baseline["precision"])),
        _macro("BaselineRecall", _format_number(baseline["recall"])),
        _macro("BaselineFOne", _format_number(baseline["f1"])),
        _macro(
            "BaselineValidationFOne",
            _format_number(
                strategy_validation.loc["canonical_baseline", "f1"]
            ),
        ),
        _macro("BaselineTP", int(baseline["tp"])),
        _macro("BaselineFP", int(baseline["fp"])),
        _macro("BaselineFN", int(baseline["fn"])),
        _macro("BaselineAP", _format_number(baseline_ap["ap"])),
        _macro("BaselineAPFifty", _format_number(baseline_ap["ap50"])),
        _macro("BaselineAPSeventyFive", _format_number(baseline_ap["ap75"])),
        _macro("BaselineARHundred", _format_number(baseline_ap["ar_100"])),
        _macro("BaselineAPSmall", _format_number(baseline_ap["ap_small"])),
        _macro("BaselineAPMedium", _format_number(baseline_ap["ap_medium"])),
        _macro("BaselineAPLarge", _format_number(baseline_ap["ap_large"])),
        _macro(
            "BaselineFOneCILow",
            _format_number(baseline_intervals.loc["f1", "ci_low"]),
        ),
        _macro(
            "BaselineFOneCIHigh",
            _format_number(baseline_intervals.loc["f1", "ci_high"]),
        ),
        _macro("BlurPrecision", _format_number(blur["precision"])),
        _macro("BlurRecall", _format_number(blur["recall"])),
        _macro("BlurFOne", _format_number(blur["f1"])),
        _macro("BlurTP", int(blur["tp"])),
        _macro("BlurFP", int(blur["fp"])),
        _macro("BlurFN", int(blur["fn"])),
        _macro("LowlightPrecision", _format_number(lowlight["precision"])),
        _macro("LowlightRecall", _format_number(lowlight["recall"])),
        _macro("LowlightFOne", _format_number(lowlight["f1"])),
        _macro("GammaPrecision", _format_number(gamma["precision"])),
        _macro("GammaRecall", _format_number(gamma["recall"])),
        _macro("GammaFOne", _format_number(gamma["f1"])),
        _macro(
            "GammaValue",
            _format_number(
                json.loads(
                    (results_dir / "selected_parameters.json").read_text(
                        encoding="utf-8"
                    )
                )["gamma_correction"]["gamma"],
                2,
            ),
        ),
        _macro(
            "CanonicalPromptFOne",
            _format_number(summary_test.loc["prompt_canonical", "f1"]),
        ),
        _macro(
            "SynonymPromptFOne",
            _format_number(summary_test.loc["prompt_synonym", "f1"]),
        ),
        _macro(
            "HypernymPromptFOne",
            _format_number(summary_test.loc["prompt_hypernym", "f1"]),
        ),
        _macro(
            "CarCanonicalRecall",
            _format_number(
                prompt_category_test.loc[("canonical", "car"), "recall"]
            ),
        ),
        _macro(
            "CarSynonymRecall",
            _format_number(
                prompt_category_test.loc[("synonym", "car"), "recall"]
            ),
        ),
        _macro(
            "CarHypernymRecall",
            _format_number(
                prompt_category_test.loc[("hypernym", "car"), "recall"]
            ),
        ),
        _macro(
            "CouchPhotoRecall",
            _format_number(
                prompt_category_test.loc[
                    ("photo_template", "couch"), "recall"
                ]
            ),
        ),
        _macro(
            "CouchSceneRecall",
            _format_number(
                prompt_category_test.loc[
                    ("scene_template", "couch"), "recall"
                ]
            ),
        ),
        _macro(
            "CupCanonicalRecall",
            _format_number(
                prompt_category_test.loc[("canonical", "cup"), "recall"]
            ),
        ),
        _macro(
            "CupSynonymRecall",
            _format_number(
                prompt_category_test.loc[("synonym", "cup"), "recall"]
            ),
        ),
        _macro("PrimaryStrategy", _escape_latex(primary)),
        _macro(
            "PrimaryValidationFOne",
            _format_number(strategy_validation.loc[primary, "f1"]),
        ),
        _macro(
            "PrimaryPrecision", _format_number(primary_test["precision"])
        ),
        _macro("PrimaryRecall", _format_number(primary_test["recall"])),
        _macro("PrimaryFOne", _format_number(primary_test["f1"])),
        _macro("PrimaryAP", _format_number(primary_ap["ap"])),
        _macro("PrimaryAPFifty", _format_number(primary_ap["ap50"])),
        _macro(
            "PrimaryDelta",
            _format_signed_number(primary_delta["estimate"]),
        ),
        _macro(
            "PrimaryDeltaLow",
            _format_signed_number(primary_delta["ci_low"]),
        ),
        _macro(
            "PrimaryDeltaHigh",
            _format_signed_number(primary_delta["ci_high"]),
        ),
        _macro(
            "NaiveValidationFOne",
            _format_number(
                strategy_validation.loc["naive_three_prompt_nms", "f1"]
            ),
        ),
        _macro(
            "NaivePrecision",
            _format_number(
                strategy_test.loc["naive_three_prompt_nms", "precision"]
            ),
        ),
        _macro(
            "NaiveRecall",
            _format_number(
                strategy_test.loc["naive_three_prompt_nms", "recall"]
            ),
        ),
        _macro(
            "NaiveFOne",
            _format_number(
                strategy_test.loc["naive_three_prompt_nms", "f1"]
            ),
        ),
        _macro(
            "NaiveAP",
            _format_number(
                ap_test.loc["strategy/naive_three_prompt_nms", "ap"]
            ),
        ),
        _macro(
            "NaiveAPFifty",
            _format_number(
                ap_test.loc["strategy/naive_three_prompt_nms", "ap50"]
            ),
        ),
        _macro(
            "NaiveDelta",
            _format_signed_number(naive_delta["estimate"]),
        ),
        _macro(
            "NaiveDeltaLow",
            _format_signed_number(naive_delta["ci_low"]),
        ),
        _macro(
            "NaiveDeltaHigh",
            _format_signed_number(naive_delta["ci_high"]),
        ),
        _macro(
            "FixedValidationFOne",
            _format_number(
                strategy_validation.loc["category_best_fixed", "f1"]
            ),
        ),
        _macro(
            "FixedPrecision",
            _format_number(
                strategy_test.loc["category_best_fixed", "precision"]
            ),
        ),
        _macro(
            "FixedRecall",
            _format_number(
                strategy_test.loc["category_best_fixed", "recall"]
            ),
        ),
        _macro(
            "FixedFOne",
            _format_number(strategy_test.loc["category_best_fixed", "f1"]),
        ),
        _macro(
            "FixedAP",
            _format_number(
                ap_test.loc["strategy/category_best_fixed", "ap"]
            ),
        ),
        _macro(
            "FixedAPFifty",
            _format_number(
                ap_test.loc["strategy/category_best_fixed", "ap50"]
            ),
        ),
        _macro(
            "FixedDelta",
            _format_signed_number(fixed_delta["estimate"]),
        ),
        _macro(
            "FixedDeltaLow",
            _format_signed_number(fixed_delta["ci_low"]),
        ),
        _macro(
            "FixedDeltaHigh",
            _format_signed_number(fixed_delta["ci_high"]),
        ),
        _macro(
            "CalibratedValidationFOne",
            _format_number(
                strategy_validation.loc["category_best_calibrated", "f1"]
            ),
        ),
        _macro(
            "CalibratedPrecision",
            _format_number(
                strategy_test.loc["category_best_calibrated", "precision"]
            ),
        ),
        _macro(
            "CalibratedRecall",
            _format_number(
                strategy_test.loc["category_best_calibrated", "recall"]
            ),
        ),
        _macro(
            "CalibratedFOne",
            _format_number(
                strategy_test.loc["category_best_calibrated", "f1"]
            ),
        ),
        _macro(
            "CalibratedAP",
            _format_number(
                ap_test.loc["strategy/category_best_calibrated", "ap"]
            ),
        ),
        _macro(
            "CalibratedAPFifty",
            _format_number(
                ap_test.loc["strategy/category_best_calibrated", "ap50"]
            ),
        ),
        _macro(
            "FusionValidationFOne",
            _format_number(
                strategy_validation.loc[
                    "reliability_weighted_fusion", "f1"
                ]
            ),
        ),
        _macro(
            "FusionPrecision",
            _format_number(
                strategy_test.loc[
                    "reliability_weighted_fusion", "precision"
                ]
            ),
        ),
        _macro(
            "FusionRecall",
            _format_number(
                strategy_test.loc["reliability_weighted_fusion", "recall"]
            ),
        ),
        _macro(
            "FusionFOne",
            _format_number(
                strategy_test.loc["reliability_weighted_fusion", "f1"]
            ),
        ),
        _macro(
            "FusionAP",
            _format_number(
                ap_test.loc["strategy/reliability_weighted_fusion", "ap"]
            ),
        ),
        _macro(
            "FusionAPFifty",
            _format_number(
                ap_test.loc[
                    "strategy/reliability_weighted_fusion", "ap50"
                ]
            ),
        ),
        _macro("GPUName", _escape_latex(environment["gpu"])),
        _macro("DriverVersion", _escape_latex(environment["nvidia_driver"])),
        _macro("PythonVersion", _escape_latex(environment["python"])),
        _macro("TorchVersion", _escape_latex(environment["torch"])),
        _macro(
            "UltralyticsVersion", _escape_latex(environment["ultralytics"])
        ),
    ]

    fixed_selection = prompt_selected["fixed_category_selection"]
    for category, macro_prefix in (
        ("car", "Car"),
        ("couch", "Couch"),
        ("airplane", "Airplane"),
        ("cup", "Cup"),
    ):
        lines.extend(
            [
                _macro(
                    f"Fixed{macro_prefix}Variant",
                    _escape_latex(fixed_selection[category]["variant"]),
                ),
                _macro(
                    f"Subset{macro_prefix}Variants",
                    _escape_latex(
                        " + ".join(subset_selection[category]["variants"])
                    ),
                ),
                _macro(
                    f"Subset{macro_prefix}Threshold",
                    _format_number(
                        subset_selection[category]["confidence_threshold"], 2
                    ),
                ),
            ]
        )

    output = report_dir / "generated_metrics.tex"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def create_latex_report(config: dict[str, Any]) -> Path:
    project_root = Path(config["_project_root"])
    results_dir = resolve_path(config, config["paths"]["results_dir"])
    tex_path = project_root / "report" / f"{REPORT_BASENAME}.tex"
    if not tex_path.exists():
        raise FileNotFoundError(f"Missing LaTeX source: {tex_path}")
    create_latex_metrics(config)

    build_dir = results_dir / "latex_build"
    build_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = results_dir / "latex_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["TEXMFVAR"] = str(cache_dir / "texmf-var")
    environment["TEXMFCACHE"] = str(cache_dir / "texmf-cache")
    environment["XDG_CACHE_HOME"] = str(cache_dir / "xdg")
    subprocess.run(
        [
            "latexmk",
            "-lualatex",
            "-g",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-file-line-error",
            f"-outdir={build_dir}",
            str(tex_path),
        ],
        cwd=project_root,
        env=environment,
        check=True,
    )
    compiled_pdf = build_dir / f"{REPORT_BASENAME}.pdf"
    output_pdf = results_dir / f"{REPORT_BASENAME}.pdf"
    shutil.copy2(compiled_pdf, output_pdf)
    return output_pdf
