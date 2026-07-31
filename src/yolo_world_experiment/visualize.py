from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "visual-media-matplotlib"),
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont

from .coco import ground_truth_by_image, load_manifest
from .config import resolve_path
from .model import apply_corruption, apply_gamma_correction
from .same_model_improvement import (
    apply_blur_aware_preprocessing,
    laplacian_variance,
)


PALETTE = {
    "precision": "#275D8C",
    "recall": "#D18F00",
    "f1": "#7A4E9D",
    "ground_truth": "#1B6B5A",
    "prediction": "#B24C35",
}


def _style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#3D4650",
            "axes.labelcolor": "#242A30",
            "text.color": "#242A30",
            "xtick.color": "#3D4650",
            "ytick.color": "#3D4650",
        }
    )


def _save_grouped_metrics(
    frame: pd.DataFrame,
    conditions: list[str],
    labels: list[str],
    title: str,
    subtitle: str,
    output: Path,
) -> None:
    _style()
    selected = frame.set_index("condition").loc[conditions].reset_index()
    metrics = ["precision", "recall", "f1"]
    x = np.arange(len(conditions))
    width = 0.23
    fig, ax = plt.subplots(figsize=(11, 6.5), constrained_layout=True)
    for offset, metric in enumerate(metrics):
        values = selected[metric].to_numpy(dtype=float)
        bars = ax.bar(
            x + (offset - 1) * width,
            values,
            width,
            label=metric.capitalize(),
            color=PALETTE[metric],
            edgecolor="#242A30",
            linewidth=0.7,
        )
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_xticks(x, labels)
    ax.set_title(title, loc="left", weight="bold", pad=20)
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=11, color="#59636E")
    ax.legend(frameon=False, ncol=3, loc="upper right")
    ax.grid(axis="x", visible=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def create_metric_charts(config: dict[str, Any]) -> list[Path]:
    results_dir = resolve_path(config, config["paths"]["results_dir"])
    figures_dir = results_dir / "figures"
    manifest = load_manifest(config)
    validation_count = sum(
        item["split"] == "validation" for item in manifest["images"]
    )
    test_count = sum(item["split"] == "test" for item in manifest["images"])
    frame = pd.read_csv(results_dir / "summary_metrics.csv")
    test = frame[frame["scope"] == "test"]
    outputs = []

    corruption_conditions = [
        "original",
        "brightness_0.50",
        "brightness_0.25",
        "blur_sigma_2",
        "blur_sigma_4",
    ]
    output = figures_dir / "failure_image_corruptions.png"
    _save_grouped_metrics(
        test,
        corruption_conditions,
        ["Original", "Brightness 0.50", "Brightness 0.25", "Blur σ=2", "Blur σ=4"],
        "Detection metrics by image degradation",
        f"COCO 2017 val test split, {test_count} images; confidence ≥ 0.25, IoU ≥ 0.50",
        output,
    )
    outputs.append(output)

    output = figures_dir / "failure_prompt_wording.png"
    _save_grouped_metrics(
        test,
        ["prompt_canonical", "prompt_synonym", "prompt_hypernym"],
        ["Canonical", "Synonym", "Descriptive/hypernym"],
        "Detection metrics by prompt wording",
        f"Same {test_count} images and boxes; only the class prompts change",
        output,
    )
    outputs.append(output)

    output = figures_dir / "improvement_prompt_ensemble.png"
    _save_grouped_metrics(
        test,
        ["baseline_tuned", "prompt_ensemble"],
        ["Single prompt", "Prompt ensemble"],
        "Single-prompt and ensemble detection metrics",
        f"Same tuned confidence threshold on the held-out {test_count}-image test split",
        output,
    )
    outputs.append(output)

    output = figures_dir / "improvement_gamma_correction.png"
    _save_grouped_metrics(
        test,
        ["lowlight_baseline", "gamma_correction"],
        ["Brightness 0.25", "Gamma-corrected"],
        "Low-light preprocessing metrics",
        f"Gamma selected on {validation_count} validation images; evaluated on the held-out {test_count}-image test split",
        output,
    )
    outputs.append(output)
    if (results_dir / "prompt_strategy_metrics.csv").exists():
        outputs.extend(_create_prompt_study_charts(config))
    return outputs


def _create_prompt_study_charts(config: dict[str, Any]) -> list[Path]:
    results_dir = resolve_path(config, config["paths"]["results_dir"])
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(config)
    validation_count = sum(
        item["split"] == "validation" for item in manifest["images"]
    )
    test_count = sum(item["split"] == "test" for item in manifest["images"])
    strategy_metrics = pd.read_csv(
        results_dir / "prompt_strategy_metrics.csv"
    )
    ap_metrics = pd.read_csv(results_dir / "detection_ap_metrics.csv")
    variant_category = pd.read_csv(
        results_dir / "prompt_variant_by_category.csv"
    )
    pr_curves = pd.read_csv(results_dir / "detection_pr_curves.csv")
    intervals = pd.read_csv(results_dir / "bootstrap_intervals.csv")
    strategies = [
        "canonical_baseline",
        "naive_three_prompt_nms",
        "category_best_fixed",
        "category_best_calibrated",
        "category_subset_nms",
        "reliability_weighted_fusion",
    ]
    labels = [
        "Canonical",
        "Naive\nensemble",
        "Category\nword",
        "Word +\nthreshold",
        "Subset\nNMS",
        "Weighted\nfusion",
    ]
    outputs: list[Path] = []

    validation_f1 = (
        strategy_metrics[strategy_metrics["scope"] == "validation"]
        .set_index("strategy")
        .loc[strategies, "f1"]
        .to_numpy(dtype=float)
    )
    test_f1 = (
        strategy_metrics[strategy_metrics["scope"] == "test"]
        .set_index("strategy")
        .loc[strategies, "f1"]
        .to_numpy(dtype=float)
    )
    ap_indexed = ap_metrics[
        (ap_metrics["scope"] == "test")
        & (ap_metrics["family"] == "prompt_strategy")
    ].set_index("condition")
    test_ap50 = np.asarray(
        [
            ap_indexed.loc[f"strategy/{strategy}", "ap50"]
            for strategy in strategies
        ],
        dtype=float,
    )
    _style()
    x = np.arange(len(strategies))
    width = 0.24
    fig, ax = plt.subplots(figsize=(12.5, 6.8), constrained_layout=True)
    for offset, (name, values, color) in enumerate(
        [
            ("Validation F1", validation_f1, "#99A6B5"),
            ("Test F1", test_f1, PALETTE["f1"]),
            ("Test AP50", test_ap50, "#1B6B5A"),
        ]
    ):
        bars = ax.bar(
            x + (offset - 1) * width,
            values,
            width,
            label=name,
            color=color,
            edgecolor="#242A30",
            linewidth=0.6,
        )
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Score")
    ax.set_xticks(x, labels)
    ax.set_title(
        "Prompt strategies: validation selection and held-out performance",
        loc="left",
        weight="bold",
        pad=20,
    )
    ax.text(
        0,
        1.02,
        f"Parameters selected on {validation_count} validation images; test uses {test_count} fixed images",
        transform=ax.transAxes,
        fontsize=10.5,
        color="#59636E",
    )
    ax.legend(frameon=False, ncol=3, loc="lower left")
    ax.grid(axis="x", visible=False)
    output = figures_dir / "prompt_strategy_comparison.png"
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    outputs.append(output)

    variants = [
        "canonical",
        "synonym",
        "hypernym",
        "photo_template",
        "scene_template",
        "plural",
    ]
    heatmap_data = (
        variant_category[variant_category["scope"] == "test"]
        .pivot(index="category", columns="variant", values="recall")
        .loc[["car", "couch", "airplane", "cup"], variants]
    )
    _style()
    fig, ax = plt.subplots(figsize=(11.5, 5.6), constrained_layout=True)
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".2f",
        vmin=0,
        vmax=1,
        cmap=sns.light_palette("#275D8C", as_cmap=True),
        linewidths=0.8,
        linecolor="white",
        cbar_kws={"label": "Recall"},
        ax=ax,
    )
    ax.set_xlabel("Prompt variant")
    ax.set_ylabel("COCO category")
    ax.set_xticklabels(
        [
            "Canonical",
            "Synonym",
            "Hypernym",
            "Photo template",
            "Scene template",
            "Plural",
        ],
        rotation=20,
        ha="right",
    )
    ax.set_title(
        "Prompt sensitivity differs by category",
        loc="left",
        weight="bold",
        pad=18,
    )
    output = figures_dir / "prompt_variant_recall_heatmap.png"
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    outputs.append(output)

    selected_curves = {
        "strategy/canonical_baseline": "Canonical",
        "strategy/naive_three_prompt_nms": "Naive ensemble",
        "strategy/category_best_fixed": "Category word",
        "strategy/category_subset_nms": "Validation-selected subset",
    }
    _style()
    fig, ax = plt.subplots(figsize=(8.4, 6.7), constrained_layout=True)
    colors = ["#275D8C", "#B24C35", "#1B6B5A", "#7A4E9D"]
    for (condition, label), color in zip(selected_curves.items(), colors):
        curve = pr_curves[
            (pr_curves["scope"] == "test")
            & (pr_curves["condition"] == condition)
        ]
        ap50 = float(ap_indexed.loc[condition, "ap50"])
        ax.step(
            curve["recall"],
            curve["precision"],
            where="post",
            linewidth=2.2,
            color=color,
            label=f"{label} (AP50={ap50:.3f})",
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Interpolated precision")
    ax.set_title(
        "Precision–recall curves at IoU = 0.50",
        loc="left",
        weight="bold",
        pad=16,
    )
    ax.legend(frameon=False, fontsize=10, loc="lower left")
    output = figures_dir / "prompt_pr_curves.png"
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    outputs.append(output)

    f1_intervals = intervals[
        (intervals["scope"] == "test")
        & (intervals["metric"] == "f1")
        & intervals["condition"].isin(
            [f"strategy/{strategy}" for strategy in strategies]
        )
    ].set_index("condition")
    estimates = np.asarray(
        [
            f1_intervals.loc[f"strategy/{strategy}", "estimate"]
            for strategy in strategies
        ],
        dtype=float,
    )
    lower = np.asarray(
        [
            f1_intervals.loc[f"strategy/{strategy}", "ci_low"]
            for strategy in strategies
        ],
        dtype=float,
    )
    upper = np.asarray(
        [
            f1_intervals.loc[f"strategy/{strategy}", "ci_high"]
            for strategy in strategies
        ],
        dtype=float,
    )
    _style()
    fig, ax = plt.subplots(figsize=(10.8, 6.2), constrained_layout=True)
    ax.errorbar(
        np.arange(len(strategies)),
        estimates,
        yerr=np.vstack([estimates - lower, upper - estimates]),
        fmt="o",
        markersize=8,
        capsize=5,
        linewidth=2,
        color="#275D8C",
        ecolor="#7A8795",
    )
    ax.axhline(
        estimates[0],
        color="#B24C35",
        linestyle="--",
        linewidth=1.4,
        label=f"Canonical F1 = {estimates[0]:.3f}",
    )
    ax.set_ylim(0.35, 1.0)
    ax.set_ylabel("F1 with image-bootstrap 95% interval")
    ax.set_xticks(np.arange(len(strategies)), labels)
    ax.set_title(
        f"Bootstrap uncertainty on the {test_count}-image test set",
        loc="left",
        weight="bold",
        pad=16,
    )
    ax.legend(frameon=False, loc="lower left")
    output = figures_dir / "prompt_f1_bootstrap_intervals.png"
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    outputs.append(output)
    return outputs


def _load_predictions(path: Path) -> dict[str, dict[int, list[dict[str, Any]]]]:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return {
        condition: {int(image_id): preds for image_id, preds in by_image.items()}
        for condition, by_image in raw.items()
    }


def _draw_boxes(
    image: Image.Image,
    ground_truth: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    confidence_threshold: float,
    title: str,
) -> Image.Image:
    image = image.convert("RGB").copy()
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=16)
    for gt in ground_truth:
        box = tuple(gt["bbox_xyxy"])
        draw.rectangle(box, outline=PALETTE["ground_truth"], width=4)
        draw.text(
            (box[0] + 3, box[1] + 3),
            f"GT {gt['category']}",
            fill="white",
            font=font,
            stroke_width=3,
            stroke_fill=PALETTE["ground_truth"],
        )
    for pred in predictions:
        if float(pred["confidence"]) < confidence_threshold:
            continue
        box = tuple(pred["bbox_xyxy"])
        draw.rectangle(box, outline=PALETTE["prediction"], width=3)
        draw.text(
            (box[0] + 3, max(0, box[1] - 20)),
            f"{pred['category']} {pred['confidence']:.2f}",
            fill="white",
            font=font,
            stroke_width=3,
            stroke_fill=PALETTE["prediction"],
        )
    title_height = 34
    canvas = Image.new("RGB", (image.width, image.height + title_height), "white")
    canvas.paste(image, (0, title_height))
    ImageDraw.Draw(canvas).text((10, 8), title, fill="#242A30", font=font)
    return canvas


def create_representative_images(config: dict[str, Any]) -> list[Path]:
    manifest = load_manifest(config)
    gt = ground_truth_by_image(manifest)
    images_dir = resolve_path(config, config["paths"]["coco_images"])
    results_dir = resolve_path(config, config["paths"]["results_dir"])
    output_dir = results_dir / "examples"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = _load_predictions(results_dir / "raw_predictions.json")
    ensemble = _load_predictions(results_dir / "ensemble_predictions.json")[
        "prompt_ensemble"
    ]
    details = pd.read_csv(results_dir / "per_image_metrics.csv")
    with (results_dir / "selected_parameters.json").open("r", encoding="utf-8") as handle:
        selected = json.load(handle)
    tuned_conf = float(selected["ensemble"]["confidence_threshold"])
    selected_gamma = float(selected["gamma_correction"]["gamma"])

    test_entries = {
        int(item["image_id"]): item
        for item in manifest["images"]
        if item["split"] == "test"
    }
    test_ids = list(test_entries)
    baseline = details[
        (details["condition"] == "original") & (details["scope"] == "test")
    ].set_index("image_id")

    worst_blur = details[
        (details["condition"] == "blur_sigma_4") & (details["scope"] == "test")
    ].set_index("image_id")
    image_id_failure = int(
        (baseline["f1"] - worst_blur["f1"]).sort_values(ascending=False).index[0]
    )

    base_tuned = details[
        (details["condition"] == "baseline_tuned") & (details["scope"] == "test")
    ].set_index("image_id")
    ensemble_details = details[
        (details["condition"] == "prompt_ensemble") & (details["scope"] == "test")
    ].set_index("image_id")
    lowlight = details[
        (details["condition"] == "lowlight_baseline") & (details["scope"] == "test")
    ].set_index("image_id")
    gamma_details = details[
        (details["condition"] == "gamma_correction") & (details["scope"] == "test")
    ].set_index("image_id")
    image_id_improvement = int(
        (gamma_details["f1"] - lowlight["f1"])
        .sort_values(ascending=False)
        .index[0]
    )

    image_id_success = int(baseline.sort_values("f1", ascending=False).index[0])
    selections = [
        (
            image_id_success,
            "original",
            0.25,
            "Baseline success",
            "baseline_success.png",
        ),
        (
            image_id_failure,
            "blur_sigma_4",
            0.25,
            "Failure under Gaussian blur (σ=4)",
            "failure_under_gaussian_blur_sigma4.png",
        ),
        (
            image_id_improvement,
            "lowlight_baseline",
            0.25,
            "Low-light baseline",
            "low_light_baseline.png",
        ),
        (
            image_id_improvement,
            "gamma_correction",
            0.25,
            "Gamma correction",
            "gamma_correction.png",
        ),
    ]
    outputs = []
    example_metadata: dict[str, Any] = {}
    for image_id, condition, threshold, title, output_name in selections:
        entry = test_entries[image_id]
        original_image = Image.open(images_dir / entry["file_name"]).convert("RGB")
        if condition == "blur_sigma_4":
            display_image = apply_corruption(original_image, "gaussian_blur", 4.0)
        elif condition == "lowlight_baseline":
            display_image = apply_corruption(original_image, "brightness", 0.25)
        elif condition == "gamma_correction":
            dark = apply_corruption(original_image, "brightness", 0.25)
            display_image = apply_gamma_correction(dark, selected_gamma)
        else:
            display_image = original_image
        raw_condition = (
            f"lowlight_gamma_{selected_gamma:.2f}"
            if condition == "gamma_correction"
            else "brightness_0.25"
            if condition == "lowlight_baseline"
            else condition
        )
        predictions = raw[raw_condition].get(image_id, [])
        canvas = _draw_boxes(
            display_image,
            gt[image_id],
            predictions,
            threshold,
            title,
        )
        output = output_dir / output_name
        canvas.save(output)
        outputs.append(output)
        example_metadata[condition] = {
            "image_id": image_id,
            "file_name": entry["file_name"],
            "output": str(output.relative_to(results_dir)),
        }

    prompt_predictions_path = (
        results_dir / "prompt_strategy_predictions.json"
    )
    prompt_details_path = results_dir / "prompt_strategy_per_image.csv"
    if prompt_predictions_path.exists() and prompt_details_path.exists():
        prompt_predictions = _load_predictions(prompt_predictions_path)
        prompt_details = pd.read_csv(prompt_details_path)
        baseline_prompt = prompt_details[
            (prompt_details["scope"] == "test")
            & (prompt_details["strategy"] == "canonical_baseline")
        ].set_index("image_id")
        category_prompt = prompt_details[
            (prompt_details["scope"] == "test")
            & (prompt_details["strategy"] == "category_best_fixed")
        ].set_index("image_id")
        prompt_image_id = int(
            (category_prompt["f1"] - baseline_prompt["f1"])
            .sort_values(ascending=False)
            .index[0]
        )
        entry = test_entries[prompt_image_id]
        prompt_image = Image.open(
            images_dir / entry["file_name"]
        ).convert("RGB")
        for strategy, title, output_name in (
            ("canonical_baseline", "Prompt baseline", "prompt_baseline.png"),
            (
                "category_best_fixed",
                "Category-specific prompt",
                "category_specific_prompt.png",
            ),
        ):
            canvas = _draw_boxes(
                prompt_image,
                gt[prompt_image_id],
                prompt_predictions[strategy].get(prompt_image_id, []),
                0.0,
                title,
            )
            output = output_dir / output_name
            canvas.save(output)
            outputs.append(output)
            example_metadata[strategy] = {
                "image_id": prompt_image_id,
                "file_name": entry["file_name"],
                "output": str(output.relative_to(results_dir)),
            }
    (output_dir / "representative_examples.json").write_text(
        json.dumps(example_metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return outputs


def create_model_scale_outputs(config: dict[str, Any]) -> list[Path]:
    """Create the final-holdout comparison chart and a paired box example."""
    results_dir = resolve_path(config, config["paths"]["results_dir"])
    improvement_dir = resolve_path(
        config, config["paths"]["improvement_results_dir"]
    )
    figures_dir = results_dir / "figures"
    examples_dir = results_dir / "examples"
    figures_dir.mkdir(parents=True, exist_ok=True)
    examples_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(
        improvement_dir / "model_final_holdout_summary.csv"
    ).set_index("condition")
    conditions = [
        "small_canonical_baseline",
        "small_threshold_tuned_ablation",
        "medium_model_improvement",
    ]
    labels = ["Small\nfixed 0.25", "Small\nclass-tuned", "Medium\nclass-tuned"]
    metrics = ["precision", "recall", "f1", "ap", "ap50", "ap75"]
    metric_labels = ["Precision", "Recall", "F1", "mAP", "AP50", "AP75"]
    colors = ["#275D8C", "#B8C0C8", "#C9752A"]
    hatches = ["", "//", ".."]
    _style()
    fig, ax = plt.subplots(figsize=(11.5, 5.8), constrained_layout=True)
    x = np.arange(len(metrics))
    width = 0.24
    for index, (condition, label, color, hatch) in enumerate(
        zip(conditions, labels, colors, hatches)
    ):
        values = summary.loc[condition, metrics].to_numpy(dtype=float)
        bars = ax.bar(
            x + (index - 1) * width,
            values,
            width,
            label=label.replace("\n", " "),
            color=color,
            hatch=hatch,
            edgecolor="#242A30",
            linewidth=0.8,
        )
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
    ax.set_ylim(0, 0.86)
    ax.set_ylabel("Score")
    ax.set_xticks(x, metric_labels)
    ax.set_title(
        "YOLO-World model-scale comparison on the evaluation set",
        loc="left",
        weight="bold",
        pad=18,
    )
    ax.text(
        0,
        1.02,
        "100 untouched COCO val images, 169 target boxes; IoU ≥ 0.50",
        transform=ax.transAxes,
        fontsize=10.5,
        color="#59636E",
    )
    ax.legend(frameon=False, ncol=3, loc="upper right")
    ax.grid(axis="x", visible=False)
    chart_path = figures_dir / "model_scale_final_holdout.png"
    fig.savefig(
        chart_path, dpi=200, bbox_inches="tight", facecolor="white"
    )
    plt.close(fig)

    manifest_path = resolve_path(
        config, config["paths"]["final_holdout_manifest"]
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ground_truth = ground_truth_by_image(manifest)
    entries = {
        int(item["image_id"]): item for item in manifest["images"]
    }
    details = pd.read_csv(
        improvement_dir / "model_final_holdout_per_image.csv"
    )
    baseline = details[
        details["condition"] == "small_canonical_baseline"
    ].set_index("image_id")
    medium = details[
        details["condition"] == "medium_model_improvement"
    ].set_index("image_id")
    ranking = pd.DataFrame(
        {
            "f1_gain": medium["f1"] - baseline["f1"],
            "fp_reduction": baseline["fp"] - medium["fp"],
            "tp_gain": medium["tp"] - baseline["tp"],
        }
    ).sort_values(
        ["f1_gain", "fp_reduction", "tp_gain"],
        ascending=False,
    )
    image_id = int(ranking.index[0])
    predictions = _load_predictions(
        improvement_dir / "model_final_holdout_predictions.json"
    )
    image = Image.open(
        resolve_path(config, config["paths"]["coco_images"])
        / entries[image_id]["file_name"]
    ).convert("RGB")
    outputs = [chart_path]
    example_paths = []
    for condition, title, filename in (
        (
            "small_canonical_baseline",
            "Small model baseline",
            "model_scale_baseline.png",
        ),
        (
            "medium_model_improvement",
            "Medium model improvement",
            "model_scale_improvement.png",
        ),
    ):
        canvas = _draw_boxes(
            image,
            ground_truth[image_id],
            predictions[condition][image_id],
            0.0,
            title,
        )
        path = examples_dir / filename
        canvas.save(path)
        outputs.append(path)
        example_paths.append(str(path.relative_to(results_dir)))

    (improvement_dir / "model_scale_example.json").write_text(
        json.dumps(
            {
                "image_id": image_id,
                "file_name": entries[image_id]["file_name"],
                "baseline_f1": float(baseline.loc[image_id, "f1"]),
                "medium_f1": float(medium.loc[image_id, "f1"]),
                "baseline_fp": int(baseline.loc[image_id, "fp"]),
                "medium_fp": int(medium.loc[image_id, "fp"]),
                "outputs": example_paths,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (improvement_dir / "CHART_MAP.md").write_text(
        "\n".join(
            [
                "# Chart map",
                "",
                "- Section: final model-scale improvement",
                "- Question: does the medium model outperform the small baseline on untouched data?",
                "- Family/type: comparison, grouped bar",
                "- Fields: condition, Precision, Recall, F1, mAP, AP50, AP75",
                "- Claim: medium improves F1 and COCO AP while reducing false positives",
                "- Palette: blue baseline, neutral ablation, orange candidate; hatch redundancy",
                f"- Artifact: {chart_path.relative_to(results_dir)}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return outputs


def create_same_model_outputs(config: dict[str, Any]) -> list[Path]:
    """Visualize the frozen Small-model blur-preprocessing comparison."""
    results_dir = resolve_path(config, config["paths"]["results_dir"])
    same_model_dir = resolve_path(
        config, config["paths"]["same_model_results_dir"]
    )
    figures_dir = results_dir / "figures"
    examples_dir = results_dir / "examples"
    figures_dir.mkdir(parents=True, exist_ok=True)
    examples_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(
        same_model_dir / "same_model_final_summary.csv"
    ).set_index(["method", "condition"])
    conditions = ["clean", "blur_sigma_2", "blur_sigma_4"]
    labels = ["Clean", "Blur σ=2", "Blur σ=4"]
    methods = ["baseline", "blur_aware_wiener"]
    method_labels = ["Baseline", "Blur-aware Wiener"]
    colors = ["#275D8C", "#C9752A"]
    hatches = ["", ".."]
    _style()
    fig, axes = plt.subplots(
        1, 2, figsize=(11.2, 5.2), constrained_layout=True
    )
    x = np.arange(len(conditions))
    width = 0.34
    for axis, metric, title in zip(
        axes,
        ("f1", "ap"),
        ("F1 at confidence 0.25", "COCO mAP"),
    ):
        for index, (method, label, color, hatch) in enumerate(
            zip(methods, method_labels, colors, hatches)
        ):
            values = np.asarray(
                [
                    summary.loc[(method, condition), metric]
                    for condition in conditions
                ],
                dtype=float,
            )
            bars = axis.bar(
                x + (index - 0.5) * width,
                values,
                width,
                label=label,
                color=color,
                hatch=hatch,
                edgecolor="#242A30",
                linewidth=0.8,
            )
            axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
        axis.set_ylim(0, 0.75)
        axis.set_xticks(x, labels)
        axis.set_title(title, weight="bold")
        axis.grid(axis="x", visible=False)
    axes[0].set_ylabel("Score")
    axes[1].legend(frameon=False, loc="upper right")
    fig.suptitle(
        "Blur preprocessing on the evaluation set",
        x=0.01,
        ha="left",
        weight="bold",
        fontsize=17,
    )
    fig.text(
        0.01,
        0.935,
        "100 COCO val images; detector: yolov8s-worldv2.pt; IoU ≥ 0.50",
        color="#59636E",
        fontsize=10.5,
    )
    chart_path = figures_dir / "same_model_blur_final_holdout.png"
    fig.savefig(
        chart_path, dpi=200, bbox_inches="tight", facecolor="white"
    )
    plt.close(fig)

    manifest_path = resolve_path(
        config, config["paths"]["same_model_holdout_manifest"]
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ground_truth = ground_truth_by_image(manifest)
    entries = {
        int(item["image_id"]): item for item in manifest["images"]
    }
    details = pd.read_csv(
        same_model_dir / "same_model_final_per_image.csv"
    )
    baseline = details[
        (details["method"] == "baseline")
        & (details["condition"] == "blur_sigma_4")
    ].set_index("image_id")
    improved = details[
        (details["method"] == "blur_aware_wiener")
        & (details["condition"] == "blur_sigma_4")
    ].set_index("image_id")
    ranking = pd.DataFrame(
        {
            "f1_gain": improved["f1"] - baseline["f1"],
            "tp_gain": improved["tp"] - baseline["tp"],
            "fp_reduction": baseline["fp"] - improved["fp"],
        }
    ).sort_values(
        ["f1_gain", "tp_gain", "fp_reduction"],
        ascending=False,
    )
    image_id = int(ranking.index[0])
    entry = entries[image_id]
    original = Image.open(
        resolve_path(config, config["paths"]["coco_images"])
        / entry["file_name"]
    ).convert("RGB")
    blurred = apply_corruption(original, "gaussian_blur", 4.0)
    selected = json.loads(
        (
            same_model_dir / "selected_same_model_method.json"
        ).read_text(encoding="utf-8")
    )
    score = laplacian_variance(blurred)
    selected_candidate = (
        selected["severe_candidate"]
        if score <= float(selected["blur_detector"]["severe_threshold"])
        else selected["mild_candidate"]
    )
    corrected, correction_metadata = apply_blur_aware_preprocessing(
        blurred,
        selected_candidate,
        selected["blur_detector"],
    )
    predictions = _load_predictions(
        same_model_dir / "same_model_final_predictions.json"
    )
    outputs = [chart_path]
    example_paths = []
    for image, condition, title, filename in (
        (
            blurred,
            "baseline_blur_sigma_4",
            "Gaussian blur sigma=4",
            "same_model_blur_baseline.png",
        ),
        (
            corrected,
            "blur_aware_wiener_blur_sigma_4",
            "Wiener preprocessing",
            "same_model_blur_improvement.png",
        ),
    ):
        canvas = _draw_boxes(
            image,
            ground_truth[image_id],
            predictions[condition][image_id],
            0.25,
            title,
        )
        path = examples_dir / filename
        canvas.save(path)
        outputs.append(path)
        example_paths.append(str(path.relative_to(results_dir)))

    metadata = {
        "image_id": image_id,
        "file_name": entry["file_name"],
        "condition": "blur_sigma_4",
        "baseline_f1": float(baseline.loc[image_id, "f1"]),
        "improved_f1": float(improved.loc[image_id, "f1"]),
        "baseline_tp": int(baseline.loc[image_id, "tp"]),
        "improved_tp": int(improved.loc[image_id, "tp"]),
        "correction": correction_metadata,
        "outputs": example_paths,
    }
    (same_model_dir / "same_model_example.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    (same_model_dir / "CHART_MAP.md").write_text(
        "\n".join(
            [
                "# Chart map",
                "",
                "- Section: same-model blur-aware improvement",
                "- Question: does classical deblurring improve blur robustness without replacing YOLO-World?",
                "- Family/type: comparison, paired grouped bars",
                "- Fields: input condition, method, F1, COCO mAP",
                "- Claim: the fixed Small model improves strongly at blur sigma 4, with negligible clean change",
                "- Palette: blue baseline, orange candidate; hatch redundancy",
                f"- Artifact: {chart_path.relative_to(results_dir)}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return outputs


def create_prompt_prototype_outputs(
    config: dict[str, Any],
) -> list[Path]:
    """Visualize the frozen canonical-anchored prompt comparison."""
    results_dir = resolve_path(config, config["paths"]["results_dir"])
    prompt_dir = resolve_path(
        config, config["paths"]["prompt_prototype_results_dir"]
    )
    figures_dir = results_dir / "figures"
    examples_dir = results_dir / "examples"
    figures_dir.mkdir(parents=True, exist_ok=True)
    examples_dir.mkdir(parents=True, exist_ok=True)

    verdict = json.loads(
        (prompt_dir / "prompt_prototype_final_verdict.json").read_text(
            encoding="utf-8"
        )
    )
    anchor_weight = float(verdict["anchor_weight"])
    candidate_method = f"canonical_anchor_{anchor_weight:.2f}"
    summary = pd.read_csv(
        prompt_dir / "prompt_prototype_final_summary.csv"
    ).set_index(["method", "condition"])
    conditions = ["canonical", "synonym", "hypernym"]
    labels = ["Canonical", "Synonym", "Hypernym"]
    methods = ["raw_prompt", candidate_method]
    method_labels = ["Raw prompt", "Canonical-anchored prototype"]
    colors = ["#275D8C", "#C9752A"]
    hatches = ["", ".."]

    _style()
    fig, axes = plt.subplots(
        1, 2, figsize=(11.2, 5.2), constrained_layout=True
    )
    x = np.arange(len(conditions))
    width = 0.34
    for axis, metric, title in zip(
        axes,
        ("f1", "ap"),
        ("F1 at confidence 0.25", "COCO mAP"),
    ):
        for index, (method, label, color, hatch) in enumerate(
            zip(methods, method_labels, colors, hatches)
        ):
            values = np.asarray(
                [
                    summary.loc[(method, condition), metric]
                    for condition in conditions
                ],
                dtype=float,
            )
            bars = axis.bar(
                x + (index - 0.5) * width,
                values,
                width,
                label=label,
                color=color,
                hatch=hatch,
                edgecolor="#242A30",
                linewidth=0.8,
            )
            axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
        axis.set_ylim(0, 0.75)
        axis.set_xticks(x, labels)
        axis.set_title(title, weight="bold")
        axis.grid(axis="x", visible=False)
    axes[0].set_ylabel("Score")
    axes[1].legend(frameon=False, loc="upper right")
    fig.suptitle(
        "Prompt prototype on the evaluation set",
        x=0.01,
        ha="left",
        weight="bold",
        fontsize=17,
    )
    fig.text(
        0.01,
        0.935,
        "100 COCO val images; detector: yolov8s-worldv2.pt; anchor weight: 0.75",
        color="#59636E",
        fontsize=10.5,
    )
    chart_path = figures_dir / "prompt_prototype_final_holdout.png"
    fig.savefig(
        chart_path, dpi=200, bbox_inches="tight", facecolor="white"
    )
    plt.close(fig)

    manifest_path = resolve_path(
        config, config["paths"]["prompt_prototype_holdout_manifest"]
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ground_truth = ground_truth_by_image(manifest)
    entries = {
        int(item["image_id"]): item for item in manifest["images"]
    }
    details = pd.read_csv(
        prompt_dir / "prompt_prototype_final_per_image.csv"
    )
    baseline = details[
        (details["method"] == "raw_prompt")
        & (details["condition"] == "hypernym")
    ].set_index("image_id")
    improved = details[
        (details["method"] == candidate_method)
        & (details["condition"] == "hypernym")
    ].set_index("image_id")
    ranking = pd.DataFrame(
        {
            "f1_gain": improved["f1"] - baseline["f1"],
            "tp_gain": improved["tp"] - baseline["tp"],
            "fp_reduction": baseline["fp"] - improved["fp"],
        }
    ).sort_values(
        ["f1_gain", "tp_gain", "fp_reduction"],
        ascending=False,
    )
    image_id = int(ranking.index[0])
    predictions = _load_predictions(
        prompt_dir / "prompt_prototype_final_predictions.json"
    )
    image = Image.open(
        resolve_path(config, config["paths"]["coco_images"])
        / entries[image_id]["file_name"]
    ).convert("RGB")
    outputs = [chart_path]
    example_paths = []
    for key, title, filename in (
        (
            "baseline__hypernym",
            "Raw hypernym prompt",
            "prompt_prototype_baseline.png",
        ),
        (
            f"anchor_{anchor_weight:.2f}__hypernym",
            "Canonical-anchored text prototype",
            "prompt_prototype_improvement.png",
        ),
    ):
        canvas = _draw_boxes(
            image,
            ground_truth[image_id],
            predictions[key][image_id],
            0.25,
            title,
        )
        path = examples_dir / filename
        canvas.save(path)
        outputs.append(path)
        example_paths.append(str(path.relative_to(results_dir)))

    (prompt_dir / "prompt_prototype_example.json").write_text(
        json.dumps(
            {
                "image_id": image_id,
                "file_name": entries[image_id]["file_name"],
                "condition": "hypernym",
                "baseline_f1": float(baseline.loc[image_id, "f1"]),
                "improved_f1": float(improved.loc[image_id, "f1"]),
                "baseline_tp": int(baseline.loc[image_id, "tp"]),
                "improved_tp": int(improved.loc[image_id, "tp"]),
                "outputs": example_paths,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (prompt_dir / "CHART_MAP.md").write_text(
        "\n".join(
            [
                "# Chart map",
                "",
                "- Section: prompt-wording robustness improvement",
                "- Question: does a canonical-anchored text prototype reduce sensitivity to synonym and hypernym prompts?",
                "- Family/type: comparison, paired grouped bars",
                "- Fields: prompt condition, method, F1, COCO mAP",
                "- Claim: canonical output is unchanged while synonym and hypernym performance improves",
                "- Palette: blue raw prompt, orange anchored prototype; hatch redundancy",
                f"- Artifact: {chart_path.relative_to(results_dir)}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return outputs
