from __future__ import annotations

import base64
import json
import sqlite3
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

from .config import resolve_path


TITLE = "YOLO-WorldのFailure Case Analysisとprompt改善"


def _metric_query(
    conditions: list[str],
    labels: dict[str, str],
) -> str:
    condition_sql = ", ".join(f"'{condition}'" for condition in conditions)
    case_sql = "CASE condition " + " ".join(
        f"WHEN '{condition}' THEN '{labels[condition]}'" for condition in conditions
    ) + " END"
    order_sql = "CASE condition " + " ".join(
        f"WHEN '{condition}' THEN {index}" for index, condition in enumerate(conditions)
    ) + " END"
    branches = []
    for metric, label, metric_order in [
        ("precision", "Precision", 0),
        ("recall", "Recall", 1),
        ("f1", "F1", 2),
    ]:
        branches.append(
            "SELECT "
            f"{case_sql} AS condition, condition AS condition_id, "
            f"'{label}' AS metric, {metric} AS value, tp, fp, fn, "
            f"{order_sql} AS condition_order, {metric_order} AS metric_order "
            "FROM summary_metrics "
            f"WHERE condition IN ({condition_sql})"
        )
    return (
        "SELECT condition, condition_id, metric, value, tp, fp, fn "
        "FROM (" + " UNION ALL ".join(branches) + ") "
        "ORDER BY condition_order, metric_order"
    )


def _query_rows(connection: sqlite3.Connection, sql: str) -> list[dict[str, Any]]:
    connection.row_factory = sqlite3.Row
    return [dict(row) for row in connection.execute(sql).fetchall()]


def _image_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    media_type = "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
    return f"data:{media_type};base64,{encoded}"


def _example_html(items: list[tuple[Path, str]]) -> str:
    figures = []
    for path, caption in items:
        figures.append(
            "<figure style=\"margin:0;min-width:0\">"
            f"<img src=\"{_image_data_url(path)}\" alt=\"{escape(caption)}\" "
            "style=\"display:block;width:100%;height:auto;border-radius:8px\">"
            f"<figcaption style=\"margin-top:6px;color:#5f6368;font-size:12px\">"
            f"{escape(caption)}</figcaption></figure>"
        )
    return (
        "<section style=\"display:grid;grid-template-columns:repeat(2,minmax(0,1fr));"
        "gap:16px\">"
        + "".join(figures)
        + "</section>"
    )


def _full_width_image_html(path: Path, caption: str) -> str:
    return (
        "<figure style=\"margin:0;max-width:900px\">"
        f"<img src=\"{_image_data_url(path)}\" alt=\"{escape(caption)}\" "
        "style=\"display:block;width:100%;height:auto;border-radius:8px\">"
        f"<figcaption style=\"margin-top:6px;color:#5f6368;font-size:12px\">"
        f"{escape(caption)}</figcaption></figure>"
    )


def _prepare_report_assets(results_dir: Path) -> None:
    output_dir = results_dir / "report_assets"
    output_dir.mkdir(parents=True, exist_ok=True)
    conversions = [
        ("examples/baseline_success.png", "baseline_success.jpg", (640, 480), 60),
        (
            "examples/failure_under_gaussian_blur_sigma4.png",
            "failure_blur_sigma4.jpg",
            (640, 480),
            60,
        ),
        ("examples/low_light_baseline.png", "lowlight_baseline.jpg", (640, 480), 60),
        ("examples/gamma_correction.png", "gamma_correction.jpg", (640, 480), 60),
        (
            "figures/failure_image_corruptions.png",
            "failure_image_corruptions.jpg",
            (900, 650),
            75,
        ),
        (
            "figures/failure_prompt_wording.png",
            "failure_prompt_wording.jpg",
            (900, 650),
            75,
        ),
        (
            "figures/improvement_gamma_correction.png",
            "improvement_gamma_correction.jpg",
            (900, 650),
            75,
        ),
        (
            "figures/improvement_prompt_ensemble.png",
            "improvement_prompt_ensemble.jpg",
            (900, 650),
            75,
        ),
        (
            "figures/prompt_strategy_comparison.png",
            "prompt_strategy_comparison.jpg",
            (1000, 650),
            78,
        ),
        (
            "figures/prompt_variant_recall_heatmap.png",
            "prompt_variant_recall_heatmap.jpg",
            (1000, 600),
            78,
        ),
        (
            "examples/prompt_baseline.png",
            "prompt_baseline.jpg",
            (640, 480),
            65,
        ),
        (
            "examples/category_specific_prompt.png",
            "category_specific_prompt.jpg",
            (640, 480),
            65,
        ),
    ]
    for source_name, output_name, max_size, quality in conversions:
        source_path = results_dir / source_name
        if not source_path.exists():
            raise FileNotFoundError(
                f"Missing report image {source_path}; run `yoloworld-experiment visualize` first."
            )
        with Image.open(source_path) as image:
            converted = image.convert("RGB")
            converted.thumbnail(max_size, Image.Resampling.LANCZOS)
            converted.save(
                output_dir / output_name,
                format="JPEG",
                quality=quality,
                optimize=True,
            )


def create_artifact(config: dict[str, Any]) -> Path:
    results_dir = resolve_path(config, config["paths"]["results_dir"])
    _prepare_report_assets(results_dir)
    subset_manifest_path = resolve_path(
        config, config["paths"]["subset_manifest"]
    )
    subset_manifest = json.loads(
        subset_manifest_path.read_text(encoding="utf-8")
    )
    validation_entries = [
        item
        for item in subset_manifest["images"]
        if item["split"] == "validation"
    ]
    test_entries = [
        item for item in subset_manifest["images"] if item["split"] == "test"
    ]
    total_image_count = len(subset_manifest["images"])
    validation_image_count = len(validation_entries)
    test_image_count = len(test_entries)
    validation_gt_count = sum(
        len(item["annotations"]) for item in validation_entries
    )
    test_gt_count = sum(len(item["annotations"]) for item in test_entries)
    validation_category_gt = {
        category: sum(
            annotation["category"] == category
            for item in validation_entries
            for annotation in item["annotations"]
        )
        for category in config["categories"]
    }
    test_small_gt_count = sum(
        float(annotation["area"]) < 32**2
        for item in test_entries
        for annotation in item["annotations"]
    )
    summary = pd.read_csv(results_dir / "summary_metrics.csv")
    categories = pd.read_csv(results_dir / "metrics_by_category.csv")
    prompt_strategies = pd.read_csv(
        results_dir / "prompt_strategy_metrics.csv"
    )
    detection_ap = pd.read_csv(results_dir / "detection_ap_metrics.csv")
    bootstrap_deltas = pd.read_csv(
        results_dir / "bootstrap_deltas_vs_baseline.csv"
    )
    environment = json.loads(
        (results_dir / "environment_metadata.json").read_text(encoding="utf-8")
    )
    selected = json.loads(
        (results_dir / "selected_parameters.json").read_text(encoding="utf-8")
    )
    prompt_selected = json.loads(
        (results_dir / "prompt_selected_parameters.json").read_text(
            encoding="utf-8"
        )
    )
    sqlite_path = results_dir / "report_data.sqlite"
    with sqlite3.connect(sqlite_path) as connection:
        summary.to_sql("summary_metrics", connection, if_exists="replace", index=False)
        categories.to_sql("category_metrics", connection, if_exists="replace", index=False)
        prompt_strategies.to_sql(
            "prompt_strategy_metrics",
            connection,
            if_exists="replace",
            index=False,
        )
        detection_ap.to_sql(
            "detection_ap_metrics",
            connection,
            if_exists="replace",
            index=False,
        )
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    row_by_condition = summary.set_index("condition")
    original = row_by_condition.loc["original"]
    dark = row_by_condition.loc["lowlight_baseline"]
    gamma = row_by_condition.loc["gamma_correction"]
    blur = row_by_condition.loc["blur_sigma_4"]
    synonym = row_by_condition.loc["prompt_synonym"]
    hypernym = row_by_condition.loc["prompt_hypernym"]
    ensemble = row_by_condition.loc["prompt_ensemble"]
    prompt_test = prompt_strategies[
        prompt_strategies["scope"] == "test"
    ].set_index("strategy")
    prompt_baseline = prompt_test.loc["canonical_baseline"]
    prompt_primary = prompt_test.loc[prompt_selected["primary_strategy"]]
    prompt_fixed = prompt_test.loc["category_best_fixed"]
    ap_test = detection_ap[detection_ap["scope"] == "test"].set_index(
        "condition"
    )
    baseline_ap = ap_test.loc["strategy/canonical_baseline"]
    primary_ap = ap_test.loc[
        f"strategy/{prompt_selected['primary_strategy']}"
    ]
    primary_delta = bootstrap_deltas[
        (bootstrap_deltas["metric"] == "f1")
        & (
            bootstrap_deltas["candidate"]
            == f"strategy/{prompt_selected['primary_strategy']}"
        )
    ].iloc[0]

    degradation_sql = _metric_query(
        [
            "original",
            "brightness_0.50",
            "brightness_0.25",
            "blur_sigma_2",
            "blur_sigma_4",
        ],
        {
            "original": "Original",
            "brightness_0.50": "Brightness 50%",
            "brightness_0.25": "Brightness 25%",
            "blur_sigma_2": "Blur σ=2",
            "blur_sigma_4": "Blur σ=4",
        },
    )
    prompt_sql = _metric_query(
        ["prompt_canonical", "prompt_synonym", "prompt_hypernym"],
        {
            "prompt_canonical": "Canonical",
            "prompt_synonym": "Synonym",
            "prompt_hypernym": "Description / hypernym",
        },
    )
    improvement_sql = _metric_query(
        [
            "lowlight_baseline",
            "gamma_correction",
            "baseline_tuned",
            "prompt_ensemble",
        ],
        {
            "lowlight_baseline": "Low-light baseline",
            "gamma_correction": "Gamma correction",
            "baseline_tuned": "Single prompt",
            "prompt_ensemble": "Prompt ensemble",
        },
    )
    key_conditions = [
        ("original", "Original"),
        ("brightness_0.25", "Brightness 25%"),
        ("blur_sigma_4", "Blur σ=4"),
        ("prompt_synonym", "Synonym"),
        ("prompt_hypernym", "Description / hypernym"),
        ("gamma_correction", "Gamma correction"),
        ("prompt_ensemble", "Prompt ensemble"),
    ]
    key_case = "CASE condition " + " ".join(
        f"WHEN '{condition}' THEN '{label}'" for condition, label in key_conditions
    ) + " END"
    key_condition_list = ", ".join(
        f"'{condition}'" for condition, _ in key_conditions
    )
    key_results_sql = (
        f"SELECT {key_case} AS condition, tp, fp, fn, precision, recall, f1 "
        "FROM summary_metrics "
        f"WHERE condition IN ({key_condition_list}) ORDER BY f1 DESC"
    )
    headline_sql = (
        "SELECT "
        "MAX(CASE WHEN condition = 'original' THEN f1 END) AS baseline_f1, "
        "MAX(CASE WHEN condition = 'blur_sigma_4' THEN f1 END) AS blur_f1, "
        "MAX(CASE WHEN condition = 'blur_sigma_4' THEN f1 END) - "
        "MAX(CASE WHEN condition = 'original' THEN f1 END) AS blur_delta, "
        "MAX(CASE WHEN condition = 'prompt_hypernym' THEN f1 END) AS hypernym_f1, "
        "MAX(CASE WHEN condition = 'prompt_hypernym' THEN f1 END) - "
        "MAX(CASE WHEN condition = 'original' THEN f1 END) AS hypernym_delta, "
        "MAX(CASE WHEN condition = 'gamma_correction' THEN f1 END) AS gamma_f1, "
        "MAX(CASE WHEN condition = 'gamma_correction' THEN f1 END) - "
        "MAX(CASE WHEN condition = 'lowlight_baseline' THEN f1 END) AS gamma_delta "
        "FROM summary_metrics"
    )
    with sqlite3.connect(sqlite_path) as connection:
        headline = _query_rows(connection, headline_sql)
        degradation = _query_rows(connection, degradation_sql)
        prompt = _query_rows(connection, prompt_sql)
        improvement = _query_rows(connection, improvement_sql)
        key_results = _query_rows(connection, key_results_sql)

    car = categories[
        (categories["category"] == "car")
        & categories["condition"].isin(
            ["prompt_canonical", "prompt_synonym", "prompt_hypernym"]
        )
    ].set_index("condition")
    cup = categories[
        (categories["category"] == "cup")
        & categories["condition"].isin(["prompt_canonical", "prompt_synonym"])
    ].set_index("condition")

    sources = [
        {
            "id": "headline_sql",
            "label": "Headline metric query",
            "path": "results/yoloworld/report_data.sqlite",
            "query": {
                "engine": "sqlite",
                "language": "sql",
                "sql": headline_sql,
                "description": "Computes baseline, failure-case, and improvement headline F1 values.",
                "tables_used": ["summary_metrics"],
                "metric_definitions": [
                    "F1 is the harmonic mean of micro-aggregated Precision and Recall.",
                    "Deltas are candidate F1 minus the corresponding baseline F1.",
                ],
            },
        },
        {
            "id": "degradation_sql",
            "label": "Image degradation metric query",
            "path": "results/yoloworld/report_data.sqlite",
            "query": {
                "engine": "sqlite",
                "language": "sql",
                "sql": degradation_sql,
                "description": "Returns Precision, Recall, and F1 for the five image input conditions.",
                "tables_used": ["summary_metrics"],
            },
        },
        {
            "id": "prompt_sql",
            "label": "Prompt wording metric query",
            "path": "results/yoloworld/report_data.sqlite",
            "query": {
                "engine": "sqlite",
                "language": "sql",
                "sql": prompt_sql,
                "description": "Returns Precision, Recall, and F1 for the three prompt wording groups.",
                "tables_used": ["summary_metrics"],
            },
        },
        {
            "id": "improvement_sql",
            "label": "Improvement comparison query",
            "path": "results/yoloworld/report_data.sqlite",
            "query": {
                "engine": "sqlite",
                "language": "sql",
                "sql": improvement_sql,
                "description": "Returns low-light and prompt-ensemble before/after metrics.",
                "tables_used": ["summary_metrics"],
            },
        },
        {
            "id": "key_results_sql",
            "label": "Key result table query",
            "path": "results/yoloworld/report_data.sqlite",
            "query": {
                "engine": "sqlite",
                "language": "sql",
                "sql": key_results_sql,
                "description": "Returns the reviewed test rows shown in the report table.",
                "tables_used": ["summary_metrics"],
                "metric_definitions": [
                    "Precision = TP / (TP + FP).",
                    "Recall = TP / (TP + FN).",
                    "F1 = harmonic mean of Precision and Recall.",
                ],
            },
        },
        {
            "id": "summary_metrics",
            "label": "Test aggregate metrics",
            "path": "results/yoloworld/summary_metrics.csv",
        },
        {
            "id": "category_metrics",
            "label": "Per-category prompt metrics",
            "path": "results/yoloworld/metrics_by_category.csv",
        },
        {
            "id": "prompt_strategy_metrics",
            "label": "Validation and test prompt strategy metrics",
            "path": "results/yoloworld/prompt_strategy_metrics.csv",
        },
        {
            "id": "prompt_parameters",
            "label": "Validation-selected prompt parameters",
            "path": "results/yoloworld/prompt_selected_parameters.json",
        },
        {
            "id": "detection_ap_metrics",
            "label": "COCO AP and AR metrics",
            "path": "results/yoloworld/detection_ap_metrics.csv",
        },
        {
            "id": "bootstrap_deltas",
            "label": "Paired image-bootstrap metric deltas",
            "path": "results/yoloworld/bootstrap_deltas_vs_baseline.csv",
        },
        {
            "id": "experiment_manifest",
            "label": "COCO subset manifest",
            "path": str(config["paths"]["subset_manifest"]),
        },
        {
            "id": "yoloworld_paper",
            "label": "YOLO-World paper (CVPR 2024)",
            "href": "https://openaccess.thecvf.com/content/CVPR2024/html/Cheng_YOLO-World_Real-Time_Open-Vocabulary_Object_Detection_CVPR_2024_paper.html",
        },
    ]

    manifest = {
        "version": 1,
        "surface": "report",
        "title": TITLE,
        "description": f"COCO 2017 valの{total_image_count}枚を用いたYOLO-WorldのFailure Case評価。",
        "generatedAt": generated_at,
        "sources": sources,
        "cards": [
            {
                "id": "baseline_card",
                "description": "通常画像・canonical promptのF1。",
                "dataset": "headline",
                "sourceId": "headline_sql",
                "metrics": [{"label": "Baseline F1", "field": "baseline_f1", "format": "percent"}],
            },
            {
                "id": "blur_card",
                "description": "Gaussian blur σ=4でのF1。",
                "dataset": "headline",
                "sourceId": "headline_sql",
                "metrics": [
                    {"label": "Blur σ=4 F1", "field": "blur_f1", "format": "percent"},
                    {"label": "vs baseline", "field": "blur_delta", "format": "percent", "signed": True},
                ],
            },
            {
                "id": "prompt_card",
                "description": "説明句・上位概念promptでのF1。",
                "dataset": "headline",
                "sourceId": "headline_sql",
                "metrics": [
                    {"label": "Hypernym F1", "field": "hypernym_f1", "format": "percent"},
                    {"label": "vs canonical", "field": "hypernym_delta", "format": "percent", "signed": True},
                ],
            },
            {
                "id": "gamma_card",
                "description": "低照度入力にgamma correctionを適用したF1。",
                "dataset": "headline",
                "sourceId": "headline_sql",
                "metrics": [
                    {"label": "Gamma F1", "field": "gamma_f1", "format": "percent"},
                    {"label": "vs low-light", "field": "gamma_delta", "format": "percent", "signed": True},
                ],
            },
        ],
        "charts": [
            {
                "id": "degradation_chart",
                "title": "Detection metrics by image degradation",
                "subtitle": f"Blur mainly reduces recall; F1 is {blur.f1:.3f} at σ=4.",
                "type": "bar",
                "dataset": "degradation_metrics",
                "sourceId": "degradation_sql",
                "valueFormat": "percent",
                "encodings": {
                    "x": {"field": "condition", "type": "nominal", "label": "Input condition"},
                    "y": {"field": "value", "type": "quantitative", "label": "Score", "format": "percent"},
                    "color": {"field": "metric", "type": "nominal", "label": "Metric"},
                    "tooltip": [
                        {"field": "tp", "type": "quantitative", "label": "TP"},
                        {"field": "fp", "type": "quantitative", "label": "FP"},
                        {"field": "fn", "type": "quantitative", "label": "FN"},
                    ],
                },
                "yAxisTitle": "Score",
            },
            {
                "id": "prompt_chart",
                "title": "Detection metrics by prompt wording",
                "subtitle": "Broader wording lowers recall, while the effect varies by category.",
                "type": "bar",
                "dataset": "prompt_metrics",
                "sourceId": "prompt_sql",
                "valueFormat": "percent",
                "encodings": {
                    "x": {"field": "condition", "type": "nominal", "label": "Prompt type"},
                    "y": {"field": "value", "type": "quantitative", "label": "Score", "format": "percent"},
                    "color": {"field": "metric", "type": "nominal", "label": "Metric"},
                },
                "yAxisTitle": "Score",
            },
            {
                "id": "improvement_chart",
                "title": "Detection metrics before and after proposed changes",
                "subtitle": "Gamma correction and prompt ensembling are compared with their respective baselines.",
                "type": "bar",
                "dataset": "improvement_metrics",
                "sourceId": "improvement_sql",
                "valueFormat": "percent",
                "encodings": {
                    "x": {"field": "condition", "type": "nominal", "label": "Method"},
                    "y": {"field": "value", "type": "quantitative", "label": "Score", "format": "percent"},
                    "color": {"field": "metric", "type": "nominal", "label": "Metric"},
                },
                "yAxisTitle": "Score",
            },
        ],
        "tables": [
            {
                "id": "key_results_table",
                "title": "Key test results",
                "subtitle": f"All rows use the same {test_image_count} test images and {test_gt_count} ground-truth boxes.",
                "dataset": "key_results",
                "sourceId": "key_results_sql",
                "defaultSort": {"field": "f1", "direction": "desc"},
                "density": "dense",
                "columns": [
                    {"field": "condition", "label": "Condition", "type": "text"},
                    {"field": "tp", "label": "TP", "format": "number"},
                    {"field": "fp", "label": "FP", "format": "number"},
                    {"field": "fn", "label": "FN", "format": "number"},
                    {"field": "precision", "label": "Precision", "format": "percent"},
                    {"field": "recall", "label": "Recall", "format": "percent"},
                    {"field": "f1", "label": "F1", "format": "percent"},
                ],
            }
        ],
        "blocks": [
            {"id": "title", "type": "markdown", "body": f"# {TITLE}"},
            {
                "id": "identity",
                "type": "markdown",
                "body": (
                    "氏名: **要記入** / 学籍番号: **要記入** / 所属専攻: **要記入**  \n"
                    "所属研究室: **要記入** / 研究テーマ: **要記入**"
                ),
            },
            {
                "id": "technical_summary",
                "type": "markdown",
                "sourceId": "summary_metrics",
                "body": (
                    "## Technical summary\n\n"
                    f"COCO 2017 valから抽出した{total_image_count}枚を、"
                    f"validation {validation_image_count}枚とtest {test_image_count}枚に分離した。"
                    f"通常画像のF1は **{original.f1:.3f}**。blur σ=4では **{blur.f1:.3f}**、"
                    f"説明句・上位概念promptでは **{hypernym.f1:.3f}**まで低下した。"
                    f"低照度入力へのgamma correctionはF1を **{dark.f1:.3f} → {gamma.f1:.3f}** "
                    "へ変化させた。validationで選んだprompt戦略のtest F1は "
                    f"**{prompt_baseline.f1:.3f} → {prompt_primary.f1:.3f}**、"
                    f"COCO mAPは **{baseline_ap.ap:.3f} → {primary_ap.ap:.3f}**だった。"
                ),
            },
            {
                "id": "headline_metrics",
                "type": "metric-strip",
                "cardIds": ["baseline_card", "blur_card", "prompt_card", "gamma_card"],
            },
            {
                "id": "paper",
                "type": "markdown",
                "sourceId": "yoloworld_paper",
                "body": (
                    "## 対象論文と手法\n\n"
                    "Cheng et al., **YOLO-World: Real-Time Open-Vocabulary Object Detection** "
                    "(CVPR 2024)を対象とした。YOLO系検出器にCLIP text encoderとRepVL-PANを組み合わせ、"
                    "region-text contrastive learningで領域と語彙を対応付ける。推論時はprompt埋め込みを"
                    "先に計算するprompt-then-detect方式を用いる。検出器には"
                    "`yolov8s-worldv2.pt`を用い、ローカルGPUで実行した。"
                ),
            },
            {
                "id": "failure_a",
                "type": "markdown",
                "sourceId": "summary_metrics",
                "body": (
                    "## Failure Case A — 画像劣化\n\n"
                    f"通常画像からblur σ=4へ変えるとTPは **{int(original.tp)} → {int(blur.tp)}**、"
                    f"FNは **{int(original.fn)} → {int(blur.fn)}**、Recallは "
                    f"**{original.recall:.3f} → {blur.recall:.3f}**となった。Precisionは"
                    f" **{original.precision:.3f} → {blur.precision:.3f}**となった。"
                    "ぼけにより輪郭や局所textureが失われ、boxを出せなくなった可能性があるが、"
                    "内部特徴を可視化していないため因果的説明ではなく結果に整合する仮説である。"
                ),
            },
            {"id": "failure_a_chart", "type": "chart", "chartId": "degradation_chart"},
            {
                "id": "failure_a_graph",
                "type": "html",
                "body": _full_width_image_html(
                    results_dir / "report_assets" / "failure_image_corruptions.jpg",
                    "Figure 1. Precision, Recall, and F1 across image degradation conditions.",
                ),
            },
            {
                "id": "failure_a_example",
                "type": "html",
                "body": _example_html(
                    [
                        (
                            results_dir / "report_assets" / "baseline_success.jpg",
                            "成功例: 通常画像でGTと予測が一致",
                        ),
                        (
                            results_dir
                            / "report_assets"
                            / "failure_blur_sigma4.jpg",
                            "失敗例: blur σ=4で正解対象を検出できない",
                        ),
                    ]
                ),
            },
            {
                "id": "failure_b",
                "type": "markdown",
                "sourceId": "summary_metrics",
                "body": (
                    "## Failure Case B — prompt表現\n\n"
                    f"canonical、synonym、説明句・上位概念のF1は **{original.f1:.3f} / "
                    f"{synonym.f1:.3f} / {hypernym.f1:.3f}**。カテゴリ別では`car`のRecallが "
                    f"**{car.loc['prompt_canonical'].recall:.3f} → "
                    f"{car.loc['prompt_synonym'].recall:.3f} → "
                    f"{car.loc['prompt_hypernym'].recall:.3f}**へ低下した。一方`cup`では"
                    f"`mug`のF1 **{cup.loc['prompt_synonym'].f1:.3f}**が`cup`の "
                    f"**{cup.loc['prompt_canonical'].f1:.3f}**を上回り、同義語が一様に悪いわけではない。"
                ),
            },
            {
                "id": "failure_b_chart",
                "type": "html",
                "body": _full_width_image_html(
                    results_dir / "report_assets" / "failure_prompt_wording.jpg",
                    "Figure 2. Precision, Recall, and F1 across prompt wording groups.",
                ),
            },
            {
                "id": "prompt_optimization",
                "type": "markdown",
                "sourceId": "prompt_strategy_metrics",
                "body": (
                    "## Prompt改善の比較\n\n"
                    "canonical、synonym、hypernym、写真template、文template、複数形の6種類を用意し、"
                    "カテゴリ別語選択、threshold calibration、subset NMS、validation信頼度付き"
                    "weighted fusionを比較した。語と全パラメータは"
                    f"validation {validation_image_count}枚だけで選択した。\n\n"
                    f"validationで選ばれた本命は`{prompt_selected['primary_strategy']}`。"
                    f"test F1は **{prompt_baseline.f1:.3f} → {prompt_primary.f1:.3f}** "
                    f"(差 {primary_delta['estimate']:+.3f}, paired bootstrap 95%区間 "
                    f"[{primary_delta['ci_low']:+.3f}, {primary_delta['ci_high']:+.3f}])。"
                    f"COCO mAPは **{baseline_ap.ap:.3f} → {primary_ap.ap:.3f}**だった。"
                    "validationでの上昇がtestへ移るかを、F1とmAPの両方で確認する。\n\n"
                    f"カテゴリ別に語だけを選ぶ方法はtest F1 **{prompt_fixed.f1:.3f}**で最大だったが、"
                    "validation全体で本命に選ばれた方法ではないため参考結果として扱う。"
                ),
            },
            {
                "id": "prompt_optimization_graphs",
                "type": "html",
                "body": _example_html(
                    [
                        (
                            results_dir
                            / "report_assets"
                            / "prompt_strategy_comparison.jpg",
                            "Figure 3. Validation F1, test F1, and test AP50.",
                        ),
                        (
                            results_dir
                            / "report_assets"
                            / "prompt_variant_recall_heatmap.jpg",
                            "Figure 4. Test Recall varies strongly by category and wording.",
                        ),
                    ]
                ),
            },
            {
                "id": "prompt_optimization_examples",
                "type": "html",
                "body": _example_html(
                    [
                        (
                            results_dir
                            / "report_assets"
                            / "prompt_baseline.jpg",
                            "Canonical: couchに加え余分なcup boxを出力",
                        ),
                        (
                            results_dir
                            / "report_assets"
                            / "category_specific_prompt.jpg",
                            "Category-specific: cupの誤検出を除去",
                        ),
                    ]
                ),
            },
            {
                "id": "improvement",
                "type": "markdown",
                "sourceId": "summary_metrics",
                "body": (
                    "## 改善と反証結果\n\n"
                    f"validation {validation_image_count}枚でgamma="
                    f"**{selected['gamma_correction']['gamma']:.2f}**を選択した。"
                    f"低照度F1は **{dark.f1:.3f} → {gamma.f1:.3f}**、Recallは "
                    f"**{dark.recall:.3f} → {gamma.recall:.3f}**。一方、prompt ensembleは"
                    f"単一promptのF1 **{original.f1:.3f}**に対して **{ensemble.f1:.3f}**だった。"
                    "単純統合とカテゴリ別方式は、validation/testの差も含めて比較した。"
                ),
            },
            {
                "id": "improvement_chart_block",
                "type": "html",
                "body": _example_html(
                    [
                        (
                            results_dir
                            / "report_assets"
                            / "improvement_gamma_correction.jpg",
                            "Figure 5. Low-light baseline and gamma correction.",
                        ),
                        (
                            results_dir
                            / "report_assets"
                            / "improvement_prompt_ensemble.jpg",
                            "Figure 6. Single prompt and naive prompt ensemble.",
                        ),
                    ]
                ),
            },
            {
                "id": "improvement_examples",
                "type": "html",
                "body": _example_html(
                    [
                        (
                            results_dir / "report_assets" / "lowlight_baseline.jpg",
                            "改善前: 明るさ25%では対応するTPなし",
                        ),
                        (
                            results_dir / "report_assets" / "gamma_correction.jpg",
                            "改善後: gamma correctionで1件のTPを回復",
                        ),
                    ]
                ),
            },
            {"id": "results_table_block", "type": "table", "tableId": "key_results_table"},
            {
                "id": "methodology",
                "type": "markdown",
                "sourceId": "experiment_manifest",
                "body": (
                    "## Methodology and reproducibility\n\n"
                    "対象は`car / couch / airplane / cup`。confidence 0.25以上の予測を降順に処理し、"
                    "同一カテゴリかつIoU 0.50以上の未対応GT boxへgreedy matchingした。"
                    "画像劣化ではpromptとGTを固定し入力だけを変更、prompt実験では元画像を固定した。"
                    "validationはprompt、閾値、NMS IoU、fusion係数、gammaの選択にのみ使用し、"
                    f"test {test_image_count}枚・GT {test_gt_count}件は固定した。\n\n"
                    f"標準検出指標もpycocotoolsで算出した。canonical baselineはmAP "
                    f"**{baseline_ap.ap:.3f}**、AP50 **{baseline_ap.ap50:.3f}**、"
                    f"AP75 **{baseline_ap.ap75:.3f}**、AR@100 **{baseline_ap.ar_100:.3f}**。"
                    "APはraw confidence 0.01以上のranking、F1は固定またはvalidation選択thresholdを使う。"
                    "不確実性はtest画像単位の2,000回bootstrapで評価した。\n\n"
                    f"環境: {environment['gpu']} / driver {environment['nvidia_driver']} / "
                    f"Python {environment['python']} / PyTorch {environment['torch']} / "
                    f"Ultralytics {environment['ultralytics']}。実行コマンドと依存関係はREADMEと"
                    "`environment.yml`に記録した。"
                ),
            },
            {
                "id": "limitations",
                "type": "markdown",
                "body": (
                    "## Limitations and next steps\n\n"
                    f"- {total_image_count}枚・4カテゴリの限定評価であり、COCO全体の性能を表さない。\n"
                    "- 人工的な暗化・Gaussian blurは実環境のnoiseやmotion blurを完全には再現しない。\n"
                    f"- validationのGTは{validation_gt_count}件で、gamma候補の選択には標本依存性がある。\n"
                    f"- カテゴリ別validation GTは{min(validation_category_gt.values())}〜"
                    f"{max(validation_category_gt.values())}件である。\n"
                    "- prompt改善のpaired bootstrap区間は0を含み、有意な改善を主張できない。\n"
                    f"- testのsmall GTは{test_small_gt_count}件で、サイズ別APは標本数とともに解釈する。\n"
                    "- 次は画像数とカテゴリ別validation GTを増やし、複数splitでprompt選択を評価する。"
                ),
            },
            {
                "id": "ai_and_refs",
                "type": "markdown",
                "body": (
                    "## 生成AI利用と参考文献\n\n"
                    "Codexを課題要件整理、候補論文比較、実験設計、環境構築、評価コード、集計、"
                    "図表、文章草稿に使用した。生成内容は実行結果および参照文献と照合し、"
                    "実験条件、数値、考察は著者が確認した。\n\n"
                    "1. Cheng et al., *YOLO-World*, CVPR 2024.\n"
                    "2. AILab-CVC, Official YOLO-World implementation.\n"
                    "3. Ultralytics YOLO-World documentation.\n"
                    "4. COCO dataset."
                ),
            },
        ],
    }

    snapshot = {
        "version": 1,
        "generatedAt": generated_at,
        "status": "ready",
        "datasets": {
            "headline": headline,
            "degradation_metrics": degradation,
            "prompt_metrics": prompt,
            "improvement_metrics": improvement,
            "key_results": key_results,
        },
    }
    artifact = {
        "surface": "report",
        "manifest": manifest,
        "snapshot": snapshot,
        "sources": sources,
    }
    output_path = results_dir / "artifact.json"
    output_path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path
