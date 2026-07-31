from __future__ import annotations

import json
import hashlib
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from .coco import ground_truth_by_image, load_manifest
from .config import category_names, resolve_path
from .detection_metrics import evaluate_coco_detection, paired_bootstrap_delta
from .metrics import box_iou, evaluate_records
from .model import YoloWorldPredictor
from .prompt_study import _read_prediction_file, _write_prediction_file


def unflip_predictions(
    predictions: list[dict[str, Any]],
    image_width: int,
) -> list[dict[str, Any]]:
    output = []
    for prediction in predictions:
        x1, y1, x2, y2 = [
            float(value) for value in prediction["bbox_xyxy"]
        ]
        output.append(
            {
                **prediction,
                "bbox_xyxy": [
                    float(image_width) - x2,
                    y1,
                    float(image_width) - x1,
                    y2,
                ],
            }
        )
    return output


def fuse_prediction_sources(
    predictions: list[dict[str, Any]],
    nms_iou: float,
    *,
    aircraft_weight: float = 1.0,
) -> list[dict[str, Any]]:
    """Fuse overlapping detections while retaining unique source discoveries."""
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        source = str(prediction.get("inference_source", "unknown"))
        weight = (
            float(aircraft_weight)
            if prediction["category"] == "airplane" and "robust_category" in source
            else 1.0
        )
        by_category[str(prediction["category"])].append(
            {
                **prediction,
                "confidence": min(
                    1.0, float(prediction["confidence"]) * weight
                ),
                "source_weight": weight,
            }
        )

    fused: list[dict[str, Any]] = []
    for category_predictions in by_category.values():
        category_predictions.sort(
            key=lambda item: float(item["confidence"]), reverse=True
        )
        clusters: list[list[dict[str, Any]]] = []
        cluster_boxes: list[list[float]] = []
        for prediction in category_predictions:
            overlaps = [
                box_iou(prediction["bbox_xyxy"], cluster_box)
                for cluster_box in cluster_boxes
            ]
            best_index = int(np.argmax(overlaps)) if overlaps else -1
            if best_index >= 0 and overlaps[best_index] >= float(nms_iou):
                clusters[best_index].append(prediction)
            else:
                clusters.append([prediction])
                cluster_boxes.append(
                    [float(value) for value in prediction["bbox_xyxy"]]
                )

            cluster_index = best_index if best_index >= 0 and overlaps[best_index] >= float(nms_iou) else len(clusters) - 1
            best_by_source: dict[str, dict[str, Any]] = {}
            for member in clusters[cluster_index]:
                source = str(member.get("inference_source", "unknown"))
                if (
                    source not in best_by_source
                    or float(member["confidence"])
                    > float(best_by_source[source]["confidence"])
                ):
                    best_by_source[source] = member
            selected = list(best_by_source.values())
            weights = np.asarray(
                [max(float(item["confidence"]), 1e-6) for item in selected],
                dtype=float,
            )
            boxes = np.asarray(
                [item["bbox_xyxy"] for item in selected], dtype=float
            )
            cluster_boxes[cluster_index] = [
                float(value)
                for value in np.average(boxes, axis=0, weights=weights)
            ]

        for cluster, fused_box in zip(clusters, cluster_boxes):
            best_by_source = {}
            for member in cluster:
                source = str(member.get("inference_source", "unknown"))
                if (
                    source not in best_by_source
                    or float(member["confidence"])
                    > float(best_by_source[source]["confidence"])
                ):
                    best_by_source[source] = member
            selected = list(best_by_source.values())
            representative = max(
                selected, key=lambda item: float(item["confidence"])
            )
            fused.append(
                {
                    **representative,
                    "bbox_xyxy": fused_box,
                    "confidence": max(
                        float(item["confidence"]) for item in selected
                    ),
                    "fusion_support": len(selected),
                    "fusion_sources": sorted(best_by_source),
                }
            )
    return sorted(
        fused, key=lambda item: float(item["confidence"]), reverse=True
    )


def optimize_category_thresholds(
    predictions_by_image: dict[int, list[dict[str, Any]]],
    ground_truth_by_image: dict[int, list[dict[str, Any]]],
    image_ids: list[int],
    categories: list[str],
    thresholds: list[float],
    match_iou: float,
) -> tuple[dict[str, float], dict[str, Any]]:
    counts: dict[tuple[str, float], tuple[int, int, int]] = {}
    for category in categories:
        category_predictions = {
            image_id: [
                prediction
                for prediction in predictions_by_image.get(image_id, [])
                if prediction["category"] == category
            ]
            for image_id in image_ids
        }
        category_ground_truth = {
            image_id: [
                annotation
                for annotation in ground_truth_by_image.get(image_id, [])
                if annotation["category"] == category
            ]
            for image_id in image_ids
        }
        for threshold in thresholds:
            summary, _ = evaluate_records(
                category_predictions,
                category_ground_truth,
                image_ids,
                float(threshold),
                match_iou,
            )
            counts[(category, float(threshold))] = (
                int(summary["tp"]),
                int(summary["fp"]),
                int(summary["fn"]),
            )

    best_key: tuple[float, ...] | None = None
    best_thresholds: dict[str, float] = {}
    best_summary: dict[str, Any] = {}
    for values in product(thresholds, repeat=len(categories)):
        selected = {
            category: float(threshold)
            for category, threshold in zip(categories, values)
        }
        tp = sum(counts[(category, selected[category])][0] for category in categories)
        fp = sum(counts[(category, selected[category])][1] for category in categories)
        fn = sum(counts[(category, selected[category])][2] for category in categories)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        distance = sum(abs(value - 0.25) for value in selected.values())
        key = (f1, recall, precision, -distance)
        if best_key is None or key > best_key:
            best_key = key
            best_thresholds = selected
            best_summary = {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
    return best_thresholds, best_summary


def apply_category_thresholds(
    predictions_by_image: dict[int, list[dict[str, Any]]],
    image_ids: list[int],
    thresholds: dict[str, float],
) -> dict[int, list[dict[str, Any]]]:
    return {
        image_id: [
            prediction
            for prediction in predictions_by_image.get(image_id, [])
            if float(prediction["confidence"])
            >= float(thresholds[str(prediction["category"])])
        ]
        for image_id in image_ids
    }


def _evidence_group(source: str, category: str) -> str:
    if category == "airplane":
        return source
    parts = source.split("_")
    return f"{parts[-2]}_{parts[-1]}"


def build_support_calibrated_predictions(
    baseline_by_image: dict[int, list[dict[str, Any]]],
    supporting_sources: dict[
        str, dict[int, list[dict[str, Any]]]
    ],
    image_ids: list[int],
    categories: list[str],
    *,
    support_iou: float,
    support_bonus: float,
    supplement_weight: float,
    supplement_min_support: int,
    supplement_cluster_iou: float,
) -> dict[int, list[dict[str, Any]]]:
    """Preserve baseline boxes and use independent views as confidence votes."""
    output: dict[int, list[dict[str, Any]]] = {}
    for image_id in image_ids:
        image_predictions: list[dict[str, Any]] = []
        for category in categories:
            baseline = [
                {
                    **prediction,
                    "inference_source": "canonical_640_original",
                }
                for prediction in baseline_by_image.get(image_id, [])
                if prediction["category"] == category
            ]
            grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for source, by_image in supporting_sources.items():
                for prediction in by_image.get(image_id, []):
                    if prediction["category"] != category:
                        continue
                    group = _evidence_group(source, category)
                    grouped[group].append(
                        {
                            **prediction,
                            "inference_source": group,
                        }
                    )

            support_groups = [set() for _ in baseline]
            unused: list[dict[str, Any]] = []
            for group, predictions in grouped.items():
                used_baseline: set[int] = set()
                for prediction in sorted(
                    predictions,
                    key=lambda item: float(item["confidence"]),
                    reverse=True,
                ):
                    candidates = [
                        (
                            index,
                            box_iou(
                                prediction["bbox_xyxy"],
                                base["bbox_xyxy"],
                            ),
                        )
                        for index, base in enumerate(baseline)
                        if index not in used_baseline
                    ]
                    best_index, best_iou = max(
                        candidates,
                        key=lambda item: item[1],
                        default=(-1, 0.0),
                    )
                    if best_index >= 0 and best_iou >= float(support_iou):
                        used_baseline.add(best_index)
                        support_groups[best_index].add(group)
                    else:
                        unused.append(prediction)

            for prediction, groups in zip(baseline, support_groups):
                image_predictions.append(
                    {
                        **prediction,
                        "original_confidence": float(
                            prediction["confidence"]
                        ),
                        "confidence": min(
                            1.0,
                            float(prediction["confidence"])
                            + float(support_bonus) * len(groups),
                        ),
                        "support_count": len(groups),
                        "support_groups": sorted(groups),
                        "strategy_source": "baseline_preserved",
                    }
                )

            supplements = fuse_prediction_sources(
                unused, float(supplement_cluster_iou)
            )
            for supplement in supplements:
                support_count = int(
                    supplement.get("fusion_support", 1)
                )
                overlaps_baseline = any(
                    box_iou(
                        supplement["bbox_xyxy"],
                        prediction["bbox_xyxy"],
                    )
                    >= float(support_iou)
                    for prediction in baseline
                )
                if (
                    support_count < int(supplement_min_support)
                    or overlaps_baseline
                ):
                    continue
                image_predictions.append(
                    {
                        **supplement,
                        "original_confidence": float(
                            supplement["confidence"]
                        ),
                        "confidence": min(
                            1.0,
                            float(supplement["confidence"])
                            * float(supplement_weight)
                            + float(support_bonus)
                            * max(0, support_count - 1),
                        ),
                        "support_count": support_count,
                        "strategy_source": "supported_supplement",
                    }
                )
        output[image_id] = sorted(
            image_predictions,
            key=lambda item: float(item["confidence"]),
            reverse=True,
        )
    return output


class ClearImprovementStudy:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.categories = category_names(config)
        self.images_dir = resolve_path(config, config["paths"]["coco_images"])
        self.results_dir = resolve_path(
            config, config["paths"]["improvement_results_dir"]
        )
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.match_iou = float(config["evaluation"]["match_iou"])
        self.bootstrap_samples = int(config["evaluation"]["bootstrap_samples"])

    def _source_cache_paths(self, scope: str) -> tuple[Path, Path]:
        return (
            self.results_dir / f"{scope}_source_predictions.json",
            self.results_dir / f"{scope}_source_metadata.json",
        )

    def _load_source_cache(
        self, scope: str
    ) -> tuple[
        dict[str, dict[int, list[dict[str, Any]]]],
        dict[str, dict[str, Any]],
    ]:
        predictions_path, metadata_path = self._source_cache_paths(scope)
        predictions = (
            _read_prediction_file(predictions_path)
            if predictions_path.exists()
            else {}
        )
        metadata = (
            json.loads(metadata_path.read_text(encoding="utf-8")).get(
                "sources", {}
            )
            if metadata_path.exists()
            else {}
        )
        return predictions, metadata

    def _save_source_cache(
        self,
        scope: str,
        predictions: dict[str, dict[int, list[dict[str, Any]]]],
        metadata: dict[str, dict[str, Any]],
    ) -> None:
        predictions_path, metadata_path = self._source_cache_paths(scope)
        _write_prediction_file(predictions_path, predictions)
        metadata_path.write_text(
            json.dumps(
                {
                    "model": self.config["model"]["name"],
                    "sources": metadata,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    def _collect_sources(
        self,
        scope: str,
        manifest: dict[str, Any],
        specifications: list[dict[str, Any]],
        *,
        model_name: str | None = None,
    ) -> dict[str, dict[int, list[dict[str, Any]]]]:
        records, metadata = self._load_source_cache(scope)
        image_ids = [int(item["image_id"]) for item in manifest["images"]]
        effective_model = (
            str(model_name)
            if model_name is not None
            else str(self.config["model"]["name"])
        )
        predictor: YoloWorldPredictor | None = None
        for specification in specifications:
            name = str(specification["name"])
            expected = {
                "prompts": list(specification["prompts"]),
                "image_size": int(specification["image_size"]),
                "horizontal_flip": bool(specification["horizontal_flip"]),
                "image_ids": image_ids,
                "model": effective_model,
            }
            if (
                name in records
                and metadata.get(name) == expected
                and set(records[name]) == set(image_ids)
            ):
                continue
            if predictor is None:
                predictor_config = {
                    **self.config,
                    "model": {
                        **self.config["model"],
                        "name": effective_model,
                    },
                }
                predictor = YoloWorldPredictor(
                    predictor_config, self.categories
                )
            predictor.set_prompts(list(specification["prompts"]))
            by_image: dict[int, list[dict[str, Any]]] = {}
            print(
                f"[clear-improvement] inference {scope}: {name} "
                f"({len(image_ids)} images)"
            )
            for entry in manifest["images"]:
                image_id = int(entry["image_id"])
                image = Image.open(
                    self.images_dir / entry["file_name"]
                ).convert("RGB")
                if specification["horizontal_flip"]:
                    transformed = image.transpose(
                        Image.Transpose.FLIP_LEFT_RIGHT
                    )
                    predictions = unflip_predictions(
                        predictor.predict(
                            transformed,
                            image_size=int(specification["image_size"]),
                        ),
                        int(entry["width"]),
                    )
                else:
                    predictions = predictor.predict(
                        image,
                        image_size=int(specification["image_size"]),
                    )
                by_image[image_id] = [
                    {**prediction, "inference_source": name}
                    for prediction in predictions
                ]
            records[name] = by_image
            metadata[name] = expected
            self._save_source_cache(scope, records, metadata)
        return records

    def _development_baseline_source(
        self, image_ids: list[int]
    ) -> dict[int, list[dict[str, Any]]]:
        prompt_records = _read_prediction_file(
            resolve_path(
                self.config,
                self.config["paths"]["results_dir"],
            )
            / "prompt_predictions.json"
        )
        return {
            image_id: [
                {
                    **prediction,
                    "inference_source": "canonical_640_original",
                }
                for prediction in prompt_records["canonical"].get(image_id, [])
            ]
            for image_id in image_ids
        }

    def _combine_sources(
        self,
        source_records: dict[str, dict[int, list[dict[str, Any]]]],
        source_names: list[str],
        image_ids: list[int],
        nms_iou: float,
        aircraft_weight: float,
    ) -> dict[int, list[dict[str, Any]]]:
        return {
            image_id: fuse_prediction_sources(
                [
                    prediction
                    for source in source_names
                    for prediction in source_records[source].get(image_id, [])
                ],
                nms_iou,
                aircraft_weight=aircraft_weight,
            )
            for image_id in image_ids
        }

    def _evaluate_candidate_grid(
        self,
        manifest: dict[str, Any],
        source_records: dict[str, dict[int, list[dict[str, Any]]]],
        candidate_specs: list[dict[str, Any]],
        baseline_raw: dict[int, list[dict[str, Any]]],
        baseline_details: list[dict[str, Any]],
    ) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
        image_ids = [int(item["image_id"]) for item in manifest["images"]]
        ground_truth = ground_truth_by_image(manifest)
        thresholds = [
            float(value)
            for value in self.config["clear_improvement"][
                "confidence_candidates"
            ]
        ]
        nms_values = [
            float(value)
            for value in self.config["clear_improvement"][
                "nms_iou_candidates"
            ]
        ]
        baseline_ap, _, _ = evaluate_coco_detection(
            manifest,
            baseline_raw,
            image_ids,
            self.config["categories"],
            confidence_threshold=0.0,
        )
        rows: list[dict[str, Any]] = []
        settings: dict[str, dict[str, Any]] = {}
        for specification in candidate_specs:
            aircraft_weights = (
                [0.75, 0.90, 1.00, 1.10]
                if specification.get("tune_aircraft_weight", False)
                else [1.0]
            )
            for nms_iou, aircraft_weight in product(
                nms_values, aircraft_weights
            ):
                raw = self._combine_sources(
                    source_records,
                    list(specification["sources"]),
                    image_ids,
                    nms_iou,
                    aircraft_weight,
                )
                selected_thresholds, _ = optimize_category_thresholds(
                    raw,
                    ground_truth,
                    image_ids,
                    self.categories,
                    thresholds,
                    self.match_iou,
                )
                fixed = apply_category_thresholds(
                    raw, image_ids, selected_thresholds
                )
                summary, details = evaluate_records(
                    fixed,
                    ground_truth,
                    image_ids,
                    0.0,
                    self.match_iou,
                )
                ap, _, _ = evaluate_coco_detection(
                    manifest,
                    raw,
                    image_ids,
                    self.config["categories"],
                    confidence_threshold=0.0,
                )
                f1_delta = next(
                    row
                    for row in paired_bootstrap_delta(
                        baseline_details,
                        details,
                        self.bootstrap_samples,
                        int(self.config["seed"]) + 17,
                    )
                    if row["metric"] == "f1"
                )
                candidate_id = (
                    f"{specification['name']}__nms{nms_iou:.2f}"
                    f"__air{aircraft_weight:.2f}"
                )
                row = {
                    "candidate_id": candidate_id,
                    "candidate": str(specification["name"]),
                    "sources": "|".join(specification["sources"]),
                    "nms_iou": nms_iou,
                    "aircraft_weight": aircraft_weight,
                    "thresholds": json.dumps(selected_thresholds, sort_keys=True),
                    **summary,
                    "ap": float(ap["ap"]),
                    "ap50": float(ap["ap50"]),
                    "ap75": float(ap["ap75"]),
                    "baseline_ap": float(baseline_ap["ap"]),
                    "f1_delta": float(f1_delta["estimate"]),
                    "f1_delta_ci_low": float(f1_delta["ci_low"]),
                    "f1_delta_ci_high": float(f1_delta["ci_high"]),
                }
                rows.append(row)
                settings[candidate_id] = {
                    "candidate": str(specification["name"]),
                    "sources": list(specification["sources"]),
                    "nms_iou": nms_iou,
                    "aircraft_weight": aircraft_weight,
                    "thresholds": selected_thresholds,
                    "dev_metrics": row,
                }
        return pd.DataFrame(rows), settings

    def run_development(self) -> dict[str, Any]:
        manifest = load_manifest(self.config)
        image_ids = [int(item["image_id"]) for item in manifest["images"]]
        ground_truth = ground_truth_by_image(manifest)
        prompt_sets = self.config["clear_improvement"]["prompt_sets"]
        canonical_prompts = list(prompt_sets["canonical"])
        image_sizes = [
            int(value)
            for value in self.config["clear_improvement"]["image_sizes"]
        ]
        source_specs = [
            {
                "name": f"canonical_{image_size}_original",
                "prompts": canonical_prompts,
                "image_size": image_size,
                "horizontal_flip": False,
            }
            for image_size in image_sizes
        ]
        sources = self._collect_sources("development", manifest, source_specs)
        sources["canonical_640_original"] = self._development_baseline_source(
            image_ids
        )
        baseline_fixed = {
            image_id: [
                prediction
                for prediction in sources["canonical_640_original"][image_id]
                if float(prediction["confidence"])
                >= float(self.config["evaluation"]["confidence"])
            ]
            for image_id in image_ids
        }
        baseline_summary, baseline_details = evaluate_records(
            baseline_fixed,
            ground_truth,
            image_ids,
            0.0,
            self.match_iou,
        )
        stage_one_specs = [
            {
                "name": f"canonical_{image_size}",
                "sources": [f"canonical_{image_size}_original"],
            }
            for image_size in [640, *image_sizes]
        ]
        stage_one, stage_one_settings = self._evaluate_candidate_grid(
            manifest,
            sources,
            stage_one_specs,
            sources["canonical_640_original"],
            baseline_details,
        )
        high_resolution = stage_one[
            stage_one["candidate"] != "canonical_640"
        ].sort_values(
            ["f1", "ap", "f1_delta_ci_low"],
            ascending=[False, False, False],
        )
        best_size = int(
            str(high_resolution.iloc[0]["candidate"]).split("_")[-1]
        )

        expanded_specs = []
        for prompt_name, prompts in prompt_sets.items():
            for flip in (False, True):
                expanded_specs.append(
                    {
                        "name": (
                            f"{prompt_name}_{best_size}_"
                            f"{'flip' if flip else 'original'}"
                        ),
                        "prompts": list(prompts),
                        "image_size": best_size,
                        "horizontal_flip": flip,
                    }
                )
        sources.update(
            self._collect_sources(
                "development", manifest, expanded_specs
            )
        )
        canonical_original = f"canonical_{best_size}_original"
        canonical_flip = f"canonical_{best_size}_flip"
        robust_original = f"robust_category_{best_size}_original"
        robust_flip = f"robust_category_{best_size}_flip"
        stage_two_specs = [
            {
                "name": f"robust_category_{best_size}",
                "sources": [robust_original],
            },
            {
                "name": f"canonical_flip_tta_{best_size}",
                "sources": [canonical_original, canonical_flip],
            },
            {
                "name": f"prompt_union_{best_size}",
                "sources": [canonical_original, robust_original],
                "tune_aircraft_weight": True,
            },
            {
                "name": f"prompt_union_flip_tta_{best_size}",
                "sources": [
                    canonical_original,
                    canonical_flip,
                    robust_original,
                    robust_flip,
                ],
                "tune_aircraft_weight": True,
            },
            {
                "name": f"canonical_multiscale_640_{best_size}",
                "sources": [
                    "canonical_640_original",
                    canonical_original,
                ],
            },
            {
                "name": f"prompt_multiscale_640_{best_size}",
                "sources": [
                    "canonical_640_original",
                    canonical_original,
                    robust_original,
                ],
                "tune_aircraft_weight": True,
            },
            {
                "name": f"prompt_multiscale_flip_tta_640_{best_size}",
                "sources": [
                    "canonical_640_original",
                    canonical_original,
                    canonical_flip,
                    robust_original,
                    robust_flip,
                ],
                "tune_aircraft_weight": True,
            },
        ]
        stage_two, stage_two_settings = self._evaluate_candidate_grid(
            manifest,
            sources,
            stage_two_specs,
            sources["canonical_640_original"],
            baseline_details,
        )
        metrics = pd.concat([stage_one, stage_two], ignore_index=True)
        settings = {**stage_one_settings, **stage_two_settings}
        minimum_gain = float(
            self.config["clear_improvement"]["minimum_f1_gain"]
        )
        maximum_map_drop = float(
            self.config["clear_improvement"]["maximum_map_drop"]
        )
        eligible = metrics[
            (metrics["f1_delta"] >= minimum_gain)
            & (metrics["f1_delta_ci_low"] > 0.0)
            & (
                metrics["ap"]
                >= metrics["baseline_ap"] - maximum_map_drop
            )
        ]
        ranking = eligible if not eligible.empty else metrics
        best_row = ranking.sort_values(
            ["f1", "ap", "f1_delta_ci_low"],
            ascending=[False, False, False],
        ).iloc[0]
        selected = settings[str(best_row["candidate_id"])]
        selected.update(
            {
                "selection_scope": "development_100",
                "best_image_size": best_size,
                "development_gate_passed": not eligible.empty,
                "baseline_metrics": baseline_summary,
                "success_criteria": {
                    "minimum_f1_gain": minimum_gain,
                    "minimum_f1_delta_ci_low": 0.0,
                    "maximum_map_drop": maximum_map_drop,
                },
            }
        )
        metrics.sort_values(
            ["f1", "ap"], ascending=[False, False]
        ).to_csv(self.results_dir / "development_candidate_metrics.csv", index=False)
        (self.results_dir / "selected_improvement.json").write_text(
            json.dumps(selected, indent=2) + "\n",
            encoding="utf-8",
        )
        return selected

    def run_holdout(self) -> dict[str, Any]:
        selected_path = self.results_dir / "selected_improvement.json"
        selected_bytes = selected_path.read_bytes()
        selected = json.loads(selected_bytes.decode("utf-8"))
        if not selected.get("development_gate_passed", False):
            raise RuntimeError(
                "Development gate did not pass; do not inspect the holdout yet"
            )
        manifest_path = resolve_path(
            self.config, self.config["paths"]["holdout_manifest"]
        )
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes.decode("utf-8"))
        image_ids = [int(item["image_id"]) for item in manifest["images"]]
        ground_truth = ground_truth_by_image(manifest)
        prompt_sets = self.config["clear_improvement"]["prompt_sets"]

        specifications = []
        source_names = set(selected["sources"]) | {
            "canonical_640_original"
        }
        for name in sorted(source_names):
            parts = name.split("_")
            flip = parts[-1] == "flip"
            image_size = int(parts[-2])
            prompt_name = (
                "robust_category"
                if name.startswith("robust_category_")
                else "canonical"
            )
            specifications.append(
                {
                    "name": name,
                    "prompts": list(prompt_sets[prompt_name]),
                    "image_size": image_size,
                    "horizontal_flip": flip,
                }
            )
        sources = self._collect_sources(
            "holdout", manifest, specifications
        )
        baseline_raw = sources["canonical_640_original"]
        baseline_fixed = {
            image_id: [
                prediction
                for prediction in baseline_raw[image_id]
                if float(prediction["confidence"])
                >= float(self.config["evaluation"]["confidence"])
            ]
            for image_id in image_ids
        }
        candidate_raw = self._combine_sources(
            sources,
            list(selected["sources"]),
            image_ids,
            float(selected["nms_iou"]),
            float(selected["aircraft_weight"]),
        )
        candidate_fixed = apply_category_thresholds(
            candidate_raw,
            image_ids,
            {
                category: float(value)
                for category, value in selected["thresholds"].items()
            },
        )
        baseline_summary, baseline_details = evaluate_records(
            baseline_fixed, ground_truth, image_ids, 0.0, self.match_iou
        )
        candidate_summary, candidate_details = evaluate_records(
            candidate_fixed, ground_truth, image_ids, 0.0, self.match_iou
        )
        baseline_ap, _, _ = evaluate_coco_detection(
            manifest,
            baseline_raw,
            image_ids,
            self.config["categories"],
            confidence_threshold=0.0,
        )
        candidate_ap, _, _ = evaluate_coco_detection(
            manifest,
            candidate_raw,
            image_ids,
            self.config["categories"],
            confidence_threshold=0.0,
        )
        deltas = paired_bootstrap_delta(
            baseline_details,
            candidate_details,
            self.bootstrap_samples,
            int(self.config["selection"]["holdout_seed"]) + 29,
        )
        f1_delta = next(row for row in deltas if row["metric"] == "f1")
        minimum_gain = float(
            self.config["clear_improvement"]["minimum_f1_gain"]
        )
        maximum_map_drop = float(
            self.config["clear_improvement"]["maximum_map_drop"]
        )
        passed = (
            float(f1_delta["estimate"]) >= minimum_gain
            and float(f1_delta["ci_low"]) > 0.0
            and float(candidate_ap["ap"])
            >= float(baseline_ap["ap"]) - maximum_map_drop
        )
        summary_rows = [
            {
                "scope": "holdout",
                "condition": "canonical_baseline",
                **baseline_summary,
                "ap": baseline_ap["ap"],
                "ap50": baseline_ap["ap50"],
                "ap75": baseline_ap["ap75"],
            },
            {
                "scope": "holdout",
                "condition": "selected_improvement",
                **candidate_summary,
                "ap": candidate_ap["ap"],
                "ap50": candidate_ap["ap50"],
                "ap75": candidate_ap["ap75"],
            },
        ]
        pd.DataFrame(summary_rows).to_csv(
            self.results_dir / "holdout_summary.csv", index=False
        )
        pd.DataFrame(
            [
                {"condition": "canonical_baseline", **row}
                for row in baseline_details
            ]
            + [
                {"condition": "selected_improvement", **row}
                for row in candidate_details
            ]
        ).to_csv(self.results_dir / "holdout_per_image.csv", index=False)
        pd.DataFrame(deltas).to_csv(
            self.results_dir / "holdout_bootstrap_deltas.csv", index=False
        )
        verdict = {
            "passed": passed,
            "selected_candidate": selected["candidate"],
            "baseline": summary_rows[0],
            "candidate": summary_rows[1],
            "f1_delta": f1_delta,
            "map_delta": float(candidate_ap["ap"] - baseline_ap["ap"]),
            "selection_was_frozen_before_holdout": True,
            "selected_parameters_sha256": hashlib.sha256(
                selected_bytes
            ).hexdigest(),
            "holdout_manifest_sha256": hashlib.sha256(
                manifest_bytes
            ).hexdigest(),
        }
        (self.results_dir / "holdout_verdict.json").write_text(
            json.dumps(verdict, indent=2) + "\n", encoding="utf-8"
        )
        _write_prediction_file(
            self.results_dir / "holdout_evaluation_predictions.json",
            {
                "canonical_baseline": baseline_fixed,
                "selected_improvement": candidate_fixed,
            },
        )
        return verdict

    def _combined_development_data(
        self,
    ) -> tuple[
        dict[str, Any],
        dict[str, dict[int, list[dict[str, Any]]]],
    ]:
        first = load_manifest(self.config)
        second_path = resolve_path(
            self.config, self.config["paths"]["holdout_manifest"]
        )
        second = json.loads(second_path.read_text(encoding="utf-8"))
        combined = {
            "source": "COCO 2017 val development union after replication",
            "images": [*first["images"], *second["images"]],
        }
        first_sources = _read_prediction_file(
            self.results_dir / "development_source_predictions.json"
        )
        first_ids = [int(item["image_id"]) for item in first["images"]]
        first_sources["canonical_640_original"] = (
            self._development_baseline_source(first_ids)
        )
        second_sources = _read_prediction_file(
            self.results_dir / "holdout_source_predictions.json"
        )
        required = [
            "canonical_640_original",
            "canonical_768_original",
            "canonical_768_flip",
            "robust_category_768_original",
            "robust_category_768_flip",
        ]
        merged: dict[str, dict[int, list[dict[str, Any]]]] = {}
        for source in required:
            merged[source] = {
                **first_sources[source],
                **second_sources[source],
            }
        return combined, merged

    def run_support_development(self) -> dict[str, Any]:
        manifest, sources = self._combined_development_data()
        image_ids = [int(item["image_id"]) for item in manifest["images"]]
        ground_truth = ground_truth_by_image(manifest)
        baseline_raw = sources["canonical_640_original"]
        baseline_fixed = {
            image_id: [
                prediction
                for prediction in baseline_raw[image_id]
                if float(prediction["confidence"])
                >= float(self.config["evaluation"]["confidence"])
            ]
            for image_id in image_ids
        }
        baseline_summary, baseline_details = evaluate_records(
            baseline_fixed, ground_truth, image_ids, 0.0, self.match_iou
        )
        baseline_ap, _, _ = evaluate_coco_detection(
            manifest,
            baseline_raw,
            image_ids,
            self.config["categories"],
            confidence_threshold=0.0,
        )
        supporting = {
            source: sources[source]
            for source in (
                "canonical_768_original",
                "canonical_768_flip",
                "robust_category_768_original",
                "robust_category_768_flip",
            )
        }
        thresholds = [
            float(value)
            for value in self.config["clear_improvement"][
                "confidence_candidates"
            ]
        ]
        grid = product(
            self.config["clear_improvement"]["support_iou_candidates"],
            self.config["clear_improvement"]["support_bonus_candidates"],
            self.config["clear_improvement"][
                "supplement_weight_candidates"
            ],
            self.config["clear_improvement"][
                "supplement_min_support_candidates"
            ],
            self.config["clear_improvement"][
                "supplement_cluster_iou_candidates"
            ],
        )
        rows: list[dict[str, Any]] = []
        settings: dict[str, dict[str, Any]] = {}
        for (
            support_iou,
            support_bonus,
            supplement_weight,
            supplement_min_support,
            supplement_cluster_iou,
        ) in grid:
            raw = build_support_calibrated_predictions(
                baseline_raw,
                supporting,
                image_ids,
                self.categories,
                support_iou=float(support_iou),
                support_bonus=float(support_bonus),
                supplement_weight=float(supplement_weight),
                supplement_min_support=int(supplement_min_support),
                supplement_cluster_iou=float(supplement_cluster_iou),
            )
            selected_thresholds, _ = optimize_category_thresholds(
                raw,
                ground_truth,
                image_ids,
                self.categories,
                thresholds,
                self.match_iou,
            )
            fixed = apply_category_thresholds(
                raw, image_ids, selected_thresholds
            )
            summary, details = evaluate_records(
                fixed, ground_truth, image_ids, 0.0, self.match_iou
            )
            f1_delta = next(
                row
                for row in paired_bootstrap_delta(
                    baseline_details,
                    details,
                    self.bootstrap_samples,
                    int(self.config["seed"]) + 41,
                )
                if row["metric"] == "f1"
            )
            ap = {"ap": float("nan"), "ap50": float("nan"), "ap75": float("nan")}
            if float(f1_delta["estimate"]) >= 0.015:
                ap, _, _ = evaluate_coco_detection(
                    manifest,
                    raw,
                    image_ids,
                    self.config["categories"],
                    confidence_threshold=0.0,
                )
            candidate_id = (
                f"si{float(support_iou):.2f}"
                f"_b{float(support_bonus):.3f}"
                f"_w{float(supplement_weight):.2f}"
                f"_s{int(supplement_min_support)}"
                f"_ci{float(supplement_cluster_iou):.2f}"
            )
            row = {
                "candidate_id": candidate_id,
                "support_iou": float(support_iou),
                "support_bonus": float(support_bonus),
                "supplement_weight": float(supplement_weight),
                "supplement_min_support": int(supplement_min_support),
                "supplement_cluster_iou": float(
                    supplement_cluster_iou
                ),
                "thresholds": json.dumps(
                    selected_thresholds, sort_keys=True
                ),
                **summary,
                "ap": float(ap["ap"]),
                "ap50": float(ap["ap50"]),
                "ap75": float(ap["ap75"]),
                "baseline_ap": float(baseline_ap["ap"]),
                "f1_delta": float(f1_delta["estimate"]),
                "f1_delta_ci_low": float(f1_delta["ci_low"]),
                "f1_delta_ci_high": float(f1_delta["ci_high"]),
            }
            rows.append(row)
            settings[candidate_id] = {
                "method": "baseline_preserving_support_calibration",
                "support_sources": list(supporting),
                "support_iou": float(support_iou),
                "support_bonus": float(support_bonus),
                "supplement_weight": float(supplement_weight),
                "supplement_min_support": int(supplement_min_support),
                "supplement_cluster_iou": float(
                    supplement_cluster_iou
                ),
                "thresholds": selected_thresholds,
                "dev_metrics": row,
            }
        metrics = pd.DataFrame(rows)
        minimum_gain = float(
            self.config["clear_improvement"]["minimum_f1_gain"]
        )
        maximum_map_drop = float(
            self.config["clear_improvement"]["maximum_map_drop"]
        )
        eligible = metrics[
            (metrics["f1_delta"] >= minimum_gain)
            & (metrics["f1_delta_ci_low"] > 0.0)
            & (
                metrics["ap"]
                >= metrics["baseline_ap"] - maximum_map_drop
            )
        ]
        ranking = eligible if not eligible.empty else metrics
        best_row = ranking.sort_values(
            ["f1", "ap", "f1_delta_ci_low"],
            ascending=[False, False, False],
        ).iloc[0]
        selected = settings[str(best_row["candidate_id"])]
        selected.update(
            {
                "selection_scope": "development_200",
                "development_gate_passed": not eligible.empty,
                "baseline_metrics": {
                    **baseline_summary,
                    "ap": float(baseline_ap["ap"]),
                    "ap50": float(baseline_ap["ap50"]),
                    "ap75": float(baseline_ap["ap75"]),
                },
                "success_criteria": {
                    "minimum_f1_gain": minimum_gain,
                    "minimum_f1_delta_ci_low": 0.0,
                    "maximum_map_drop": maximum_map_drop,
                },
            }
        )
        metrics.sort_values(
            ["f1", "ap"], ascending=[False, False]
        ).to_csv(
            self.results_dir / "support_development_metrics.csv",
            index=False,
        )
        (self.results_dir / "selected_support_improvement.json").write_text(
            json.dumps(selected, indent=2) + "\n", encoding="utf-8"
        )
        return selected

    def run_final_holdout(self) -> dict[str, Any]:
        selected_path = (
            self.results_dir / "selected_support_improvement.json"
        )
        selected_bytes = selected_path.read_bytes()
        selected = json.loads(selected_bytes.decode("utf-8"))
        if not selected.get("development_gate_passed", False):
            raise RuntimeError(
                "Support-calibration development gate did not pass"
            )
        manifest_path = resolve_path(
            self.config, self.config["paths"]["final_holdout_manifest"]
        )
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes.decode("utf-8"))
        image_ids = [int(item["image_id"]) for item in manifest["images"]]
        ground_truth = ground_truth_by_image(manifest)
        prompt_sets = self.config["clear_improvement"]["prompt_sets"]
        specifications = [
            {
                "name": "canonical_640_original",
                "prompts": list(prompt_sets["canonical"]),
                "image_size": 640,
                "horizontal_flip": False,
            },
            {
                "name": "canonical_768_original",
                "prompts": list(prompt_sets["canonical"]),
                "image_size": 768,
                "horizontal_flip": False,
            },
            {
                "name": "canonical_768_flip",
                "prompts": list(prompt_sets["canonical"]),
                "image_size": 768,
                "horizontal_flip": True,
            },
            {
                "name": "robust_category_768_original",
                "prompts": list(prompt_sets["robust_category"]),
                "image_size": 768,
                "horizontal_flip": False,
            },
            {
                "name": "robust_category_768_flip",
                "prompts": list(prompt_sets["robust_category"]),
                "image_size": 768,
                "horizontal_flip": True,
            },
        ]
        sources = self._collect_sources(
            "final_holdout", manifest, specifications
        )
        baseline_raw = sources["canonical_640_original"]
        supporting = {
            source: sources[source]
            for source in selected["support_sources"]
        }
        candidate_raw = build_support_calibrated_predictions(
            baseline_raw,
            supporting,
            image_ids,
            self.categories,
            support_iou=float(selected["support_iou"]),
            support_bonus=float(selected["support_bonus"]),
            supplement_weight=float(selected["supplement_weight"]),
            supplement_min_support=int(
                selected["supplement_min_support"]
            ),
            supplement_cluster_iou=float(
                selected["supplement_cluster_iou"]
            ),
        )
        baseline_fixed = {
            image_id: [
                prediction
                for prediction in baseline_raw[image_id]
                if float(prediction["confidence"])
                >= float(self.config["evaluation"]["confidence"])
            ]
            for image_id in image_ids
        }
        candidate_fixed = apply_category_thresholds(
            candidate_raw,
            image_ids,
            {
                category: float(value)
                for category, value in selected["thresholds"].items()
            },
        )
        baseline_summary, baseline_details = evaluate_records(
            baseline_fixed, ground_truth, image_ids, 0.0, self.match_iou
        )
        candidate_summary, candidate_details = evaluate_records(
            candidate_fixed, ground_truth, image_ids, 0.0, self.match_iou
        )
        baseline_ap, _, _ = evaluate_coco_detection(
            manifest,
            baseline_raw,
            image_ids,
            self.config["categories"],
            confidence_threshold=0.0,
        )
        candidate_ap, _, _ = evaluate_coco_detection(
            manifest,
            candidate_raw,
            image_ids,
            self.config["categories"],
            confidence_threshold=0.0,
        )
        deltas = paired_bootstrap_delta(
            baseline_details,
            candidate_details,
            self.bootstrap_samples,
            int(self.config["selection"]["final_holdout_seed"]) + 53,
        )
        f1_delta = next(row for row in deltas if row["metric"] == "f1")
        minimum_gain = float(
            self.config["clear_improvement"]["minimum_f1_gain"]
        )
        maximum_map_drop = float(
            self.config["clear_improvement"]["maximum_map_drop"]
        )
        passed = (
            float(f1_delta["estimate"]) >= minimum_gain
            and float(f1_delta["ci_low"]) > 0.0
            and float(candidate_ap["ap"])
            >= float(baseline_ap["ap"]) - maximum_map_drop
        )
        summary_rows = [
            {
                "scope": "final_holdout",
                "condition": "canonical_baseline",
                **baseline_summary,
                "ap": baseline_ap["ap"],
                "ap50": baseline_ap["ap50"],
                "ap75": baseline_ap["ap75"],
            },
            {
                "scope": "final_holdout",
                "condition": "support_calibrated_improvement",
                **candidate_summary,
                "ap": candidate_ap["ap"],
                "ap50": candidate_ap["ap50"],
                "ap75": candidate_ap["ap75"],
            },
        ]
        pd.DataFrame(summary_rows).to_csv(
            self.results_dir / "final_holdout_summary.csv", index=False
        )
        pd.DataFrame(deltas).to_csv(
            self.results_dir / "final_holdout_bootstrap_deltas.csv",
            index=False,
        )
        pd.DataFrame(
            [
                {"condition": "canonical_baseline", **row}
                for row in baseline_details
            ]
            + [
                {
                    "condition": "support_calibrated_improvement",
                    **row,
                }
                for row in candidate_details
            ]
        ).to_csv(
            self.results_dir / "final_holdout_per_image.csv", index=False
        )
        verdict = {
            "passed": passed,
            "selected_method": selected["method"],
            "baseline": summary_rows[0],
            "candidate": summary_rows[1],
            "f1_delta": f1_delta,
            "map_delta": float(candidate_ap["ap"] - baseline_ap["ap"]),
            "selection_was_frozen_before_holdout": True,
            "selected_parameters_sha256": hashlib.sha256(
                selected_bytes
            ).hexdigest(),
            "holdout_manifest_sha256": hashlib.sha256(
                manifest_bytes
            ).hexdigest(),
        }
        (self.results_dir / "final_holdout_verdict.json").write_text(
            json.dumps(verdict, indent=2) + "\n", encoding="utf-8"
        )
        _write_prediction_file(
            self.results_dir / "final_holdout_predictions.json",
            {
                "canonical_baseline": baseline_fixed,
                "support_calibrated_improvement": candidate_fixed,
            },
        )
        return verdict

    def run_model_scale_development(self) -> dict[str, Any]:
        manifest, small_sources = self._combined_development_data()
        image_ids = [int(item["image_id"]) for item in manifest["images"]]
        ground_truth = ground_truth_by_image(manifest)
        baseline_raw = small_sources["canonical_640_original"]
        baseline_fixed = {
            image_id: [
                prediction
                for prediction in baseline_raw[image_id]
                if float(prediction["confidence"])
                >= float(self.config["evaluation"]["confidence"])
            ]
            for image_id in image_ids
        }
        baseline_summary, baseline_details = evaluate_records(
            baseline_fixed, ground_truth, image_ids, 0.0, self.match_iou
        )
        baseline_ap, _, _ = evaluate_coco_detection(
            manifest,
            baseline_raw,
            image_ids,
            self.config["categories"],
            confidence_threshold=0.0,
        )
        threshold_candidates = [
            float(value)
            for value in self.config["clear_improvement"][
                "confidence_candidates"
            ]
        ]
        baseline_tuned_thresholds, _ = optimize_category_thresholds(
            baseline_raw,
            ground_truth,
            image_ids,
            self.categories,
            threshold_candidates,
            self.match_iou,
        )
        baseline_tuned_fixed = apply_category_thresholds(
            baseline_raw, image_ids, baseline_tuned_thresholds
        )
        baseline_tuned_summary, _ = evaluate_records(
            baseline_tuned_fixed,
            ground_truth,
            image_ids,
            0.0,
            self.match_iou,
        )
        model_name = str(
            self.config["clear_improvement"]["scale_model"]
        )
        image_sizes = [
            int(value)
            for value in self.config["clear_improvement"][
                "scale_image_sizes"
            ]
        ]
        prompts = list(
            self.config["clear_improvement"]["prompt_sets"]["canonical"]
        )
        specifications = [
            {
                "name": f"medium_{image_size}_original",
                "prompts": prompts,
                "image_size": image_size,
                "horizontal_flip": False,
            }
            for image_size in image_sizes
        ]
        medium_sources = self._collect_sources(
            "medium_development",
            manifest,
            specifications,
            model_name=model_name,
        )
        rows: list[dict[str, Any]] = []
        settings: dict[str, dict[str, Any]] = {}
        for specification in specifications:
            source = str(specification["name"])
            raw = medium_sources[source]
            selected_thresholds, _ = optimize_category_thresholds(
                raw,
                ground_truth,
                image_ids,
                self.categories,
                threshold_candidates,
                self.match_iou,
            )
            fixed = apply_category_thresholds(
                raw, image_ids, selected_thresholds
            )
            summary, details = evaluate_records(
                fixed, ground_truth, image_ids, 0.0, self.match_iou
            )
            ap, _, _ = evaluate_coco_detection(
                manifest,
                raw,
                image_ids,
                self.config["categories"],
                confidence_threshold=0.0,
            )
            f1_delta = next(
                row
                for row in paired_bootstrap_delta(
                    baseline_details,
                    details,
                    self.bootstrap_samples,
                    int(self.config["seed"]) + 67,
                )
                if row["metric"] == "f1"
            )
            row = {
                "candidate": source,
                "model": model_name,
                "image_size": int(specification["image_size"]),
                "thresholds": json.dumps(
                    selected_thresholds, sort_keys=True
                ),
                **summary,
                "ap": float(ap["ap"]),
                "ap50": float(ap["ap50"]),
                "ap75": float(ap["ap75"]),
                "baseline_ap": float(baseline_ap["ap"]),
                "f1_delta": float(f1_delta["estimate"]),
                "f1_delta_ci_low": float(f1_delta["ci_low"]),
                "f1_delta_ci_high": float(f1_delta["ci_high"]),
            }
            rows.append(row)
            settings[source] = {
                "method": "model_scale",
                "model": model_name,
                "source": source,
                "image_size": int(specification["image_size"]),
                "prompts": prompts,
                "thresholds": selected_thresholds,
                "dev_metrics": row,
            }
        metrics = pd.DataFrame(rows)
        minimum_gain = float(
            self.config["clear_improvement"]["minimum_f1_gain"]
        )
        maximum_map_drop = float(
            self.config["clear_improvement"]["maximum_map_drop"]
        )
        eligible = metrics[
            (metrics["f1_delta"] >= minimum_gain)
            & (metrics["f1_delta_ci_low"] > 0.0)
            & (
                metrics["ap"]
                >= metrics["baseline_ap"] - maximum_map_drop
            )
        ]
        ranking = eligible if not eligible.empty else metrics
        tolerance = float(
            self.config["clear_improvement"][
                "scale_simplicity_tolerance"
            ]
        )
        near_best = ranking[
            ranking["f1"] >= float(ranking["f1"].max()) - tolerance
        ]
        best_row = near_best.sort_values(
            ["image_size", "ap", "f1"],
            ascending=[True, False, False],
        ).iloc[0]
        selected = settings[str(best_row["candidate"])]
        selected.update(
            {
                "selection_scope": "development_200",
                "development_gate_passed": not eligible.empty,
                "baseline_metrics": {
                    **baseline_summary,
                    "ap": float(baseline_ap["ap"]),
                    "ap50": float(baseline_ap["ap50"]),
                    "ap75": float(baseline_ap["ap75"]),
                },
                "baseline_tuned_thresholds": baseline_tuned_thresholds,
                "baseline_tuned_metrics": baseline_tuned_summary,
                "success_criteria": {
                    "minimum_f1_gain": minimum_gain,
                    "minimum_f1_delta_ci_low": 0.0,
                    "maximum_map_drop": maximum_map_drop,
                    "near_best_f1_simplicity_tolerance": tolerance,
                },
            }
        )
        metrics.to_csv(
            self.results_dir / "model_scale_development_metrics.csv",
            index=False,
        )
        (self.results_dir / "selected_model_scale.json").write_text(
            json.dumps(selected, indent=2) + "\n", encoding="utf-8"
        )
        return selected

    def run_model_scale_final_holdout(self) -> dict[str, Any]:
        selected_path = self.results_dir / "selected_model_scale.json"
        selected_bytes = selected_path.read_bytes()
        selected = json.loads(selected_bytes.decode("utf-8"))
        if not selected.get("development_gate_passed", False):
            raise RuntimeError("Model-scale development gate did not pass")
        manifest_path = resolve_path(
            self.config, self.config["paths"]["final_holdout_manifest"]
        )
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes.decode("utf-8"))
        image_ids = [int(item["image_id"]) for item in manifest["images"]]
        ground_truth = ground_truth_by_image(manifest)
        specifications = [
            {
                "name": "canonical_640_original",
                "prompts": list(
                    self.config["clear_improvement"]["prompt_sets"][
                        "canonical"
                    ]
                ),
                "image_size": 640,
                "horizontal_flip": False,
            }
        ]
        small_sources = self._collect_sources(
            "model_final_small", manifest, specifications
        )
        medium_specification = [
            {
                "name": str(selected["source"]),
                "prompts": list(selected["prompts"]),
                "image_size": int(selected["image_size"]),
                "horizontal_flip": False,
            }
        ]
        medium_sources = self._collect_sources(
            "model_final_medium",
            manifest,
            medium_specification,
            model_name=str(selected["model"]),
        )
        baseline_raw = small_sources["canonical_640_original"]
        candidate_raw = medium_sources[str(selected["source"])]
        baseline_fixed = {
            image_id: [
                prediction
                for prediction in baseline_raw[image_id]
                if float(prediction["confidence"])
                >= float(self.config["evaluation"]["confidence"])
            ]
            for image_id in image_ids
        }
        candidate_fixed = apply_category_thresholds(
            candidate_raw,
            image_ids,
            {
                category: float(value)
                for category, value in selected["thresholds"].items()
            },
        )
        baseline_tuned_fixed = apply_category_thresholds(
            baseline_raw,
            image_ids,
            {
                category: float(value)
                for category, value in selected[
                    "baseline_tuned_thresholds"
                ].items()
            },
        )
        baseline_summary, baseline_details = evaluate_records(
            baseline_fixed, ground_truth, image_ids, 0.0, self.match_iou
        )
        baseline_tuned_summary, baseline_tuned_details = evaluate_records(
            baseline_tuned_fixed,
            ground_truth,
            image_ids,
            0.0,
            self.match_iou,
        )
        candidate_summary, candidate_details = evaluate_records(
            candidate_fixed, ground_truth, image_ids, 0.0, self.match_iou
        )
        baseline_ap, _, _ = evaluate_coco_detection(
            manifest,
            baseline_raw,
            image_ids,
            self.config["categories"],
            confidence_threshold=0.0,
        )
        candidate_ap, _, _ = evaluate_coco_detection(
            manifest,
            candidate_raw,
            image_ids,
            self.config["categories"],
            confidence_threshold=0.0,
        )
        deltas = paired_bootstrap_delta(
            baseline_details,
            candidate_details,
            self.bootstrap_samples,
            int(self.config["selection"]["final_holdout_seed"]) + 79,
        )
        tuned_deltas = paired_bootstrap_delta(
            baseline_tuned_details,
            candidate_details,
            self.bootstrap_samples,
            int(self.config["selection"]["final_holdout_seed"]) + 83,
        )
        f1_delta = next(row for row in deltas if row["metric"] == "f1")
        minimum_gain = float(
            self.config["clear_improvement"]["minimum_f1_gain"]
        )
        maximum_map_drop = float(
            self.config["clear_improvement"]["maximum_map_drop"]
        )
        passed = (
            float(f1_delta["estimate"]) >= minimum_gain
            and float(f1_delta["ci_low"]) > 0.0
            and float(candidate_ap["ap"])
            >= float(baseline_ap["ap"]) - maximum_map_drop
        )
        rows = [
            {
                "scope": "final_holdout",
                "condition": "small_canonical_baseline",
                **baseline_summary,
                "ap": baseline_ap["ap"],
                "ap50": baseline_ap["ap50"],
                "ap75": baseline_ap["ap75"],
            },
            {
                "scope": "final_holdout",
                "condition": "small_threshold_tuned_ablation",
                **baseline_tuned_summary,
                "ap": baseline_ap["ap"],
                "ap50": baseline_ap["ap50"],
                "ap75": baseline_ap["ap75"],
            },
            {
                "scope": "final_holdout",
                "condition": "medium_model_improvement",
                **candidate_summary,
                "ap": candidate_ap["ap"],
                "ap50": candidate_ap["ap50"],
                "ap75": candidate_ap["ap75"],
            },
        ]
        pd.DataFrame(rows).to_csv(
            self.results_dir / "model_final_holdout_summary.csv",
            index=False,
        )
        pd.DataFrame(
            [
                {"comparison": "medium_vs_small_fixed", **row}
                for row in deltas
            ]
            + [
                {"comparison": "medium_vs_small_tuned", **row}
                for row in tuned_deltas
            ]
        ).to_csv(
            self.results_dir / "model_final_holdout_bootstrap_deltas.csv",
            index=False,
        )
        pd.DataFrame(
            [
                {"condition": "small_canonical_baseline", **row}
                for row in baseline_details
            ]
            + [
                {"condition": "small_threshold_tuned_ablation", **row}
                for row in baseline_tuned_details
            ]
            + [
                {"condition": "medium_model_improvement", **row}
                for row in candidate_details
            ]
        ).to_csv(
            self.results_dir / "model_final_holdout_per_image.csv",
            index=False,
        )
        verdict = {
            "passed": passed,
            "selected_method": selected["method"],
            "baseline": rows[0],
            "small_threshold_tuned_ablation": rows[1],
            "candidate": rows[2],
            "f1_delta": f1_delta,
            "f1_delta_vs_small_tuned": next(
                row for row in tuned_deltas if row["metric"] == "f1"
            ),
            "map_delta": float(candidate_ap["ap"] - baseline_ap["ap"]),
            "selection_was_frozen_before_holdout": True,
            "selected_parameters_sha256": hashlib.sha256(
                selected_bytes
            ).hexdigest(),
            "holdout_manifest_sha256": hashlib.sha256(
                manifest_bytes
            ).hexdigest(),
        }
        (self.results_dir / "model_final_holdout_verdict.json").write_text(
            json.dumps(verdict, indent=2) + "\n", encoding="utf-8"
        )
        _write_prediction_file(
            self.results_dir / "model_final_holdout_predictions.json",
            {
                "small_canonical_baseline": baseline_fixed,
                "small_threshold_tuned_ablation": baseline_tuned_fixed,
                "medium_model_improvement": candidate_fixed,
            },
        )
        return verdict
