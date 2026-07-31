from __future__ import annotations

import json
from itertools import combinations, product
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from PIL import Image

from .coco import ground_truth_by_image, load_manifest
from .config import category_names, resolve_path
from .metrics import box_iou, canonical_nms, evaluate_records
from .model import YoloWorldPredictor


BASE_VARIANTS = {
    "canonical": ("prompt_canonical", 0),
    "synonym": ("prompt_synonym", 1),
    "hypernym": ("prompt_hypernym", 2),
}


def _read_prediction_file(path: Path) -> dict[str, dict[int, list[dict[str, Any]]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        condition: {
            int(image_id): predictions
            for image_id, predictions in by_image.items()
        }
        for condition, by_image in payload.items()
    }


def _write_prediction_file(
    path: Path,
    records: dict[str, dict[int, list[dict[str, Any]]]],
) -> None:
    payload = {
        condition: {
            str(image_id): predictions
            for image_id, predictions in by_image.items()
        }
        for condition, by_image in records.items()
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _metric_sort_key(
    row: dict[str, Any],
    *,
    complexity: float = 0.0,
    threshold: float = 0.25,
) -> tuple[float, ...]:
    return (
        float(row["f1"]),
        float(row["recall"]),
        float(row["precision"]),
        -float(complexity),
        -abs(float(threshold) - 0.25),
    )


def _nonempty_subsets(values: list[str]) -> Iterable[tuple[str, ...]]:
    for size in range(1, len(values) + 1):
        yield from combinations(values, size)


class PromptStudy:
    """Validation-selected prompt variants and fusion strategies."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.manifest = load_manifest(config)
        self.categories = category_names(config)
        self.gt_by_image = ground_truth_by_image(self.manifest)
        self.images_dir = resolve_path(config, config["paths"]["coco_images"])
        self.results_dir = resolve_path(config, config["paths"]["results_dir"])
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.entries = self.manifest["images"]
        self.validation_ids = [
            int(item["image_id"])
            for item in self.entries
            if item["split"] == "validation"
        ]
        self.test_ids = [
            int(item["image_id"])
            for item in self.entries
            if item["split"] == "test"
        ]
        self.all_ids = self.validation_ids + self.test_ids

    def prompt_variants(self) -> dict[str, list[str]]:
        variants = {
            name: [
                str(details["prompts"][prompt_index])
                for details in self.config["categories"].values()
            ]
            for name, (_, prompt_index) in BASE_VARIANTS.items()
        }
        improvement = self.config["prompt_improvement"]["variants"]
        for variant_name, specification in improvement.items():
            if isinstance(specification, str):
                variants[variant_name] = [
                    specification.format(
                        category=category,
                        article=(
                            "an"
                            if category[0].lower() in {"a", "e", "i", "o", "u"}
                            else "a"
                        ),
                    )
                    for category in self.categories
                ]
            else:
                variants[variant_name] = [
                    str(specification[category]) for category in self.categories
                ]
        return variants

    def collect_predictions(
        self,
    ) -> dict[str, dict[int, list[dict[str, Any]]]]:
        raw_path = self.results_dir / "raw_predictions.json"
        if not raw_path.exists():
            raise FileNotFoundError(
                "raw_predictions.json is required; run the full experiment first"
            )
        raw = _read_prediction_file(raw_path)
        variants = self.prompt_variants()
        cache_path = self.results_dir / "prompt_predictions.json"
        metadata_path = self.results_dir / "prompt_prediction_metadata.json"
        cached: dict[str, dict[int, list[dict[str, Any]]]] = {}
        cached_prompts: dict[str, list[str]] = {}
        if cache_path.exists() and metadata_path.exists():
            cached = _read_prediction_file(cache_path)
            cached_prompts = json.loads(
                metadata_path.read_text(encoding="utf-8")
            ).get("prompts", {})

        records: dict[str, dict[int, list[dict[str, Any]]]] = {}
        for variant, (condition, _) in BASE_VARIANTS.items():
            records[variant] = {
                image_id: [
                    {**prediction, "prompt_variant": variant}
                    for prediction in raw[condition].get(image_id, [])
                ]
                for image_id in self.all_ids
            }

        missing = [
            variant
            for variant in variants
            if variant not in BASE_VARIANTS
            and (
                variant not in cached
                or cached_prompts.get(variant) != variants[variant]
                or set(cached[variant]) != set(self.all_ids)
            )
        ]
        for variant in variants:
            if variant not in missing and variant not in records:
                records[variant] = cached[variant]

        if missing:
            predictor = YoloWorldPredictor(self.config, self.categories)
            for variant in missing:
                print(f"[prompt-study] inference: {variant} ({len(self.entries)} images)")
                predictor.set_prompts(variants[variant])
                by_image: dict[int, list[dict[str, Any]]] = {}
                for entry in self.entries:
                    image_id = int(entry["image_id"])
                    image = Image.open(
                        self.images_dir / entry["file_name"]
                    ).convert("RGB")
                    by_image[image_id] = [
                        {**prediction, "prompt_variant": variant}
                        for prediction in predictor.predict(image)
                    ]
                records[variant] = by_image

        _write_prediction_file(cache_path, records)
        metadata_path.write_text(
            json.dumps(
                {
                    "prompts": variants,
                    "image_ids": self.all_ids,
                    "model": self.config["model"]["name"],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return records

    def _category_view(
        self,
        records: dict[int, list[dict[str, Any]]],
        image_ids: list[int],
        category: str,
    ) -> tuple[
        dict[int, list[dict[str, Any]]],
        dict[int, list[dict[str, Any]]],
    ]:
        predictions = {
            image_id: [
                prediction
                for prediction in records.get(image_id, [])
                if prediction["category"] == category
            ]
            for image_id in image_ids
        }
        ground_truth = {
            image_id: [
                annotation
                for annotation in self.gt_by_image.get(image_id, [])
                if annotation["category"] == category
            ]
            for image_id in image_ids
        }
        return predictions, ground_truth

    def _evaluate_category(
        self,
        records: dict[int, list[dict[str, Any]]],
        image_ids: list[int],
        category: str,
        confidence: float,
    ) -> dict[str, Any]:
        predictions, ground_truth = self._category_view(
            records, image_ids, category
        )
        summary, _ = evaluate_records(
            predictions,
            ground_truth,
            image_ids,
            confidence,
            float(self.config["evaluation"]["match_iou"]),
        )
        return summary

    def _variant_metrics(
        self,
        records: dict[str, dict[int, list[dict[str, Any]]]],
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        fixed_confidence = float(self.config["evaluation"]["confidence"])
        summary_rows: list[dict[str, Any]] = []
        category_rows: list[dict[str, Any]] = []
        tuning_rows: list[dict[str, Any]] = []
        for scope, image_ids in (
            ("validation", self.validation_ids),
            ("test", self.test_ids),
        ):
            for variant, by_image in records.items():
                summary, _ = evaluate_records(
                    by_image,
                    self.gt_by_image,
                    image_ids,
                    fixed_confidence,
                    float(self.config["evaluation"]["match_iou"]),
                )
                summary_rows.append(
                    {
                        "scope": scope,
                        "variant": variant,
                        "confidence_threshold": fixed_confidence,
                        **summary,
                    }
                )
                for category in self.categories:
                    category_rows.append(
                        {
                            "scope": scope,
                            "variant": variant,
                            "category": category,
                            "confidence_threshold": fixed_confidence,
                            **self._evaluate_category(
                                by_image,
                                image_ids,
                                category,
                                fixed_confidence,
                            ),
                        }
                    )

        for category, variant, confidence in product(
            self.categories,
            records,
            self.config["evaluation"]["tune_confidences"],
        ):
            tuning_rows.append(
                {
                    "category": category,
                    "variant": variant,
                    "confidence_threshold": float(confidence),
                    **self._evaluate_category(
                        records[variant],
                        self.validation_ids,
                        category,
                        float(confidence),
                    ),
                }
            )
        return (
            pd.DataFrame(summary_rows),
            pd.DataFrame(category_rows),
            pd.DataFrame(tuning_rows),
        )

    def _build_category_selected(
        self,
        records: dict[str, dict[int, list[dict[str, Any]]]],
        image_ids: list[int],
        selection: dict[str, dict[str, Any]],
    ) -> dict[int, list[dict[str, Any]]]:
        output = {image_id: [] for image_id in image_ids}
        for category, parameters in selection.items():
            variant = str(parameters["variant"])
            confidence = float(parameters["confidence_threshold"])
            for image_id in image_ids:
                output[image_id].extend(
                    {
                        **prediction,
                        "strategy_source": variant,
                    }
                    for prediction in records[variant].get(image_id, [])
                    if prediction["category"] == category
                    and float(prediction["confidence"]) >= confidence
                )
        for predictions in output.values():
            predictions.sort(
                key=lambda item: float(item["confidence"]), reverse=True
            )
        return output

    def _tune_naive_ensemble(
        self,
        records: dict[str, dict[int, list[dict[str, Any]]]],
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        variants = list(BASE_VARIANTS)
        rows = []
        best: dict[str, Any] | None = None
        for confidence, nms_iou in product(
            self.config["evaluation"]["tune_confidences"],
            self.config["evaluation"]["tune_nms_ious"],
        ):
            output = self._build_subset_nms(
                records,
                self.validation_ids,
                {category: variants for category in self.categories},
                {category: float(confidence) for category in self.categories},
                {category: float(nms_iou) for category in self.categories},
            )
            summary, _ = evaluate_records(
                output,
                self.gt_by_image,
                self.validation_ids,
                0.0,
                float(self.config["evaluation"]["match_iou"]),
            )
            row = {
                "strategy": "naive_three_prompt_nms",
                "category": "__all__",
                "variants": "|".join(variants),
                "variant_count": len(variants),
                "confidence_threshold": float(confidence),
                "nms_iou": float(nms_iou),
                **summary,
            }
            rows.append(row)
            if best is None or _metric_sort_key(
                row,
                complexity=len(variants),
                threshold=float(confidence),
            ) > _metric_sort_key(
                best,
                complexity=len(variants),
                threshold=float(best["confidence_threshold"]),
            ):
                best = row
        assert best is not None
        return (
            {
                "variants": variants,
                "confidence_threshold": float(best["confidence_threshold"]),
                "nms_iou": float(best["nms_iou"]),
            },
            rows,
        )

    def _build_subset_nms(
        self,
        records: dict[str, dict[int, list[dict[str, Any]]]],
        image_ids: list[int],
        variants_by_category: dict[str, list[str]],
        confidence_by_category: dict[str, float],
        nms_by_category: dict[str, float],
    ) -> dict[int, list[dict[str, Any]]]:
        output = {image_id: [] for image_id in image_ids}
        for image_id in image_ids:
            for category in self.categories:
                merged = []
                for variant in variants_by_category[category]:
                    merged.extend(
                        {
                            **prediction,
                            "strategy_source": variant,
                        }
                        for prediction in records[variant].get(image_id, [])
                        if prediction["category"] == category
                    )
                output[image_id].extend(
                    canonical_nms(
                        merged,
                        confidence_by_category[category],
                        nms_by_category[category],
                    )
                )
            output[image_id].sort(
                key=lambda item: float(item["confidence"]), reverse=True
            )
        return output

    def _tune_subset_ensemble(
        self,
        records: dict[str, dict[int, list[dict[str, Any]]]],
    ) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
        variants = list(records)
        rows: list[dict[str, Any]] = []
        selected: dict[str, dict[str, Any]] = {}
        for category in self.categories:
            best: dict[str, Any] | None = None
            for subset, confidence, nms_iou in product(
                _nonempty_subsets(variants),
                self.config["evaluation"]["tune_confidences"],
                self.config["evaluation"]["tune_nms_ious"],
            ):
                output = self._build_subset_nms(
                    records,
                    self.validation_ids,
                    {name: list(subset) if name == category else [] for name in self.categories},
                    {name: float(confidence) for name in self.categories},
                    {name: float(nms_iou) for name in self.categories},
                )
                summary = self._evaluate_category(
                    output,
                    self.validation_ids,
                    category,
                    0.0,
                )
                row = {
                    "strategy": "category_subset_nms",
                    "category": category,
                    "variants": "|".join(subset),
                    "variant_count": len(subset),
                    "confidence_threshold": float(confidence),
                    "nms_iou": float(nms_iou),
                    **summary,
                }
                rows.append(row)
                if best is None or _metric_sort_key(
                    row,
                    complexity=len(subset),
                    threshold=float(confidence),
                ) > _metric_sort_key(
                    best,
                    complexity=int(best["variant_count"]),
                    threshold=float(best["confidence_threshold"]),
                ):
                    best = row
            assert best is not None
            selected[category] = {
                "variants": str(best["variants"]).split("|"),
                "confidence_threshold": float(best["confidence_threshold"]),
                "nms_iou": float(best["nms_iou"]),
                "validation_f1": float(best["f1"]),
            }
        return selected, rows

    @staticmethod
    def _fuse_cluster(
        cluster: list[dict[str, Any]],
        reliability: dict[str, float],
        agreement_bonus: float,
    ) -> dict[str, Any]:
        best_by_source: dict[str, dict[str, Any]] = {}
        for prediction in cluster:
            source = str(prediction["prompt_variant"])
            current = best_by_source.get(source)
            if current is None or float(prediction["confidence"]) > float(
                current["confidence"]
            ):
                best_by_source[source] = prediction
        selected = list(best_by_source.values())
        coordinate_weights = np.asarray(
            [
                float(item["confidence"])
                * max(float(reliability[str(item["prompt_variant"])]), 1e-6)
                for item in selected
            ],
            dtype=float,
        )
        boxes = np.asarray(
            [item["bbox_xyxy"] for item in selected], dtype=float
        )
        fused_box = np.average(boxes, axis=0, weights=coordinate_weights)
        reliability_weights = np.asarray(
            [
                max(float(reliability[str(item["prompt_variant"])]), 1e-6)
                for item in selected
            ],
            dtype=float,
        )
        base_confidence = float(
            np.average(
                np.asarray(
                    [float(item["confidence"]) for item in selected],
                    dtype=float,
                ),
                weights=reliability_weights,
            )
        )
        support = len(selected)
        confidence = min(
            1.0,
            base_confidence + float(agreement_bonus) * max(0, support - 1),
        )
        representative = max(
            selected, key=lambda item: float(item["confidence"])
        )
        return {
            **representative,
            "bbox_xyxy": [float(value) for value in fused_box],
            "confidence": confidence,
            "original_confidence": float(representative["confidence"]),
            "fusion_support": support,
            "fusion_sources": sorted(best_by_source),
            "strategy_source": "weighted_box_fusion",
        }

    def _weighted_fusion_for_category(
        self,
        records: dict[str, dict[int, list[dict[str, Any]]]],
        image_ids: list[int],
        category: str,
        reliability: dict[str, float],
        cluster_iou: float,
        agreement_bonus: float,
        confidence_threshold: float,
    ) -> dict[int, list[dict[str, Any]]]:
        output: dict[int, list[dict[str, Any]]] = {}
        for image_id in image_ids:
            candidates = [
                prediction
                for variant, by_image in records.items()
                for prediction in by_image.get(image_id, [])
                if prediction["category"] == category
                and float(prediction["confidence"])
                * float(reliability[variant])
                >= float(self.config["model"]["raw_confidence"])
            ]
            candidates.sort(
                key=lambda item: float(item["confidence"])
                * float(reliability[str(item["prompt_variant"])]),
                reverse=True,
            )
            clusters: list[list[dict[str, Any]]] = []
            fused_boxes: list[list[float]] = []
            for prediction in candidates:
                overlaps = [
                    box_iou(prediction["bbox_xyxy"], fused_box)
                    for fused_box in fused_boxes
                ]
                best_index = (
                    int(np.argmax(overlaps)) if overlaps else -1
                )
                if best_index >= 0 and overlaps[best_index] >= cluster_iou:
                    clusters[best_index].append(prediction)
                    fused_boxes[best_index] = self._fuse_cluster(
                        clusters[best_index], reliability, agreement_bonus
                    )["bbox_xyxy"]
                else:
                    clusters.append([prediction])
                    fused_boxes.append(
                        [float(value) for value in prediction["bbox_xyxy"]]
                    )
            fused = [
                self._fuse_cluster(cluster, reliability, agreement_bonus)
                for cluster in clusters
            ]
            output[image_id] = [
                prediction
                for prediction in fused
                if float(prediction["confidence"]) >= confidence_threshold
            ]
            output[image_id].sort(
                key=lambda item: float(item["confidence"]), reverse=True
            )
        return output

    def _tune_weighted_fusion(
        self,
        records: dict[str, dict[int, list[dict[str, Any]]]],
        variant_tuning: pd.DataFrame,
    ) -> tuple[
        dict[str, dict[str, Any]],
        dict[str, dict[str, float]],
        list[dict[str, Any]],
    ]:
        reliability: dict[str, dict[str, float]] = {}
        for category in self.categories:
            category_tuning = variant_tuning[
                variant_tuning["category"] == category
            ]
            best_by_variant = (
                category_tuning.sort_values(
                    ["f1", "recall", "precision"],
                    ascending=[False, False, False],
                )
                .groupby("variant", sort=False)
                .first()["f1"]
                .to_dict()
            )
            maximum = max(float(value) for value in best_by_variant.values())
            reliability[category] = {
                variant: (
                    max(0.05, float(best_by_variant[variant]) / maximum)
                    if maximum > 0
                    else (1.0 if variant == "canonical" else 0.05)
                )
                for variant in records
            }

        selected: dict[str, dict[str, Any]] = {}
        rows: list[dict[str, Any]] = []
        for category in self.categories:
            best: dict[str, Any] | None = None
            for cluster_iou, confidence, bonus in product(
                self.config["evaluation"]["tune_nms_ious"],
                self.config["evaluation"]["tune_confidences"],
                self.config["prompt_improvement"]["agreement_bonus_candidates"],
            ):
                output = self._weighted_fusion_for_category(
                    records,
                    self.validation_ids,
                    category,
                    reliability[category],
                    float(cluster_iou),
                    float(bonus),
                    float(confidence),
                )
                summary = self._evaluate_category(
                    output,
                    self.validation_ids,
                    category,
                    0.0,
                )
                row = {
                    "strategy": "reliability_weighted_fusion",
                    "category": category,
                    "variants": "|".join(records),
                    "variant_count": len(records),
                    "confidence_threshold": float(confidence),
                    "nms_iou": float(cluster_iou),
                    "agreement_bonus": float(bonus),
                    **summary,
                }
                rows.append(row)
                if best is None or _metric_sort_key(
                    row,
                    complexity=len(records),
                    threshold=float(confidence),
                ) > _metric_sort_key(
                    best,
                    complexity=len(records),
                    threshold=float(best["confidence_threshold"]),
                ):
                    best = row
            assert best is not None
            selected[category] = {
                "cluster_iou": float(best["nms_iou"]),
                "confidence_threshold": float(best["confidence_threshold"]),
                "agreement_bonus": float(best["agreement_bonus"]),
                "validation_f1": float(best["f1"]),
            }
        return selected, reliability, rows

    def _build_weighted_fusion(
        self,
        records: dict[str, dict[int, list[dict[str, Any]]]],
        image_ids: list[int],
        selected: dict[str, dict[str, Any]],
        reliability: dict[str, dict[str, float]],
    ) -> dict[int, list[dict[str, Any]]]:
        output = {image_id: [] for image_id in image_ids}
        for category in self.categories:
            params = selected[category]
            category_output = self._weighted_fusion_for_category(
                records,
                image_ids,
                category,
                reliability[category],
                float(params["cluster_iou"]),
                float(params["agreement_bonus"]),
                float(params["confidence_threshold"]),
            )
            for image_id in image_ids:
                output[image_id].extend(category_output[image_id])
                output[image_id].sort(
                    key=lambda item: float(item["confidence"]), reverse=True
                )
        return output

    def _evaluate_strategies(
        self,
        strategies: dict[str, dict[int, list[dict[str, Any]]]],
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        summary_rows = []
        category_rows = []
        detail_rows = []
        for scope, image_ids in (
            ("validation", self.validation_ids),
            ("test", self.test_ids),
        ):
            for strategy, by_image in strategies.items():
                summary, details = evaluate_records(
                    by_image,
                    self.gt_by_image,
                    image_ids,
                    0.0,
                    float(self.config["evaluation"]["match_iou"]),
                )
                summary_rows.append(
                    {"scope": scope, "strategy": strategy, **summary}
                )
                detail_rows.extend(
                    {
                        "scope": scope,
                        "strategy": strategy,
                        **detail,
                    }
                    for detail in details
                )
                for category in self.categories:
                    category_rows.append(
                        {
                            "scope": scope,
                            "strategy": strategy,
                            "category": category,
                            **self._evaluate_category(
                                by_image, image_ids, category, 0.0
                            ),
                        }
                    )
        return (
            pd.DataFrame(summary_rows),
            pd.DataFrame(category_rows),
            pd.DataFrame(detail_rows),
        )

    def run(self) -> dict[str, Any]:
        records = self.collect_predictions()
        variant_metrics, variant_categories, variant_tuning = (
            self._variant_metrics(records)
        )
        fixed_confidence = float(self.config["evaluation"]["confidence"])

        fixed_selection: dict[str, dict[str, Any]] = {}
        calibrated_selection: dict[str, dict[str, Any]] = {}
        for category in self.categories:
            fixed_rows = variant_categories[
                (variant_categories["scope"] == "validation")
                & (variant_categories["category"] == category)
            ]
            fixed_best = max(
                fixed_rows.to_dict("records"),
                key=lambda row: _metric_sort_key(
                    row, threshold=fixed_confidence
                ),
            )
            fixed_selection[category] = {
                "variant": str(fixed_best["variant"]),
                "confidence_threshold": fixed_confidence,
                "validation_f1": float(fixed_best["f1"]),
            }

            tuning_rows = variant_tuning[
                variant_tuning["category"] == category
            ].to_dict("records")
            tuned_best = max(
                tuning_rows,
                key=lambda row: _metric_sort_key(
                    row,
                    threshold=float(row["confidence_threshold"]),
                ),
            )
            calibrated_selection[category] = {
                "variant": str(tuned_best["variant"]),
                "confidence_threshold": float(
                    tuned_best["confidence_threshold"]
                ),
                "validation_f1": float(tuned_best["f1"]),
            }

        naive_params, naive_rows = self._tune_naive_ensemble(records)
        subset_params, subset_rows = self._tune_subset_ensemble(records)
        fusion_params, reliability, fusion_rows = self._tune_weighted_fusion(
            records, variant_tuning
        )

        strategies: dict[str, dict[int, list[dict[str, Any]]]] = {
            "canonical_baseline": self._build_category_selected(
                records,
                self.all_ids,
                {
                    category: {
                        "variant": "canonical",
                        "confidence_threshold": fixed_confidence,
                    }
                    for category in self.categories
                },
            ),
            "naive_three_prompt_nms": self._build_subset_nms(
                records,
                self.all_ids,
                {
                    category: list(naive_params["variants"])
                    for category in self.categories
                },
                {
                    category: float(naive_params["confidence_threshold"])
                    for category in self.categories
                },
                {
                    category: float(naive_params["nms_iou"])
                    for category in self.categories
                },
            ),
            "category_best_fixed": self._build_category_selected(
                records, self.all_ids, fixed_selection
            ),
            "category_best_calibrated": self._build_category_selected(
                records, self.all_ids, calibrated_selection
            ),
            "category_subset_nms": self._build_subset_nms(
                records,
                self.all_ids,
                {
                    category: list(subset_params[category]["variants"])
                    for category in self.categories
                },
                {
                    category: float(
                        subset_params[category]["confidence_threshold"]
                    )
                    for category in self.categories
                },
                {
                    category: float(subset_params[category]["nms_iou"])
                    for category in self.categories
                },
            ),
            "reliability_weighted_fusion": self._build_weighted_fusion(
                records,
                self.all_ids,
                fusion_params,
                reliability,
            ),
        }
        raw_confidence = float(self.config["model"]["raw_confidence"])
        strategy_ap_candidates: dict[
            str, dict[int, list[dict[str, Any]]]
        ] = {
            "canonical_baseline": self._build_category_selected(
                records,
                self.all_ids,
                {
                    category: {
                        "variant": "canonical",
                        "confidence_threshold": raw_confidence,
                    }
                    for category in self.categories
                },
            ),
            "naive_three_prompt_nms": self._build_subset_nms(
                records,
                self.all_ids,
                {
                    category: list(naive_params["variants"])
                    for category in self.categories
                },
                {category: raw_confidence for category in self.categories},
                {
                    category: float(naive_params["nms_iou"])
                    for category in self.categories
                },
            ),
            "category_best_fixed": self._build_category_selected(
                records,
                self.all_ids,
                {
                    category: {
                        **fixed_selection[category],
                        "confidence_threshold": raw_confidence,
                    }
                    for category in self.categories
                },
            ),
            "category_best_calibrated": self._build_category_selected(
                records,
                self.all_ids,
                {
                    category: {
                        **calibrated_selection[category],
                        "confidence_threshold": raw_confidence,
                    }
                    for category in self.categories
                },
            ),
            "category_subset_nms": self._build_subset_nms(
                records,
                self.all_ids,
                {
                    category: list(subset_params[category]["variants"])
                    for category in self.categories
                },
                {category: raw_confidence for category in self.categories},
                {
                    category: float(subset_params[category]["nms_iou"])
                    for category in self.categories
                },
            ),
            "reliability_weighted_fusion": self._build_weighted_fusion(
                records,
                self.all_ids,
                {
                    category: {
                        **fusion_params[category],
                        "confidence_threshold": raw_confidence,
                    }
                    for category in self.categories
                },
                reliability,
            ),
        }
        strategy_metrics, strategy_categories, strategy_details = (
            self._evaluate_strategies(strategies)
        )
        validation_rows = strategy_metrics[
            strategy_metrics["scope"] == "validation"
        ].to_dict("records")
        primary = max(
            validation_rows,
            key=lambda row: _metric_sort_key(
                row,
                complexity=(
                    0.0 if row["strategy"] == "canonical_baseline" else 1.0
                ),
                threshold=fixed_confidence,
            ),
        )

        variant_metrics.to_csv(
            self.results_dir / "prompt_variant_metrics.csv", index=False
        )
        variant_categories.to_csv(
            self.results_dir / "prompt_variant_by_category.csv", index=False
        )
        variant_tuning.to_csv(
            self.results_dir / "prompt_variant_threshold_tuning.csv",
            index=False,
        )
        pd.DataFrame(naive_rows + subset_rows + fusion_rows).to_csv(
            self.results_dir / "prompt_strategy_tuning.csv", index=False
        )
        strategy_metrics.to_csv(
            self.results_dir / "prompt_strategy_metrics.csv", index=False
        )
        strategy_categories.to_csv(
            self.results_dir / "prompt_strategy_by_category.csv", index=False
        )
        strategy_details.to_csv(
            self.results_dir / "prompt_strategy_per_image.csv", index=False
        )
        _write_prediction_file(
            self.results_dir / "prompt_strategy_predictions.json",
            strategies,
        )
        _write_prediction_file(
            self.results_dir / "prompt_strategy_ap_predictions.json",
            strategy_ap_candidates,
        )
        parameters = {
            "selection_scope": "validation",
            "fixed_test_scope": "test",
            "prompt_variants": self.prompt_variants(),
            "fixed_category_selection": fixed_selection,
            "calibrated_category_selection": calibrated_selection,
            "naive_three_prompt_nms": naive_params,
            "category_subset_nms": subset_params,
            "reliability_weighted_fusion": {
                "parameters": fusion_params,
                "reliability": reliability,
            },
            "primary_strategy": str(primary["strategy"]),
            "primary_validation_f1": float(primary["f1"]),
        }
        (self.results_dir / "prompt_selected_parameters.json").write_text(
            json.dumps(parameters, indent=2) + "\n",
            encoding="utf-8",
        )
        return parameters
