from __future__ import annotations

import json
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

from .coco import ground_truth_by_image, load_manifest
from .config import category_names, prompt_set, resolve_path
from .metrics import canonical_nms, evaluate_records
from .model import YoloWorldPredictor, apply_corruption, apply_gamma_correction


PROMPT_CONDITIONS = {
    "prompt_canonical": 0,
    "prompt_synonym": 1,
    "prompt_hypernym": 2,
}


class ExperimentRunner:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.manifest = load_manifest(config)
        self.categories = category_names(config)
        self.gt_by_image = ground_truth_by_image(self.manifest)
        self.images_dir = resolve_path(config, config["paths"]["coco_images"])
        self.results_dir = resolve_path(config, config["paths"]["results_dir"])
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.predictor = YoloWorldPredictor(config, self.categories)

    def _entries_for_scope(self, scope: str) -> list[dict[str, Any]]:
        if scope == "pilot":
            return [item for item in self.manifest["images"] if item["pilot"]]
        if scope == "validation":
            return [
                item for item in self.manifest["images"]
                if item["split"] == "validation"
            ]
        if scope == "test":
            return [
                item for item in self.manifest["images"]
                if item["split"] == "test"
            ]
        if scope == "all":
            return self.manifest["images"]
        raise ValueError(f"Unknown scope: {scope}")

    def _predict_corruptions(
        self,
        entries: list[dict[str, Any]],
    ) -> dict[str, dict[int, list[dict[str, Any]]]]:
        records: dict[str, dict[int, list[dict[str, Any]]]] = {}
        self.predictor.set_prompts(prompt_set(self.config, 0))
        for corruption in self.config["corruptions"]:
            condition = corruption["name"]
            records[condition] = {}
            for entry in entries:
                image = Image.open(self.images_dir / entry["file_name"]).convert("RGB")
                transformed = apply_corruption(
                    image,
                    corruption["type"],
                    float(corruption["value"]),
                )
                records[condition][int(entry["image_id"])] = self.predictor.predict(
                    transformed
                )
        return records

    def _predict_prompts(
        self,
        entries: list[dict[str, Any]],
        canonical_records: dict[int, list[dict[str, Any]]],
    ) -> dict[str, dict[int, list[dict[str, Any]]]]:
        records = {"prompt_canonical": canonical_records}
        for condition, prompt_index in PROMPT_CONDITIONS.items():
            if condition == "prompt_canonical":
                continue
            self.predictor.set_prompts(prompt_set(self.config, prompt_index))
            records[condition] = {}
            for entry in entries:
                image = Image.open(self.images_dir / entry["file_name"]).convert("RGB")
                records[condition][int(entry["image_id"])] = self.predictor.predict(
                    image
                )
        return records

    def _predict_gamma_corrections(
        self,
        entries: list[dict[str, Any]],
    ) -> dict[str, dict[int, list[dict[str, Any]]]]:
        records: dict[str, dict[int, list[dict[str, Any]]]] = {}
        self.predictor.set_prompts(prompt_set(self.config, 0))
        brightness = float(self.config["improvement"]["lowlight_brightness"])
        for gamma in self.config["improvement"]["gamma_candidates"]:
            condition = f"lowlight_gamma_{float(gamma):.2f}"
            records[condition] = {}
            for entry in entries:
                image = Image.open(self.images_dir / entry["file_name"]).convert("RGB")
                dark = apply_corruption(image, "brightness", brightness)
                corrected = apply_gamma_correction(dark, float(gamma))
                records[condition][int(entry["image_id"])] = self.predictor.predict(
                    corrected
                )
        return records

    def collect_predictions(
        self,
        scope: str,
        include_gamma: bool = False,
    ) -> dict[str, dict[int, list[dict[str, Any]]]]:
        entries = self._entries_for_scope(scope)
        records = self._predict_corruptions(entries)
        records.update(self._predict_prompts(entries, records["original"]))
        if include_gamma:
            records.update(self._predict_gamma_corrections(entries))
        return records

    def _evaluate_conditions(
        self,
        records: dict[str, dict[int, list[dict[str, Any]]]],
        image_ids: list[int],
        scope: str,
        confidence: float,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        summary_rows: list[dict[str, Any]] = []
        detail_rows: list[dict[str, Any]] = []
        corruption_names = {item["name"] for item in self.config["corruptions"]}
        for condition, condition_records in records.items():
            summary, details = evaluate_records(
                condition_records,
                self.gt_by_image,
                image_ids,
                confidence,
                float(self.config["evaluation"]["match_iou"]),
            )
            if condition in corruption_names:
                family = "corruption"
            elif condition.startswith("lowlight_gamma_"):
                family = "preprocessing_candidate"
            else:
                family = "prompt"
            summary_rows.append(
                {
                    "scope": scope,
                    "family": family,
                    "condition": condition,
                    "confidence_threshold": confidence,
                    **summary,
                }
            )
            for detail in details:
                detail_rows.append(
                    {
                        "scope": scope,
                        "family": family,
                        "condition": condition,
                        "confidence_threshold": confidence,
                        **detail,
                    }
                )
        return summary_rows, detail_rows

    def run_pilot(self) -> dict[str, Any]:
        entries = self._entries_for_scope("pilot")
        image_ids = [int(item["image_id"]) for item in entries]
        records = self.collect_predictions("pilot")
        confidence = float(self.config["evaluation"]["confidence"])
        summaries, details = self._evaluate_conditions(
            records, image_ids, "pilot", confidence
        )
        by_condition = {row["condition"]: row for row in summaries}
        baseline = by_condition["original"]
        compared = [
            row for row in summaries
            if row["condition"] in {
                "brightness_0.25",
                "blur_sigma_4",
                "prompt_synonym",
                "prompt_hypernym",
            }
        ]
        max_f1_change = max(
            abs(float(row["f1"]) - float(baseline["f1"])) for row in compared
        )
        gate = self.config["pilot_gate"]
        continue_yoloworld = (
            float(baseline["recall"]) >= float(gate["minimum_baseline_recall"])
            and max_f1_change >= float(gate["minimum_absolute_f1_change"])
        )
        decision = {
            "continue_yoloworld": continue_yoloworld,
            "baseline_recall": float(baseline["recall"]),
            "baseline_f1": float(baseline["f1"]),
            "max_absolute_f1_change": max_f1_change,
            "required_baseline_recall": float(gate["minimum_baseline_recall"]),
            "required_absolute_f1_change": float(
                gate["minimum_absolute_f1_change"]
            ),
        }
        pd.DataFrame(summaries).to_csv(
            self.results_dir / "pilot_summary.csv", index=False
        )
        pd.DataFrame(details).to_csv(
            self.results_dir / "pilot_per_image.csv", index=False
        )
        with (self.results_dir / "pilot_decision.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(decision, handle, indent=2)
        self._save_predictions(records, self.results_dir / "pilot_predictions.json")
        return decision

    def _build_ensemble(
        self,
        records: dict[str, dict[int, list[dict[str, Any]]]],
        image_ids: list[int],
        confidence: float,
        nms_iou: float,
    ) -> dict[int, list[dict[str, Any]]]:
        ensemble: dict[int, list[dict[str, Any]]] = {}
        for image_id in image_ids:
            merged = []
            for condition in PROMPT_CONDITIONS:
                merged.extend(records[condition].get(image_id, []))
            ensemble[image_id] = canonical_nms(merged, confidence, nms_iou)
        return ensemble

    def tune_ensemble(
        self,
        records: dict[str, dict[int, list[dict[str, Any]]]],
        validation_ids: list[int],
    ) -> tuple[dict[str, float], pd.DataFrame]:
        rows = []
        for confidence, nms_iou in product(
            self.config["evaluation"]["tune_confidences"],
            self.config["evaluation"]["tune_nms_ious"],
        ):
            ensemble = self._build_ensemble(
                records, validation_ids, float(confidence), float(nms_iou)
            )
            summary, _ = evaluate_records(
                ensemble,
                self.gt_by_image,
                validation_ids,
                0.0,
                float(self.config["evaluation"]["match_iou"]),
            )
            rows.append(
                {
                    "confidence_threshold": float(confidence),
                    "nms_iou": float(nms_iou),
                    **summary,
                }
            )
        frame = pd.DataFrame(rows).sort_values(
            ["f1", "recall", "precision"],
            ascending=[False, False, False],
        )
        best = frame.iloc[0]
        selected = {
            "confidence_threshold": float(best["confidence_threshold"]),
            "nms_iou": float(best["nms_iou"]),
        }
        return selected, frame

    def tune_gamma(
        self,
        records: dict[str, dict[int, list[dict[str, Any]]]],
        validation_ids: list[int],
    ) -> tuple[dict[str, float], pd.DataFrame]:
        rows = []
        confidence = float(self.config["evaluation"]["confidence"])
        for gamma in self.config["improvement"]["gamma_candidates"]:
            condition = f"lowlight_gamma_{float(gamma):.2f}"
            summary, _ = evaluate_records(
                records[condition],
                self.gt_by_image,
                validation_ids,
                confidence,
                float(self.config["evaluation"]["match_iou"]),
            )
            rows.append(
                {
                    "gamma": float(gamma),
                    "distance_to_identity": abs(1.0 - float(gamma)),
                    **summary,
                }
            )
        frame = pd.DataFrame(rows).sort_values(
            ["f1", "recall", "precision", "distance_to_identity"],
            ascending=[False, False, False, True],
        )
        return {"gamma": float(frame.iloc[0]["gamma"])}, frame

    def run_full(self) -> dict[str, Any]:
        entries = self._entries_for_scope("all")
        validation_ids = [
            int(item["image_id"]) for item in entries
            if item["split"] == "validation"
        ]
        test_ids = [
            int(item["image_id"]) for item in entries
            if item["split"] == "test"
        ]
        records = self.collect_predictions("all", include_gamma=True)
        fixed_confidence = float(self.config["evaluation"]["confidence"])
        summaries, details = self._evaluate_conditions(
            records, test_ids, "test", fixed_confidence
        )

        selected, tuning = self.tune_ensemble(records, validation_ids)
        tuned_confidence = selected["confidence_threshold"]
        nms_iou = selected["nms_iou"]
        ensemble_test = self._build_ensemble(
            records, test_ids, tuned_confidence, nms_iou
        )
        ensemble_summary, ensemble_details = evaluate_records(
            ensemble_test,
            self.gt_by_image,
            test_ids,
            0.0,
            float(self.config["evaluation"]["match_iou"]),
        )
        baseline_tuned, baseline_tuned_details = evaluate_records(
            records["prompt_canonical"],
            self.gt_by_image,
            test_ids,
            tuned_confidence,
            float(self.config["evaluation"]["match_iou"]),
        )

        selected_gamma, gamma_tuning = self.tune_gamma(records, validation_ids)
        gamma = selected_gamma["gamma"]
        gamma_condition = f"lowlight_gamma_{gamma:.2f}"
        lowlight_baseline, lowlight_baseline_details = evaluate_records(
            records["brightness_0.25"],
            self.gt_by_image,
            test_ids,
            fixed_confidence,
            float(self.config["evaluation"]["match_iou"]),
        )
        gamma_summary, gamma_details = evaluate_records(
            records[gamma_condition],
            self.gt_by_image,
            test_ids,
            fixed_confidence,
            float(self.config["evaluation"]["match_iou"]),
        )
        summaries.extend(
            [
                {
                    "scope": "test",
                    "family": "improvement",
                    "condition": "baseline_tuned",
                    "confidence_threshold": tuned_confidence,
                    "nms_iou": None,
                    **baseline_tuned,
                },
                {
                    "scope": "test",
                    "family": "improvement",
                    "condition": "prompt_ensemble",
                    "confidence_threshold": tuned_confidence,
                    "nms_iou": nms_iou,
                    **ensemble_summary,
                },
                {
                    "scope": "test",
                    "family": "improvement",
                    "condition": "lowlight_baseline",
                    "confidence_threshold": fixed_confidence,
                    "nms_iou": None,
                    **lowlight_baseline,
                },
                {
                    "scope": "test",
                    "family": "improvement",
                    "condition": "gamma_correction",
                    "confidence_threshold": fixed_confidence,
                    "nms_iou": None,
                    "gamma": gamma,
                    **gamma_summary,
                },
            ]
        )
        for condition, rows in [
            ("baseline_tuned", baseline_tuned_details),
            ("prompt_ensemble", ensemble_details),
            ("lowlight_baseline", lowlight_baseline_details),
            ("gamma_correction", gamma_details),
        ]:
            details.extend(
                {
                    "scope": "test",
                    "family": "improvement",
                    "condition": condition,
                    "confidence_threshold": tuned_confidence,
                    **row,
                }
                for row in rows
            )

        pd.DataFrame(summaries).to_csv(
            self.results_dir / "summary_metrics.csv", index=False
        )
        pd.DataFrame(details).to_csv(
            self.results_dir / "per_image_metrics.csv", index=False
        )
        category_rows = []
        category_conditions = [
            ("original", records["original"], fixed_confidence),
            ("brightness_0.25", records["brightness_0.25"], fixed_confidence),
            ("blur_sigma_4", records["blur_sigma_4"], fixed_confidence),
            ("prompt_canonical", records["prompt_canonical"], fixed_confidence),
            ("prompt_synonym", records["prompt_synonym"], fixed_confidence),
            ("prompt_hypernym", records["prompt_hypernym"], fixed_confidence),
            ("prompt_ensemble", ensemble_test, 0.0),
            ("lowlight_baseline", records["brightness_0.25"], fixed_confidence),
            ("gamma_correction", records[gamma_condition], fixed_confidence),
        ]
        for condition, condition_records, threshold in category_conditions:
            for category in self.categories:
                category_predictions = {
                    image_id: [
                        pred for pred in condition_records.get(image_id, [])
                        if pred["category"] == category
                    ]
                    for image_id in test_ids
                }
                category_ground_truth = {
                    image_id: [
                        gt for gt in self.gt_by_image.get(image_id, [])
                        if gt["category"] == category
                    ]
                    for image_id in test_ids
                }
                category_summary, _ = evaluate_records(
                    category_predictions,
                    category_ground_truth,
                    test_ids,
                    threshold,
                    float(self.config["evaluation"]["match_iou"]),
                )
                category_rows.append(
                    {
                        "scope": "test",
                        "condition": condition,
                        "category": category,
                        "confidence_threshold": threshold,
                        **category_summary,
                    }
                )
        pd.DataFrame(category_rows).to_csv(
            self.results_dir / "metrics_by_category.csv", index=False
        )
        tuning.to_csv(self.results_dir / "ensemble_tuning.csv", index=False)
        gamma_tuning.to_csv(self.results_dir / "gamma_tuning.csv", index=False)
        self._save_predictions(records, self.results_dir / "raw_predictions.json")
        self._save_predictions(
            {"prompt_ensemble": ensemble_test},
            self.results_dir / "ensemble_predictions.json",
        )
        with (self.results_dir / "selected_parameters.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(
                {
                    "ensemble": selected,
                    "gamma_correction": selected_gamma,
                },
                handle,
                indent=2,
            )
        return {
            "selected_parameters": {
                "ensemble": selected,
                "gamma_correction": selected_gamma,
            },
            "summary_rows": summaries,
            "test_image_ids": test_ids,
        }

    @staticmethod
    def _save_predictions(
        records: dict[str, dict[int, list[dict[str, Any]]]],
        path: Path,
    ) -> None:
        serializable = {
            condition: {str(image_id): preds for image_id, preds in by_image.items()}
            for condition, by_image in records.items()
        }
        with path.open("w", encoding="utf-8") as handle:
            json.dump(serializable, handle, indent=2)
