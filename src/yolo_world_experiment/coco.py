from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from .config import resolve_path


def load_coco_annotations(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def xywh_to_xyxy(box: list[float]) -> list[float]:
    x, y, width, height = box
    return [float(x), float(y), float(x + width), float(y + height)]


def build_subset_manifest(config: dict[str, Any]) -> dict[str, Any]:
    return _build_balanced_manifest(
        config,
        output_path=resolve_path(config, config["paths"]["subset_manifest"]),
        seed=int(config["seed"]),
        total_images=int(config["selection"]["total_images"]),
        quotas={
            name: int(details["quota"])
            for name, details in config["categories"].items()
        },
        excluded_ids=set(),
        validation_count=int(config["selection"]["validation_images"]),
        pilot_count=int(config["selection"]["pilot_images"]),
        fixed_split_name=None,
    )


def build_holdout_manifest(config: dict[str, Any]) -> dict[str, Any]:
    """Build a balanced, untouched holdout with no overlap with the dev set."""
    development = load_manifest(config)
    excluded_ids = {
        int(item["image_id"]) for item in development["images"]
    }
    total_images = int(config["selection"]["holdout_images"])
    category_count = len(config["categories"])
    if total_images % category_count:
        raise ValueError(
            "holdout_images must be divisible by the number of categories"
        )
    quota = total_images // category_count
    return _build_balanced_manifest(
        config,
        output_path=resolve_path(config, config["paths"]["holdout_manifest"]),
        seed=int(config["selection"]["holdout_seed"]),
        total_images=total_images,
        quotas={name: quota for name in config["categories"]},
        excluded_ids=excluded_ids,
        validation_count=0,
        pilot_count=0,
        fixed_split_name="holdout",
    )


def build_final_holdout_manifest(config: dict[str, Any]) -> dict[str, Any]:
    """Build a second untouched holdout after the first replication check."""
    excluded_ids: set[int] = set()
    for key in ("subset_manifest", "holdout_manifest"):
        path = resolve_path(config, config["paths"][key])
        payload = json.loads(path.read_text(encoding="utf-8"))
        excluded_ids.update(
            int(item["image_id"]) for item in payload["images"]
        )
    total_images = int(config["selection"]["holdout_images"])
    category_count = len(config["categories"])
    if total_images % category_count:
        raise ValueError(
            "holdout_images must be divisible by the number of categories"
        )
    quota = total_images // category_count
    return _build_balanced_manifest(
        config,
        output_path=resolve_path(
            config, config["paths"]["final_holdout_manifest"]
        ),
        seed=int(config["selection"]["final_holdout_seed"]),
        total_images=total_images,
        quotas={name: quota for name in config["categories"]},
        excluded_ids=excluded_ids,
        validation_count=0,
        pilot_count=0,
        fixed_split_name="final_holdout",
    )


def build_same_model_holdout_manifest(
    config: dict[str, Any],
) -> dict[str, Any]:
    """Build the untouched holdout after the same-model method is frozen."""
    selected_path = (
        resolve_path(config, config["paths"]["same_model_results_dir"])
        / "selected_same_model_method.json"
    )
    if not selected_path.exists():
        raise FileNotFoundError(
            "Freeze the same-model method before creating its holdout"
        )
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    if not bool(selected.get("development_gate_passed")):
        raise RuntimeError(
            "The same-model method did not pass its development gate"
        )
    if selected.get("model") != "yolov8s-worldv2.pt":
        raise RuntimeError("The frozen method does not use the Small model")

    excluded_ids: set[int] = set()
    for key in (
        "subset_manifest",
        "holdout_manifest",
        "final_holdout_manifest",
    ):
        path = resolve_path(config, config["paths"][key])
        payload = json.loads(path.read_text(encoding="utf-8"))
        excluded_ids.update(
            int(item["image_id"]) for item in payload["images"]
        )
    total_images = int(config["selection"]["holdout_images"])
    quotas = {
        category: int(quota)
        for category, quota in config["same_model_improvement"][
            "final_holdout_quotas"
        ].items()
    }
    if set(quotas) != set(config["categories"]):
        raise ValueError(
            "same-model holdout quotas must cover every target category"
        )
    if sum(quotas.values()) != total_images:
        raise ValueError(
            "same-model holdout quotas must sum to holdout_images"
        )
    return _build_balanced_manifest(
        config,
        output_path=resolve_path(
            config, config["paths"]["same_model_holdout_manifest"]
        ),
        seed=int(config["selection"]["same_model_holdout_seed"]),
        total_images=total_images,
        quotas=quotas,
        excluded_ids=excluded_ids,
        validation_count=0,
        pilot_count=0,
        fixed_split_name="same_model_holdout",
    )


def build_prompt_prototype_holdout_manifest(
    config: dict[str, Any],
) -> dict[str, Any]:
    """Build a new holdout after the prompt prototype is frozen."""
    selected_path = (
        resolve_path(config, config["paths"]["prompt_prototype_results_dir"])
        / "selected_prompt_prototype_method.json"
    )
    if not selected_path.exists():
        raise FileNotFoundError(
            "Freeze the prompt prototype method before creating its holdout"
        )
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    if not bool(selected.get("development_gate_passed")):
        raise RuntimeError(
            "The prompt prototype method did not pass development"
        )
    if selected.get("model") != "yolov8s-worldv2.pt":
        raise RuntimeError("The frozen prompt method does not use Small")
    if not bool(selected.get("canonical_condition_is_unchanged")):
        raise RuntimeError(
            "The frozen prompt method does not preserve canonical prompting"
        )

    excluded_ids: set[int] = set()
    for key in (
        "subset_manifest",
        "holdout_manifest",
        "final_holdout_manifest",
        "same_model_holdout_manifest",
    ):
        payload = json.loads(
            resolve_path(config, config["paths"][key]).read_text(
                encoding="utf-8"
            )
        )
        excluded_ids.update(
            int(item["image_id"]) for item in payload["images"]
        )
    quotas = {
        category: int(quota)
        for category, quota in config["prompt_prototype_improvement"][
            "final_holdout_quotas"
        ].items()
    }
    total_images = int(config["selection"]["holdout_images"])
    if set(quotas) != set(config["categories"]):
        raise ValueError(
            "prompt prototype holdout quotas must cover all categories"
        )
    if sum(quotas.values()) != total_images:
        raise ValueError(
            "prompt prototype holdout quotas must sum to holdout_images"
        )
    category_order = sorted(
        quotas,
        key=lambda category: (
            quotas[category],
            list(config["categories"]).index(category),
        ),
    )
    return _build_balanced_manifest(
        config,
        output_path=resolve_path(
            config, config["paths"]["prompt_prototype_holdout_manifest"]
        ),
        seed=int(config["selection"]["prompt_prototype_holdout_seed"]),
        total_images=total_images,
        quotas=quotas,
        excluded_ids=excluded_ids,
        validation_count=0,
        pilot_count=0,
        fixed_split_name="prompt_prototype_holdout",
        min_box_area_override=float(
            config["prompt_prototype_improvement"][
                "final_holdout_min_box_area"
            ]
        ),
        category_order=category_order,
    )


def _build_balanced_manifest(
    config: dict[str, Any],
    *,
    output_path: Path,
    seed: int,
    total_images: int,
    quotas: dict[str, int],
    excluded_ids: set[int],
    validation_count: int,
    pilot_count: int,
    fixed_split_name: str | None,
    min_box_area_override: float | None = None,
    category_order: list[str] | None = None,
) -> dict[str, Any]:
    annotations_path = resolve_path(config, config["paths"]["coco_annotations"])
    images_dir = resolve_path(config, config["paths"]["coco_images"])
    coco = load_coco_annotations(annotations_path)

    image_by_id = {int(item["id"]): item for item in coco["images"]}
    target_by_coco_id = {
        int(details["coco_id"]): name
        for name, details in config["categories"].items()
    }
    eligible_by_category: dict[str, list[int]] = defaultdict(list)
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    min_area = (
        float(config["selection"]["min_box_area"])
        if min_box_area_override is None
        else float(min_box_area_override)
    )

    for ann in coco["annotations"]:
        coco_id = int(ann["category_id"])
        if coco_id not in target_by_coco_id or int(ann.get("iscrowd", 0)) == 1:
            continue
        if float(ann["bbox"][2] * ann["bbox"][3]) < min_area:
            continue
        category = target_by_coco_id[coco_id]
        image_id = int(ann["image_id"])
        annotations_by_image[image_id].append(
            {
                "annotation_id": int(ann["id"]),
                "category": category,
                "category_id": coco_id,
                "bbox_xyxy": xywh_to_xyxy(ann["bbox"]),
                "area": float(ann["area"]),
            }
        )

    for image_id, anns in annotations_by_image.items():
        if image_id in excluded_ids:
            continue
        for category in sorted({ann["category"] for ann in anns}):
            eligible_by_category[category].append(image_id)

    rng = random.Random(seed)
    for image_ids in eligible_by_category.values():
        rng.shuffle(image_ids)

    quotas = dict(quotas)
    positions = defaultdict(int)
    chosen: list[tuple[int, str]] = []
    chosen_ids: set[int] = set()

    while any(quotas[name] > 0 for name in quotas):
        made_progress = False
        for category in (
            list(config["categories"])
            if category_order is None
            else category_order
        ):
            if quotas[category] <= 0:
                continue
            candidates = eligible_by_category[category]
            while (
                positions[category] < len(candidates)
                and candidates[positions[category]] in chosen_ids
            ):
                positions[category] += 1
            if positions[category] >= len(candidates):
                raise RuntimeError(f"Not enough eligible COCO images for {category}")
            image_id = candidates[positions[category]]
            positions[category] += 1
            chosen_ids.add(image_id)
            chosen.append((image_id, category))
            quotas[category] -= 1
            made_progress = True
        if not made_progress:
            raise RuntimeError("Could not complete balanced COCO subset selection")

    expected_total = int(total_images)
    if len(chosen) != expected_total:
        raise RuntimeError(
            f"Selection produced {len(chosen)} images, expected {expected_total}"
        )

    entries = []
    for index, (image_id, primary_category) in enumerate(chosen):
        image = image_by_id[image_id]
        file_path = images_dir / image["file_name"]
        if not file_path.exists():
            raise FileNotFoundError(f"Missing COCO image: {file_path}")
        split = (
            fixed_split_name
            if fixed_split_name is not None
            else ("validation" if index < validation_count else "test")
        )
        entries.append(
            {
                "image_id": image_id,
                "file_name": image["file_name"],
                "width": int(image["width"]),
                "height": int(image["height"]),
                "primary_category": primary_category,
                "split": split,
                "pilot": index < pilot_count,
                "annotations": annotations_by_image[image_id],
            }
        )

    manifest = {
        "source": "COCO 2017 validation set",
        "seed": seed,
        "selection": {
            "total_images": expected_total,
            "validation_images": validation_count,
            "test_images": (
                expected_total - validation_count
                if fixed_split_name is None
                else 0
            ),
            "holdout_images": (
                expected_total if fixed_split_name == "holdout" else 0
            ),
            "final_holdout_images": (
                expected_total
                if fixed_split_name == "final_holdout"
                else 0
            ),
            "same_model_holdout_images": (
                expected_total
                if fixed_split_name == "same_model_holdout"
                else 0
            ),
            "prompt_prototype_holdout_images": (
                expected_total
                if fixed_split_name == "prompt_prototype_holdout"
                else 0
            ),
            "pilot_images": pilot_count,
            "min_box_area": min_area,
            "excluded_image_count": len(excluded_ids),
        },
        "categories": list(config["categories"].keys()),
        "images": entries,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    return manifest


def load_manifest(config: dict[str, Any]) -> dict[str, Any]:
    manifest_path = resolve_path(config, config["paths"]["subset_manifest"])
    with manifest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ground_truth_by_image(
    manifest: dict[str, Any],
) -> dict[int, list[dict[str, Any]]]:
    return {
        int(item["image_id"]): [
            {
                "category": ann["category"],
                "bbox_xyxy": [float(value) for value in ann["bbox_xyxy"]],
            }
            for ann in item["annotations"]
        ]
        for item in manifest["images"]
    }
