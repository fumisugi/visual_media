import json

from yolo_world_experiment.coco import build_holdout_manifest, xywh_to_xyxy


def test_xywh_to_xyxy():
    assert xywh_to_xyxy([10, 20, 30, 40]) == [10.0, 20.0, 40.0, 60.0]


def test_holdout_manifest_excludes_development_ids(tmp_path):
    annotations = {
        "images": [
            {
                "id": image_id,
                "file_name": f"{image_id}.jpg",
                "width": 100,
                "height": 100,
            }
            for image_id in range(1, 10)
        ],
        "annotations": [
            {
                "id": image_id,
                "image_id": image_id,
                "category_id": 47,
                "bbox": [0, 0, 40, 40],
                "area": 1600,
                "iscrowd": 0,
            }
            for image_id in range(1, 10)
        ],
    }
    annotations_path = tmp_path / "instances.json"
    annotations_path.write_text(json.dumps(annotations), encoding="utf-8")
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    for image_id in range(1, 10):
        (images_dir / f"{image_id}.jpg").touch()
    development_path = tmp_path / "development.json"
    development_path.write_text(
        json.dumps({"images": [{"image_id": 1}, {"image_id": 2}]}),
        encoding="utf-8",
    )
    holdout_path = tmp_path / "holdout.json"
    config = {
        "_project_root": str(tmp_path),
        "seed": 1,
        "paths": {
            "coco_annotations": str(annotations_path),
            "coco_images": str(images_dir),
            "subset_manifest": str(development_path),
            "holdout_manifest": str(holdout_path),
        },
        "selection": {
            "min_box_area": 1024,
            "holdout_images": 4,
            "holdout_seed": 2,
        },
        "categories": {
            "cup": {"coco_id": 47, "quota": 4},
        },
    }

    manifest = build_holdout_manifest(config)

    selected_ids = {item["image_id"] for item in manifest["images"]}
    assert len(selected_ids) == 4
    assert selected_ids.isdisjoint({1, 2})
    assert {item["split"] for item in manifest["images"]} == {"holdout"}
