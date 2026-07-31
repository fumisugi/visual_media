#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
raw_dir="${project_root}/data/raw"
mkdir -p "${raw_dir}"

if [[ ! -f "${raw_dir}/val2017.zip" ]]; then
  curl -fL --retry 3 \
    "http://images.cocodataset.org/zips/val2017.zip" \
    -o "${raw_dir}/val2017.zip"
fi

if [[ ! -f "${raw_dir}/annotations_trainval2017.zip" ]]; then
  curl -fL --retry 3 \
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip" \
    -o "${raw_dir}/annotations_trainval2017.zip"
fi

if [[ ! -d "${raw_dir}/val2017" ]]; then
  unzip -q "${raw_dir}/val2017.zip" -d "${raw_dir}"
fi

if [[ ! -f "${raw_dir}/annotations/instances_val2017.json" ]]; then
  unzip -q "${raw_dir}/annotations_trainval2017.zip" -d "${raw_dir}"
fi

echo "COCO 2017 val is ready under ${raw_dir}"
