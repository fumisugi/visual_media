#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

conda run -n visual_media_yoloworld \
  yoloworld-experiment \
  --config configs/experiment.yaml \
  latex-report

pdfinfo results/yoloworld/YOLO_World_Failure_Case_Report.pdf \
  | sed -n '/^Pages:/p;/^Page size:/p;/^File size:/p'
