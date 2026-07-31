#!/usr/bin/env bash
set -euo pipefail

: "${DATA_ANALYTICS_PLUGIN_ROOT:?Set DATA_ANALYTICS_PLUGIN_ROOT to the installed Data Analytics plugin directory.}"

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
artifact_path="${project_root}/results/yoloworld/artifact.json"
html_path="${project_root}/results/yoloworld/YOLO_World_Failure_Case_Report.html"
pdf_path="${project_root}/results/yoloworld/YOLO_World_Failure_Case_Report.pdf"
builder="${DATA_ANALYTICS_PLUGIN_ROOT}/skills/build-report/scripts/deliver_portable_artifact.mjs"
chrome_path="${CHROMIUM_EXECUTABLE_PATH:-/usr/bin/google-chrome}"

node "${builder}" --input "${artifact_path}" --output "${html_path}"

"${chrome_path}" \
  --headless=new \
  --no-sandbox \
  --disable-gpu \
  --disable-dev-shm-usage \
  --no-first-run \
  --no-default-browser-check \
  --no-pdf-header-footer \
  --run-all-compositor-stages-before-draw \
  --virtual-time-budget=8000 \
  "--print-to-pdf=${pdf_path}" \
  "file://${html_path}"

pdfinfo "${pdf_path}" | sed -n '/^Pages:/p;/^Page size:/p;/^File size:/p'
