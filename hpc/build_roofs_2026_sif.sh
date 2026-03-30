#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_SIF="${1:-${SCRIPT_DIR}/roofs_2026.sif}"

cd "${PROJECT_ROOT}"
apptainer build "${OUTPUT_SIF}" "${SCRIPT_DIR}/roofs_2026.def"
