#!/usr/bin/env bash
set -euo pipefail

target="${1:-help}"

case "${target}" in
  longbench)
    python scripts/dataset_prep/download_longbench.py
    ;;
  ruler)
    bash scripts/dataset_prep/download_ruler.sh
    ;;
  math_reasoning)
    python scripts/dataset_prep/download_math_reasoning.py
    ;;
  longalpaca)
    python scripts/probe/prepare_longalpaca_decode_calib.py
    ;;
  all)
    python scripts/dataset_prep/download_longbench.py
    bash scripts/dataset_prep/download_ruler.sh
    python scripts/dataset_prep/download_math_reasoning.py
    ;;
  help|--help|-h)
    echo "Usage: bash scripts/setup_external_assets.sh <longbench|ruler|math_reasoning|longalpaca|all>"
    ;;
  *)
    echo "Unknown target: ${target}" >&2
    echo "Usage: bash scripts/setup_external_assets.sh <longbench|ruler|math_reasoning|longalpaca|all>" >&2
    exit 1
    ;;
esac
