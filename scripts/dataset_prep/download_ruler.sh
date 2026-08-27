#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../../utils/ruler_utils/data/synthetic/json"
python download_paulgraham_essay.py
bash download_qa_dataset.sh
