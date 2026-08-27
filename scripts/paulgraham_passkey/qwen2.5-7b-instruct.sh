#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${MODEL:-qwen2.5-7b-instruct}" DATASET="${DATASET:-81920words_10x10x3_7digits}" exec bash "${SCRIPT_DIR}/../common/paulgraham_passkey.sh" "$@"
