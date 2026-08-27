#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="qwen2.5-7b-instruct" MAX_SEQ_LENGTHS="16000 32000 64000 96000" exec bash "${SCRIPT_DIR}/../common/ruler.sh" "$@"
