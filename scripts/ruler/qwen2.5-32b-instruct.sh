#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="qwen2.5-32b-instruct" MAX_SEQ_LENGTHS="8000 16000 32000" exec bash "${SCRIPT_DIR}/../common/ruler.sh" "$@"
