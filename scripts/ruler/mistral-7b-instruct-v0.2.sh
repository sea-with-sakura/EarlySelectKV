#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="mistral-7b-instruct-v0.2" MAX_SEQ_LENGTHS="8000 16000 24000 32000" exec bash "${SCRIPT_DIR}/../common/ruler.sh" "$@"
