#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="llama3.1-8b-instruct" MAX_SEQ_LENGTHS="16000 32000 64000 96000" exec bash "${SCRIPT_DIR}/../common/ruler.sh" "$@"
