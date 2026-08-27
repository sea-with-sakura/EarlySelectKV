#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="llama3-8b-instruct" MAX_SEQ_LENGTHS="4000 8000" exec bash "${SCRIPT_DIR}/../common/ruler.sh" "$@"
