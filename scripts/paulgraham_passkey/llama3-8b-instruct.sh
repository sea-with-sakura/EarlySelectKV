#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${MODEL:-llama3-8b-instruct}" DATASET="${DATASET:-4096words_10x10x3_7digits}" exec bash "${SCRIPT_DIR}/../common/paulgraham_passkey.sh" "$@"
