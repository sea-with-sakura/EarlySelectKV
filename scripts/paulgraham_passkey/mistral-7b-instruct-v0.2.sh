#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${MODEL:-mistral-7b-instruct-v0.2}" DATASET="${DATASET:-20480words_10x10x3_7digits}" exec bash "${SCRIPT_DIR}/../common/paulgraham_passkey.sh" "$@"
