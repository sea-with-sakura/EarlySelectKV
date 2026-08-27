#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="qwen2.5-32b-instruct" exec bash "${SCRIPT_DIR}/../common/longbench.sh" "$@"
