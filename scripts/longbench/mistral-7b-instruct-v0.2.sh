#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="mistral-7b-instruct-v0.2" exec bash "${SCRIPT_DIR}/../common/longbench.sh" "$@"
