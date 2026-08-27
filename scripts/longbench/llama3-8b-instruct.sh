#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="llama3-8b-instruct" exec bash "${SCRIPT_DIR}/../common/longbench.sh" "$@"
