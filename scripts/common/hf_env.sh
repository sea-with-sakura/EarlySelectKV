#!/usr/bin/env bash

hf_export_env() {
  if [ -n "${HF_HOME:-}" ]; then
    export HF_HOME
  else
    export HF_HOME="${HOME}/.cache/huggingface"
  fi

  export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
  export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
  export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"
  export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

  if [ -z "${HF_TOKEN:-}" ]; then
    local token_file="${HF_HOME}/token"
    if [ -r "${token_file}" ]; then
      local token
      token="$(head -n 1 "${token_file}" | tr -d '\r\n')"
      if [ -n "${token}" ]; then
        export HF_TOKEN="${token}"
      fi
    fi
  fi

  if [ -n "${HF_TOKEN:-}" ]; then
    export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN}}"
    export HUGGINGFACE_TOKEN="${HUGGINGFACE_TOKEN:-${HF_TOKEN}}"
    export SKIP_HF_LOGIN="${SKIP_HF_LOGIN:-1}"
  fi

  if [ "${SAKURA_HF_OFFLINE:-0}" = "1" ]; then
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
  fi
}
