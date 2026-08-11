#!/usr/bin/env bash
set -eu

project_root=${PIXI_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
if [[ -x "$HOME/.pixi/bin/pixi" ]]; then
    export PATH="$HOME/.pixi/bin:$PATH"
fi
config_path="$project_root/.vslamlab-storage"
if [[ -f "$config_path" ]]; then
    shared_root=$(<"$config_path")
    if [[ "$shared_root" = /* && -d "$shared_root" ]]; then
        export HF_HOME="$shared_root/cache/huggingface"
        export TORCH_HOME="$shared_root/cache/torch"
        export TRITON_CACHE_DIR="$shared_root/cache/triton"
    fi
fi
