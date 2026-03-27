#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_matterport_dce_eval.sh
#
# Launch the DCE RL navigation policy evaluation in a Matterport 3D scene.
#
# The script wraps eval_matterport_dce.py with sensible defaults that mirror
# run_trained_navigation_policy.sh, while also opening the Isaac Gym viewer
# and a real-time RGB/depth matplotlib window.
#
# Usage (from the dce_rl_navigation directory):
#   ./run_matterport_dce_eval.sh [extra args...]
#
# Examples:
#   # Use default checkpoint and scene
#   ./run_matterport_dce_eval.sh
#
#   # Custom scene folder
#   ./run_matterport_dce_eval.sh --scene_folder resources/envs/00801-HaxA7YrQdEC
#
#   # Custom checkpoint + more episodes
#   TRAIN_DIR=/path/to/checkpoint ./run_matterport_dce_eval.sh --max_episodes=100
#
# Environment variable overrides:
#   TRAIN_DIR         path to the Sample Factory train_dir  (default: $(pwd)/selected_network)
#   EXPERIMENT        SF experiment name                    (default: selected_network)
#   CHECKPOINT_KIND   "best" or "latest"                    (default: best)
# ---------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TRAIN_DIR="${TRAIN_DIR:-$(pwd)/selected_network}"
EXPERIMENT="${EXPERIMENT:-selected_network}"
CHECKPOINT_KIND="${CHECKPOINT_KIND:-best}"

python3 "${SCRIPT_DIR}/eval_matterport_dce.py" \
    --train_dir="${TRAIN_DIR}" \
    --experiment="${EXPERIMENT}" \
    --env=test \
    --obs_key="observations" \
    --load_checkpoint_kind="${CHECKPOINT_KIND}" \
    "$@"
