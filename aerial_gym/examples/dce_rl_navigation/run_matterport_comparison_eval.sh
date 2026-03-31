#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_matterport_comparison_eval.sh
#
# Launch the DCE vs ViT side-by-side comparison evaluation.
# Two parallel drones navigate the same Matterport scene to the same goal:
#   Drone 0 — depth → VAE   (DCE pipeline)
#   Drone 1 — RGB   → ViT + adapter  (ViT pipeline)
#
# Both drones share the same RL policy weights.
# Per-pipeline episode statistics (success / crash / timeout) are logged
# separately at the end.
#
# Prerequisites:
#   1. Run the export script inside the sampl_geometic_head docker first:
#        python export/export_vit_adapter.py \
#            --output_dir /home/anto/Documents/shared_models/
#
#   2. Make sure VIT_MODEL_PATH and VIT_METADATA below point to the exported
#      files (or override them with environment variables before calling this
#      script).
#
# Usage (from the dce_rl_navigation directory):
#   ./run_matterport_comparison_eval.sh [extra args...]
#
# Examples:
#   # Default scene + default shared_models path
#   ./run_matterport_comparison_eval.sh
#
#   # Custom scene
#   ./run_matterport_comparison_eval.sh --scene_folder resources/envs/00801-HaxA7YrQdEC
#
#   # Custom model paths
#   VIT_MODEL_PATH=/my/path/vit_adapter_pipeline_240x320.pt \
#   VIT_METADATA=/my/path/metadata.json \
#   ./run_matterport_comparison_eval.sh
#
# Environment variable overrides:
#   TRAIN_DIR              path to the Sample Factory train_dir  (default: $(pwd)/selected_network)
#   EXPERIMENT             SF experiment name                    (default: selected_network)
#   CHECKPOINT_KIND        "best" or "latest"                    (default: best)
#   VIT_MODEL_PATH         path to vit_adapter_pipeline_HxW.pt  (default: shared_models dir)
#   VIT_METADATA           path to metadata.json                 (default: shared_models dir)
#   SHARED_MODELS_DIR      base dir for VIT_MODEL_PATH / VIT_METADATA overrides
# ---------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- RL policy checkpoint ---
TRAIN_DIR="${TRAIN_DIR:-$(pwd)/selected_network}"
EXPERIMENT="${EXPERIMENT:-selected_network}"
CHECKPOINT_KIND="${CHECKPOINT_KIND:-best}"

# --- ViT export artifacts ---
SHARED_MODELS_DIR="${SHARED_MODELS_DIR:-/shared_models}"
VIT_MODEL_PATH="${VIT_MODEL_PATH:-${SHARED_MODELS_DIR}/vit_adapter_pipeline_240x320.pt}"
VIT_METADATA="${VIT_METADATA:-${SHARED_MODELS_DIR}/metadata.json}"

# --- Timing / rendering ---
VIEWER_EVERY="${VIEWER_EVERY:-3}"
DISPLAY_EVERY="${DISPLAY_EVERY:-4}"
POLICY_EVERY="${POLICY_EVERY:-1}"
VAE_ENCODE_EVERY="${VAE_ENCODE_EVERY:-${POLICY_EVERY}}"
TEXTURE_ATLAS_TILE_SIZE="${TEXTURE_ATLAS_TILE_SIZE:-1024}"
SCENE_SCALE="${SCENE_SCALE:-2.0}"
MAX_EPISODES="${MAX_EPISODES:-500}"

# Validate that the ViT model files exist before launching Isaac Gym
if [ ! -f "${VIT_MODEL_PATH}" ]; then
    echo "ERROR: ViT model not found: ${VIT_MODEL_PATH}"
    echo ""
    echo "Export it first (inside the sampl_geometic_head docker):"
    echo "  python export/export_vit_adapter.py --output_dir ${SHARED_MODELS_DIR}"
    exit 1
fi
if [ ! -f "${VIT_METADATA}" ]; then
    echo "ERROR: ViT metadata not found: ${VIT_METADATA}"
    exit 1
fi

echo "=== DCE vs ViT Comparison Eval ==="
echo "  RL policy  : ${TRAIN_DIR}/${EXPERIMENT}"
echo "  ViT model  : ${VIT_MODEL_PATH}"
echo "  ViT meta   : ${VIT_METADATA}"
echo "  Scene scale: ${SCENE_SCALE}"
echo ""

python3 "${SCRIPT_DIR}/eval_matterport_dce.py" \
    --pipeline=comparison \
    --vit_model_path="${VIT_MODEL_PATH}" \
    --vit_metadata="${VIT_METADATA}" \
    --train_dir="${TRAIN_DIR}" \
    --experiment="${EXPERIMENT}" \
    --env=test \
    --obs_key="observations" \
    --load_checkpoint_kind="${CHECKPOINT_KIND}" \
    --viewer_every="${VIEWER_EVERY}" \
    --display_every="${DISPLAY_EVERY}" \
    --policy_every="${POLICY_EVERY}" \
    --vae_encode_every="${VAE_ENCODE_EVERY}" \
    --scene_scale="${SCENE_SCALE}" \
    --texture_atlas_tile_size="${TEXTURE_ATLAS_TILE_SIZE}" \
    --max_episodes="${MAX_EPISODES}" \
    "$@"
