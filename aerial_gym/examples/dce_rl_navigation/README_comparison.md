# DCE vs ViT Comparison Evaluation

Side-by-side benchmark of two depth-encoding pipelines driving the same RL
navigation policy in a Matterport 3D scene.

| | DCE (Drone 0) | ViT (Drone 1) |
|---|---|---|
| **Input modality** | Depth image | RGB image |
| **Encoder** | Frozen VAE (ResNet-8) | DINOv3 ViT-S/16 + FactorizedDenseHeadV2 |
| **Latent dim** | 64 | 64 |
| **RL policy** | shared — same checkpoint | shared — same checkpoint |

Both drones spawn at the same position and navigate to the same goal every
episode. Per-pipeline statistics (success / crash / timeout rates) are logged
separately so the results are directly comparable.

---

## Quick start

### Step 1 — Export the ViT models (sampl_geometic_head docker)

```bash
python export/export_vit_adapter.py \
    --output_dir /home/anto/Documents/shared_models/
```

This writes `vit_adapter_pipeline_240x320.pt` and `metadata.json` to the
shared folder. Only needs to be done once per checkpoint.

### Step 2 — Run the comparison (aerial_gym docker)

```bash
cd aerial_gym/examples/dce_rl_navigation
./run_matterport_comparison_eval.sh
```

Or with explicit paths if the shared folder is different:

```bash
VIT_MODEL_PATH=/my/models/vit_adapter_pipeline_240x320.pt \
VIT_METADATA=/my/models/metadata.json \
./run_matterport_comparison_eval.sh
```

---

## Shell script env-var overrides

| Variable | Default | Description |
|----------|---------|-------------|
| `TRAIN_DIR` | `$(pwd)/selected_network` | Sample Factory train dir |
| `EXPERIMENT` | `selected_network` | SF experiment name |
| `CHECKPOINT_KIND` | `best` | `best` or `latest` |
| `VIT_MODEL_PATH` | `shared_models/vit_adapter_pipeline_240x320.pt` | TorchScript model |
| `VIT_METADATA` | `shared_models/metadata.json` | Export metadata |
| `SHARED_MODELS_DIR` | `/home/anto/Documents/shared_models` | Base dir for model paths |
| `SCENE_SCALE` | `2.0` | Global scene scale factor |
| `MAX_EPISODES` | `500` | Episodes before stopping |
| `POLICY_EVERY` | `1` | Policy inference cadence |
| `VAE_ENCODE_EVERY` | `= POLICY_EVERY` | Encoding cadence |
| `VIEWER_EVERY` | `3` | Isaac Gym viewer refresh |
| `DISPLAY_EVERY` | `4` | OpenCV window refresh |

---

## Display layout

The OpenCV window shows a **2x2 grid** during comparison mode. Panels for the
*unused* modality are dimmed and labelled so it is immediately clear which
signal each drone actually processes.

```
┌──────────────────────┬──────────────────────┐
│  DCE: RGB (unused)   │  DCE: Depth (used)   │  ← Drone 0 / DCE pipeline
├──────────────────────┼──────────────────────┤
│  ViT: RGB (used)     │  ViT: Depth (unused) │  ← Drone 1 / ViT pipeline
└──────────────────────┴──────────────────────┘
```

The Isaac Gym 3D viewer always shows Drone 0 (DCE).

---

## New / modified files

| File | Role |
|------|------|
| `eval_matterport_dce.py` | Refactored eval script; add `--pipeline=comparison` |
| `dce_vit_comparison_task.py` | `DCEViTComparisonTask` — dual-encoder navigation task |
| `vit_adapter_encoder.py` | Loads TorchScript ViT+adapter, mirrors `VAEImageEncoder.encode()` |
| `run_matterport_comparison_eval.sh` | One-command launcher for comparison mode |
| `run_matterport_dce_eval.sh` | Original DCE-only launcher (unchanged) |

The DCE-only workflow (`run_matterport_dce_eval.sh` / `--pipeline=dce`) is
fully unchanged.
