import argparse
import gc
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime
from typing import Any, Dict

import numpy as np
import yaml

from aerial_gym.benchmark.benchmark_utils import (
    collect_grid_frame,
    compute_benchmark_metrics,
    deep_update,
    maybe_save_case_gif,
)
from aerial_gym.benchmark.matterport_spawn_helpers import (
    apply_navmesh_config,
    apply_spawn_region,
    resolve_scene_bundle,
)
from aerial_gym.config.env_config.matterport_glb_env import MatterportGLBEnvCfg
from aerial_gym.config.sensor_config.camera_config.base_depth_camera_config import (
    BaseDepthCameraConfig,
)
from aerial_gym.config.sensor_config.camera_config.shaded_rgbd_camera_config import (
    ShadedRGBDCameraConfig,
)
from aerial_gym.sim.sim_builder import SimBuilder
from aerial_gym.utils.logging import CustomLogger
import torch


logger = CustomLogger(__name__)


def _default_config() -> Dict[str, Any]:
    return {
        "seed": 0,
        "env_counts": [1, 2, 4, 8, 16],
        "warmup_steps": 100,
        "bench_steps": 300,
        "headless": True,
        "device": "cuda:0",
        "controller_name": "lee_position_control",
        "camera": {
            "width": 240,
            "height": 135,
            "max_range": 80.0,
            "enable_lighting": False,
        },
        "render": {
            "render_every": 1,
        },
        "gif": {
            "save_gifs": False,
            "every": 5,
            "max_frames": 300,
            "duration_ms": 60,
        },
        "scene": {
            "file": None,
            "folder": None,
            "root": "resources/envs",
        },
        "spawn_region": {},
        "navmesh": {
            "enabled": None,
            "navmesh_file": None,
            "edge_padding": 0.30,
            "oversample_factor": 6,
            "max_resample_rounds": 10,
            "strict_edge_padding": True,
            "min_edge_padding_ratio": 0.35,
            "padding_relaxation_factor": 0.70,
            "max_padding_relax_rounds": 3,
            "enforce_env_bounds": True,
            "max_bound_resample_rounds": 5,
            "spawn_height_offset_range": [0.15, 0.40],
            "zero_velocity_on_spawn": True,
        },
        "robots": {
            "depth_only": "base_quadrotor_with_camera",
            "shaded_rgbd": "base_quadrotor_with_shaded_rgbd_camera",
        },
        "output": {
            "report_path": "aerial_gym/benchmark/stored_data/matterport_depth_vs_rgbd_benchmark.json",
        },
    }


def _load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config root must be a mapping in {config_path}")

    cfg = _default_config()
    deep_update(cfg, loaded)

    if not isinstance(cfg.get("env_counts"), list) or len(cfg["env_counts"]) == 0:
        raise ValueError("config.env_counts must be a non-empty list of integers")
    cfg["env_counts"] = [int(x) for x in cfg["env_counts"]]
    return cfg


def _configure_camera_classes(width: int, height: int, max_range: float, enable_lighting: bool):
    BaseDepthCameraConfig.width = width
    BaseDepthCameraConfig.height = height
    BaseDepthCameraConfig.max_range = max_range
    BaseDepthCameraConfig.segmentation_camera = False
    BaseDepthCameraConfig.calculate_depth = True

    ShadedRGBDCameraConfig.width = width
    ShadedRGBDCameraConfig.height = height
    ShadedRGBDCameraConfig.max_range = max_range
    ShadedRGBDCameraConfig.enable_lighting = enable_lighting
    ShadedRGBDCameraConfig.debug_uv_checker = False


def _run_case(
    robot_name: str,
    controller_name: str,
    num_envs: int,
    warmup_steps: int,
    bench_steps: int,
    device: str,
    headless: bool,
    save_gif: bool,
    gif_every: int,
    gif_max_frames: int,
    gif_duration_ms: int,
    render_every: int,
):
    env_manager = SimBuilder().build_env(
        sim_name="base_sim",
        env_name="matterport_glb_env",
        robot_name=robot_name,
        controller_name=controller_name,
        args=None,
        device=device,
        num_envs=num_envs,
        headless=headless,
        use_warp=True,
    )

    actions = torch.zeros((env_manager.num_envs, 4), device=device)
    env_manager.reset()

    frames = []
    total_steps = 0
    render_count = 0
    render_every = max(int(render_every), 1)

    robot_position = env_manager.global_tensor_dict["robot_position"]
    robot_euler_angles = env_manager.global_tensor_dict.get("robot_euler_angles", None)
    if "env_origins" in env_manager.global_tensor_dict:
        env_origins = env_manager.global_tensor_dict["env_origins"]
    elif hasattr(env_manager, "env_origins"):
        env_origins = env_manager.env_origins
    else:
        env_origins = torch.zeros_like(robot_position)

    target_local_pos = torch.zeros((env_manager.num_envs, 3), device=device)
    target_yaw = torch.zeros(env_manager.num_envs, device=device)

    def refresh_targets(env_ids=None):
        local_pos = robot_position - env_origins
        if env_ids is None:
            target_local_pos[:] = local_pos
            if robot_euler_angles is not None:
                target_yaw[:] = robot_euler_angles[:, 2]
            else:
                target_yaw[:] = 0.0
            return

        if len(env_ids) == 0:
            return
        target_local_pos[env_ids] = local_pos[env_ids]
        if robot_euler_angles is not None:
            target_yaw[env_ids] = robot_euler_angles[env_ids, 2]
        else:
            target_yaw[env_ids] = 0.0

    def apply_targets_to_actions():
        actions[:, 0:3] = target_local_pos
        actions[:, 3] = target_yaw

    refresh_targets()

    def maybe_capture_frame():
        nonlocal total_steps
        if (not save_gif) or gif_every <= 0 or len(frames) >= gif_max_frames:
            return
        if total_steps % gif_every != 0:
            return
        frames.append(collect_grid_frame(env_manager))

    with torch.no_grad():
        for _ in range(warmup_steps):
            apply_targets_to_actions()
            env_manager.step(actions=actions)
            if total_steps % render_every == 0:
                env_manager.render(render_components="sensors")
                render_count += 1
            reset_env_ids = env_manager.reset_terminated_and_truncated_envs()
            refresh_targets(reset_env_ids)
            apply_targets_to_actions()
            maybe_capture_frame()
            total_steps += 1

        render_count = 0
        start = time.time()
        for _ in range(bench_steps):
            apply_targets_to_actions()
            env_manager.step(actions=actions)
            if total_steps % render_every == 0:
                env_manager.render(render_components="sensors")
                render_count += 1
            reset_env_ids = env_manager.reset_terminated_and_truncated_envs()
            refresh_targets(reset_env_ids)
            # Ensure reset envs do not execute one stale step toward a pre-reset target.
            apply_targets_to_actions()
            maybe_capture_frame()
            total_steps += 1
        elapsed = time.time() - start

    sim_dt = float(env_manager.sim_config.sim.dt)
    metrics = compute_benchmark_metrics(
        bench_steps=bench_steps,
        num_envs=num_envs,
        elapsed=elapsed,
        sim_dt=sim_dt,
        render_count=render_count,
        render_every=render_every,
    )

    del env_manager
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    gif_path = maybe_save_case_gif(
        frames=frames,
        label=robot_name,
        num_envs=num_envs,
        gif_duration_ms=gif_duration_ms,
    )

    return {
        "robot_name": robot_name,
        "controller_name": controller_name,
        "num_envs": num_envs,
        "elapsed_s": elapsed,
        "render_count": render_count,
        "render_every": render_every,
        "gif_path": gif_path,
        **metrics,
    }


def _run_case_subprocess(config_path: str, case_name: str, num_envs: int):
    cmd = [
        sys.executable,
        __file__,
        "--config",
        config_path,
        "--child-case",
        case_name,
        "--num-envs",
        str(num_envs),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Child benchmark failed for case={case_name}, num_envs={num_envs}.\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )

    for line in proc.stdout.splitlines():
        if line.startswith("RESULT_JSON:"):
            return json.loads(line[len("RESULT_JSON:") :].strip())

    raise RuntimeError(
        f"Child benchmark did not emit RESULT_JSON for case={case_name}, num_envs={num_envs}.\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )


def _print_table(depth_rows, rgbd_rows):
    logger.warning("\n=== Matterport Parallel Benchmark: Depth-only vs RGBD ===")
    print(
        f"{'envs':>6} | {'depth FPS':>12} | {'rgbd FPS':>12} | {'d FPS/env':>10} | {'r FPS/env':>10} | {'rgbd/depth':>11} | "
        f"{'depth RTF':>10} | {'rgbd RTF':>10}"
    )
    print("-" * 110)
    for d, r in zip(depth_rows, rgbd_rows):
        ratio = r["fps"] / max(d["fps"], 1.0e-9)
        print(
            f"{d['num_envs']:6d} | {d['fps']:12.2f} | {r['fps']:12.2f} | {d['fps_per_env']:10.2f} | {r['fps_per_env']:10.2f} | {ratio:11.3f} | "
            f"{d['real_time_speedup']:10.2f} | {r['real_time_speedup']:10.2f}"
        )


def _apply_world_config(config: Dict[str, Any]):
    scene_cfg = config.get("scene", {})
    scene_bundle = resolve_scene_bundle(
        MatterportGLBEnvCfg,
        scene_file=scene_cfg.get("file"),
        scene_folder=scene_cfg.get("folder"),
        scene_root=scene_cfg.get("root", "resources/envs"),
        logger=logger,
    )
    spawn_region = apply_spawn_region(
        MatterportGLBEnvCfg,
        spawn_cfg=config.get("spawn_region", {}),
        logger=logger,
        source_name="yaml",
    )
    navmesh_settings = apply_navmesh_config(
        MatterportGLBEnvCfg,
        navmesh_cfg=config.get("navmesh", {}),
        scene_bundle=scene_bundle,
        logger=logger,
    )
    return scene_bundle["scene_file"], spawn_region, navmesh_settings


def _run_single_case_from_child(config_path: str, case_name: str, num_envs: int):
    config = _load_config(config_path)
    random.seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))
    torch.manual_seed(int(config["seed"]))
    torch.cuda.manual_seed_all(int(config["seed"]))

    camera_cfg = config["camera"]
    _configure_camera_classes(
        width=int(camera_cfg["width"]),
        height=int(camera_cfg["height"]),
        max_range=float(camera_cfg["max_range"]),
        enable_lighting=bool(camera_cfg.get("enable_lighting", False)),
    )
    _apply_world_config(config)

    robot_name = config["robots"][case_name]
    row = _run_case(
        robot_name=robot_name,
        controller_name=str(config["controller_name"]),
        num_envs=int(num_envs),
        warmup_steps=int(config["warmup_steps"]),
        bench_steps=int(config["bench_steps"]),
        device=str(config["device"]),
        headless=bool(config["headless"]),
        save_gif=bool(config["gif"].get("save_gifs", False)),
        gif_every=int(config["gif"].get("every", 5)),
        gif_max_frames=int(config["gif"].get("max_frames", 300)),
        gif_duration_ms=int(config["gif"].get("duration_ms", 60)),
        render_every=int(config["render"].get("render_every", 1)),
    )
    print("RESULT_JSON:" + json.dumps(row))


def main():
    parser = argparse.ArgumentParser(description="Matterport depth-vs-rgbd benchmark")
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "configs", "matterport_depth_vs_rgbd.yaml"),
        help="Path to YAML benchmark config.",
    )
    parser.add_argument("--child-case", choices=["depth_only", "shaded_rgbd"], default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    args = parser.parse_args()

    config = _load_config(args.config)

    if args.child_case is not None:
        if args.num_envs is None:
            raise ValueError("--num-envs is required when --child-case is used")
        _run_single_case_from_child(args.config, args.child_case, args.num_envs)
        return

    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    camera_cfg = config["camera"]
    _configure_camera_classes(
        width=int(camera_cfg["width"]),
        height=int(camera_cfg["height"]),
        max_range=float(camera_cfg["max_range"]),
        enable_lighting=bool(camera_cfg.get("enable_lighting", False)),
    )
    selected_scene_file, spawn_region, navmesh_settings = _apply_world_config(config)

    logger.warning(
        "Running benchmark on env counts %s, warmup=%d, bench=%d, resolution=%dx%d, max_range=%.1f",
        config["env_counts"],
        int(config["warmup_steps"]),
        int(config["bench_steps"]),
        int(camera_cfg["width"]),
        int(camera_cfg["height"]),
        float(camera_cfg["max_range"]),
    )

    depth_rows = []
    rgbd_rows = []

    for n in config["env_counts"]:
        logger.warning("Depth-only case: num_envs=%d", n)
        depth_rows.append(_run_case_subprocess(args.config, "depth_only", int(n)))

        logger.warning("Shaded RGBD case: num_envs=%d", n)
        rgbd_rows.append(_run_case_subprocess(args.config, "shaded_rgbd", int(n)))

    _print_table(depth_rows, rgbd_rows)

    payload = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "config_path": os.path.abspath(args.config),
        "settings": {
            "seed": seed,
            "env_counts": config["env_counts"],
            "warmup_steps": int(config["warmup_steps"]),
            "bench_steps": int(config["bench_steps"]),
            "width": int(camera_cfg["width"]),
            "height": int(camera_cfg["height"]),
            "max_range": float(camera_cfg["max_range"]),
            "enable_lighting": bool(camera_cfg.get("enable_lighting", False)),
            "controller_name": str(config["controller_name"]),
            "render_every": int(config["render"].get("render_every", 1)),
            "headless": bool(config["headless"]),
            "device": str(config["device"]),
            "save_gifs": bool(config["gif"].get("save_gifs", False)),
            "gif_every": int(config["gif"].get("every", 5)),
            "gif_max_frames": int(config["gif"].get("max_frames", 300)),
            "gif_duration_ms": int(config["gif"].get("duration_ms", 60)),
            "spawn_region": spawn_region,
            "scene_file": selected_scene_file,
            "navmesh_settings": navmesh_settings,
        },
        "depth_only": depth_rows,
        "shaded_rgbd": rgbd_rows,
    }

    report_path = str(config.get("output", {}).get("report_path", "")).strip()
    if not report_path:
        report_path = os.path.join(
            os.path.dirname(__file__), "stored_data", "matterport_depth_vs_rgbd_benchmark.json"
        )
    out_dir = os.path.dirname(report_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.warning("Saved benchmark report: %s", report_path)


if __name__ == "__main__":
    main()
