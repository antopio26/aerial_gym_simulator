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
from aerial_gym.benchmark.matterport_spawn_helpers import apply_spawn_region
from aerial_gym.config.sensor_config.camera_config.base_depth_camera_config import (
    BaseDepthCameraConfig,
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
        "env_name": "env_with_obstacles",
        "robot_name": "base_quadrotor_with_camera",
        "camera": {
            "width": 240,
            "height": 135,
            "max_range": 80.0,
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
        "spawn_region": {},
        "output": {
            "report_path": "aerial_gym/benchmark/stored_data/depth_only_benchmark.json",
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


def _configure_depth_camera_class(width: int, height: int, max_range: float):
    BaseDepthCameraConfig.width = width
    BaseDepthCameraConfig.height = height
    BaseDepthCameraConfig.max_range = max_range
    BaseDepthCameraConfig.segmentation_camera = False
    BaseDepthCameraConfig.calculate_depth = True


def _apply_spawn_region_config(env_name: str, spawn_cfg: dict):
    import aerial_gym.env_manager  # noqa: F401
    from aerial_gym.registry.env_registry import env_config_registry

    cfg_cls = env_config_registry.get_env_config(env_name)
    return apply_spawn_region(cfg_cls, spawn_cfg=spawn_cfg, logger=logger, source_name="yaml")


def _run_case(
    env_name: str,
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
        env_name=env_name,
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

    def maybe_capture_frame():
        nonlocal total_steps
        if (not save_gif) or gif_every <= 0 or len(frames) >= gif_max_frames:
            return
        if total_steps % gif_every != 0:
            return
        frames.append(collect_grid_frame(env_manager))

    with torch.no_grad():
        for _ in range(warmup_steps):
            actions[:, 0:3] = robot_position
            if robot_euler_angles is not None:
                actions[:, 3] = robot_euler_angles[:, 2]
            else:
                actions[:, 3] = 0.0
            env_manager.step(actions=actions)
            if total_steps % render_every == 0:
                env_manager.render(render_components="sensors")
                render_count += 1
            env_manager.reset_terminated_and_truncated_envs()
            maybe_capture_frame()
            total_steps += 1

        render_count = 0
        start = time.time()
        for _ in range(bench_steps):
            actions[:, 0:3] = robot_position
            if robot_euler_angles is not None:
                actions[:, 3] = robot_euler_angles[:, 2]
            else:
                actions[:, 3] = 0.0
            env_manager.step(actions=actions)
            if total_steps % render_every == 0:
                env_manager.render(render_components="sensors")
                render_count += 1
            env_manager.reset_terminated_and_truncated_envs()
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
        label=f"depth_only_{env_name}_{robot_name}",
        num_envs=num_envs,
        gif_duration_ms=gif_duration_ms,
    )

    return {
        "env_name": env_name,
        "robot_name": robot_name,
        "controller_name": controller_name,
        "num_envs": num_envs,
        "elapsed_s": elapsed,
        "render_count": render_count,
        "render_every": render_every,
        "gif_path": gif_path,
        **metrics,
    }


def _run_case_subprocess(config_path: str, num_envs: int):
    cmd = [
        sys.executable,
        __file__,
        "--config",
        config_path,
        "--child-case",
        "depth_only",
        "--num-envs",
        str(num_envs),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Child benchmark failed for num_envs={num_envs}.\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )

    for line in proc.stdout.splitlines():
        if line.startswith("RESULT_JSON:"):
            return json.loads(line[len("RESULT_JSON:"):].strip())

    raise RuntimeError(
        f"Child benchmark did not emit RESULT_JSON for num_envs={num_envs}.\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )


def _print_table(rows):
    logger.warning("\n=== Depth-only Parallel Benchmark ===")
    print(
        f"{'envs':>6} | {'step FPS':>12} | {'render FPS':>12} | {'step RTF':>10} | {'render RTF':>10} | {'elapsed(s)':>12}"
    )
    print("-" * 76)
    for r in rows:
        print(
            f"{r['num_envs']:6d} | {r['fps']:12.2f} | {r['rendered_fps']:12.2f} | {r['real_time_speedup']:10.2f} | {r['rendered_real_time_speedup']:10.2f} | {r['elapsed_s']:12.3f}"
        )


def _run_single_case_from_child(config_path: str, num_envs: int):
    config = _load_config(config_path)
    random.seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))
    torch.manual_seed(int(config["seed"]))
    torch.cuda.manual_seed_all(int(config["seed"]))

    camera_cfg = config["camera"]
    _configure_depth_camera_class(
        width=int(camera_cfg["width"]),
        height=int(camera_cfg["height"]),
        max_range=float(camera_cfg["max_range"]),
    )
    env_name = str(config["env_name"])
    _apply_spawn_region_config(env_name, config.get("spawn_region", {}))

    row = _run_case(
        env_name=env_name,
        robot_name=str(config["robot_name"]),
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
    parser = argparse.ArgumentParser(description="Depth-only parallel benchmark")
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "configs", "matterport_depth_only.yaml"),
        help="Path to YAML benchmark config.",
    )
    parser.add_argument("--child-case", choices=["depth_only"], default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    args = parser.parse_args()

    config = _load_config(args.config)

    if args.child_case is not None:
        if args.num_envs is None:
            raise ValueError("--num-envs is required when --child-case is used")
        _run_single_case_from_child(args.config, args.num_envs)
        return

    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    camera_cfg = config["camera"]
    _configure_depth_camera_class(
        width=int(camera_cfg["width"]),
        height=int(camera_cfg["height"]),
        max_range=float(camera_cfg["max_range"]),
    )
    env_name = str(config["env_name"])
    robot_name = str(config["robot_name"])
    spawn_region = _apply_spawn_region_config(env_name, config.get("spawn_region", {}))

    logger.warning(
        "Running depth-only benchmark: env=%s, robot=%s, env_counts=%s, warmup=%d, bench=%d, res=%dx%d, max_range=%.1f",
        env_name,
        robot_name,
        config["env_counts"],
        int(config["warmup_steps"]),
        int(config["bench_steps"]),
        int(camera_cfg["width"]),
        int(camera_cfg["height"]),
        float(camera_cfg["max_range"]),
    )

    rows = []
    for n in config["env_counts"]:
        logger.warning("Depth-only case: num_envs=%d", n)
        rows.append(_run_case_subprocess(args.config, int(n)))

    _print_table(rows)

    payload = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "config_path": os.path.abspath(args.config),
        "settings": {
            "env_name": env_name,
            "robot_name": robot_name,
            "controller_name": str(config["controller_name"]),
            "env_counts": config["env_counts"],
            "warmup_steps": int(config["warmup_steps"]),
            "bench_steps": int(config["bench_steps"]),
            "width": int(camera_cfg["width"]),
            "height": int(camera_cfg["height"]),
            "max_range": float(camera_cfg["max_range"]),
            "headless": bool(config["headless"]),
            "device": str(config["device"]),
            "render_every": int(config["render"].get("render_every", 1)),
            "save_gifs": bool(config["gif"].get("save_gifs", False)),
            "gif_every": int(config["gif"].get("every", 5)),
            "gif_max_frames": int(config["gif"].get("max_frames", 300)),
            "gif_duration_ms": int(config["gif"].get("duration_ms", 60)),
            "spawn_region": spawn_region,
        },
        "depth_only": rows,
    }

    report_path = str(config.get("output", {}).get("report_path", "")).strip()
    if not report_path:
        report_path = os.path.join(
            os.path.dirname(__file__), "stored_data", "depth_only_benchmark.json"
        )
    out_dir = os.path.dirname(report_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.warning("Saved depth-only benchmark report: %s", report_path)


if __name__ == "__main__":
    main()
