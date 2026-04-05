import argparse
import gc
import json
import os
import random
import sys
from typing import Any, Dict

import numpy as np
import yaml

from aerial_gym.benchmark.matterport_spawn_helpers import (
    apply_navmesh_config,
    apply_spawn_region,
    resolve_scene_bundle,
)
from aerial_gym.config.env_config.matterport_glb_env import MatterportGLBEnvCfg
from aerial_gym.sim.sim_builder import SimBuilder
from aerial_gym.utils.logging import CustomLogger
import torch


logger = CustomLogger(__name__)


def _default_config() -> Dict[str, Any]:
    return {
        "seed": 0,
        "headless": True,
        "device": "cuda:0",
        "controller_name": "lee_position_control",
        "scene": {
            "file": None,
            "folder": None,
            "root": "resources/envs/val",
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
        },
    }


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config root must be a mapping in {config_path}")
    cfg = _default_config()
    _deep_update(cfg, loaded)
    return cfg


def _apply_world_config(config: Dict[str, Any]):
    scene_cfg = config.get("scene", {})
    scene_bundle = resolve_scene_bundle(
        MatterportGLBEnvCfg,
        scene_file=scene_cfg.get("file"),
        scene_folder=scene_cfg.get("folder"),
        scene_root=scene_cfg.get("root", "resources/envs"),
        logger=logger,
    )
    apply_spawn_region(
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
    return scene_bundle, navmesh_settings


def _point_inside_triangles_2d(points_xy: torch.Tensor, tri_xy: torch.Tensor) -> torch.Tensor:
    # points_xy: (N, 2), tri_xy: (M, 3, 2) -> (N, M)
    p = points_xy[:, None, :]
    a = tri_xy[None, :, 0, :]
    b = tri_xy[None, :, 1, :]
    c = tri_xy[None, :, 2, :]

    ab = b - a
    bc = c - b
    ca = a - c
    ap = p - a
    bp = p - b
    cp = p - c

    cross1 = ab[..., 0] * ap[..., 1] - ab[..., 1] * ap[..., 0]
    cross2 = bc[..., 0] * bp[..., 1] - bc[..., 1] * bp[..., 0]
    cross3 = ca[..., 0] * cp[..., 1] - ca[..., 1] * cp[..., 0]

    pos = (cross1 >= 0.0) & (cross2 >= 0.0) & (cross3 >= 0.0)
    neg = (cross1 <= 0.0) & (cross2 <= 0.0) & (cross3 <= 0.0)
    return pos | neg


def _min_edge_distance_2d(points_xy: torch.Tensor, tri_xy: torch.Tensor) -> torch.Tensor:
    # points_xy: (N, 2), tri_xy: (M, 3, 2) -> (N, M)
    p = points_xy[:, None, None, :]
    e0 = tri_xy[None, :, 0, :]
    e1 = tri_xy[None, :, 1, :]
    e2 = tri_xy[None, :, 2, :]
    edge_a = torch.stack((e0, e1, e2), dim=2)
    edge_b = torch.stack((e1, e2, e0), dim=2)

    ab = edge_b - edge_a
    ap = p - edge_a
    ab_sq = torch.clamp(torch.sum(ab * ab, dim=3), min=1.0e-12)
    t = torch.sum(ap * ab, dim=3) / ab_sq
    t = torch.clamp(t, 0.0, 1.0)
    closest = edge_a + t[..., None] * ab
    dists = torch.norm(p - closest, dim=3)
    return torch.min(dists, dim=2)[0]


def _build_hold_actions(env_manager, device):
    robot_position = env_manager.global_tensor_dict["robot_position"]
    robot_euler_angles = env_manager.global_tensor_dict.get("robot_euler_angles", None)
    if "env_origins" in env_manager.global_tensor_dict:
        env_origins = env_manager.global_tensor_dict["env_origins"]
    elif hasattr(env_manager, "IGE_env") and hasattr(env_manager.IGE_env, "env_origins"):
        env_origins = torch.tensor(
            env_manager.IGE_env.env_origins,
            dtype=robot_position.dtype,
            device=robot_position.device,
        )
    elif hasattr(env_manager, "env_origins"):
        env_origins = torch.tensor(
            env_manager.env_origins,
            dtype=robot_position.dtype,
            device=robot_position.device,
        )
    else:
        env_origins = torch.zeros_like(robot_position)

    actions = torch.zeros((env_manager.num_envs, 4), device=device)
    actions[:, 0:3] = robot_position - env_origins
    if robot_euler_angles is not None:
        actions[:, 3] = robot_euler_angles[:, 2]
    return actions


def run_debug(
    config_path: str,
    num_envs: int,
    resets: int,
    steps_per_reset: int,
    max_outside_navmesh_rate: float,
    max_edge_violation_rate: float,
    max_crash_rate: float,
):
    config = _load_config(config_path)
    seed = int(config.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    scene_bundle, navmesh_settings = _apply_world_config(config)
    logger.warning(
        "Spawn debug setup: scene=%s navmesh_enabled=%s",
        scene_bundle["scene_file"],
        navmesh_settings is not None,
    )

    robot_name = config.get("robots", {}).get("depth_only", "base_quadrotor_with_camera")
    controller_name = str(config.get("controller_name", "lee_position_control"))
    device = str(config.get("device", "cuda:0"))
    headless = bool(config.get("headless", True))

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

    env_ids = torch.arange(env_manager.num_envs, device=device)
    env_manager.reset_idx(env_ids)

    robot_position = env_manager.global_tensor_dict["robot_position"]
    if "env_origins" in env_manager.global_tensor_dict:
        env_origins = env_manager.global_tensor_dict["env_origins"]
    elif hasattr(env_manager, "IGE_env") and hasattr(env_manager.IGE_env, "env_origins"):
        env_origins = torch.tensor(
            env_manager.IGE_env.env_origins,
            dtype=robot_position.dtype,
            device=robot_position.device,
        )
    elif hasattr(env_manager, "env_origins"):
        env_origins = torch.tensor(
            env_manager.env_origins,
            dtype=robot_position.dtype,
            device=robot_position.device,
        )
    else:
        env_origins = torch.zeros_like(robot_position)
    env_bounds_min = env_manager.global_tensor_dict.get("env_bounds_min", None)
    env_bounds_max = env_manager.global_tensor_dict.get("env_bounds_max", None)

    sampler = getattr(env_manager, "navmesh_sampler", None)
    navmesh_enabled = sampler is not None and sampler.enabled
    edge_padding = float(getattr(sampler.nav_cfg, "edge_padding", 0.0)) if navmesh_enabled else 0.0

    tri_xy = None
    nav_planar_axes = (0, 1)
    if navmesh_enabled:
        nav_planar_axes = tuple(getattr(sampler.navmesh, "planar_axes", (0, 1)))
        nav_verts = sampler.navmesh.pt_vertices.to(device)
        nav_verts_scene = sampler._apply_scene_transform(nav_verts)
        nav_tris = sampler.navmesh.pt_polygons.to(device)
        tri_xy = nav_verts_scene[nav_tris][:, :, list(nav_planar_axes)]

    total_samples = 0
    out_of_bounds_world_count = 0
    out_of_bounds_local_count = 0
    outside_navmesh_count = 0
    edge_violation_count = 0
    crash_after_spawn_count = 0

    with torch.no_grad():
        for reset_idx in range(resets):
            env_manager.reset_idx(env_ids)
            world_pos = robot_position.clone()
            local_pos = world_pos - env_origins

            if env_bounds_min is not None and env_bounds_max is not None:
                world_lower_ok = world_pos >= env_bounds_min
                world_upper_ok = world_pos <= env_bounds_max
                world_in_bounds = torch.logical_and(world_lower_ok, world_upper_ok).all(dim=1)
                out_of_bounds_world_count += int((~world_in_bounds).sum().item())

                local_lower_ok = local_pos >= env_bounds_min
                local_upper_ok = local_pos <= env_bounds_max
                local_in_bounds = torch.logical_and(local_lower_ok, local_upper_ok).all(dim=1)
                out_of_bounds_local_count += int((~local_in_bounds).sum().item())

            if navmesh_enabled and tri_xy is not None:
                pts_xy = local_pos[:, list(nav_planar_axes)]
                inside_any = _point_inside_triangles_2d(pts_xy, tri_xy).any(dim=1)
                outside_navmesh_count += int((~inside_any).sum().item())

                min_edge_dist_per_tri = _min_edge_distance_2d(pts_xy, tri_xy)
                inside_mask = _point_inside_triangles_2d(pts_xy, tri_xy)
                inf = torch.full_like(min_edge_dist_per_tri, float("inf"))
                min_inside_edge = torch.where(inside_mask, min_edge_dist_per_tri, inf).min(dim=1)[0]
                violated = torch.logical_and(inside_any, min_inside_edge < (edge_padding - 1.0e-4))
                edge_violation_count += int(violated.sum().item())

            total_samples += env_manager.num_envs

            any_crashed = torch.zeros(env_manager.num_envs, dtype=torch.bool, device=device)
            for _ in range(steps_per_reset):
                actions = _build_hold_actions(env_manager, device=device)
                env_manager.step(actions=actions)
                any_crashed |= env_manager.global_tensor_dict["crashes"].bool()
                env_manager.reset_terminated_and_truncated_envs()
            crash_after_spawn_count += int(any_crashed.sum().item())

            if (reset_idx + 1) % max(resets // 10, 1) == 0:
                logger.warning(
                    "Spawn debug progress: %d/%d resets",
                    reset_idx + 1,
                    resets,
                )

    results = {
        "config_path": config_path,
        "num_envs": env_manager.num_envs,
        "resets": resets,
        "steps_per_reset": steps_per_reset,
        "total_samples": total_samples,
        "navmesh_enabled": navmesh_enabled,
        "edge_padding": edge_padding,
        "out_of_bounds_world_count": out_of_bounds_world_count,
        "out_of_bounds_local_count": out_of_bounds_local_count,
        "outside_navmesh_count": outside_navmesh_count,
        "edge_violation_count": edge_violation_count,
        "crash_after_spawn_count": crash_after_spawn_count,
        "out_of_bounds_world_rate": out_of_bounds_world_count / max(total_samples, 1),
        "out_of_bounds_local_rate": out_of_bounds_local_count / max(total_samples, 1),
        "outside_navmesh_rate": outside_navmesh_count / max(total_samples, 1),
        "edge_violation_rate": edge_violation_count / max(total_samples, 1),
        "crash_after_spawn_rate": crash_after_spawn_count / max(total_samples, 1),
        "thresholds": {
            "max_outside_navmesh_rate": max_outside_navmesh_rate,
            "max_edge_violation_rate": max_edge_violation_rate,
            "max_crash_rate": max_crash_rate,
        },
    }

    print("RESULT_JSON:" + json.dumps(results))
    logger.warning("Spawn debug summary:\n%s", json.dumps(results, indent=2))

    failed = False
    if results["outside_navmesh_rate"] > max_outside_navmesh_rate:
        failed = True
        logger.error(
            "outside_navmesh_rate %.4f exceeded threshold %.4f",
            results["outside_navmesh_rate"],
            max_outside_navmesh_rate,
        )
    if results["edge_violation_rate"] > max_edge_violation_rate:
        failed = True
        logger.error(
            "edge_violation_rate %.4f exceeded threshold %.4f",
            results["edge_violation_rate"],
            max_edge_violation_rate,
        )
    if results["crash_after_spawn_rate"] > max_crash_rate:
        failed = True
        logger.error(
            "crash_after_spawn_rate %.4f exceeded threshold %.4f",
            results["crash_after_spawn_rate"],
            max_crash_rate,
        )

    del env_manager
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return 1 if failed else 0


def main():
    parser = argparse.ArgumentParser(description="Deep spawn validator for Matterport navmesh spawning")
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "configs", "matterport_depth_vs_rgbd.yaml"),
        help="Path to YAML config.",
    )
    parser.add_argument("--num-envs", type=int, default=8, help="Number of parallel envs to test.")
    parser.add_argument("--resets", type=int, default=120, help="Number of reset cycles to test.")
    parser.add_argument(
        "--steps-per-reset",
        type=int,
        default=4,
        help="Simulation steps run after each reset while holding position.",
    )
    parser.add_argument(
        "--max-outside-navmesh-rate",
        type=float,
        default=0.0,
        help="Fail if outside-navmesh spawn rate exceeds this threshold.",
    )
    parser.add_argument(
        "--max-edge-violation-rate",
        type=float,
        default=0.01,
        help="Fail if edge-padding violation rate exceeds this threshold.",
    )
    parser.add_argument(
        "--max-crash-rate",
        type=float,
        default=0.10,
        help="Fail if immediate post-spawn crash rate exceeds this threshold.",
    )
    args = parser.parse_args()

    # Isaac Gym also parses process args; keep only script name to avoid noisy unknown-arg logs.
    sys.argv = [sys.argv[0]]

    raise SystemExit(
        run_debug(
            config_path=args.config,
            num_envs=int(args.num_envs),
            resets=int(args.resets),
            steps_per_reset=int(args.steps_per_reset),
            max_outside_navmesh_rate=float(args.max_outside_navmesh_rate),
            max_edge_violation_rate=float(args.max_edge_violation_rate),
            max_crash_rate=float(args.max_crash_rate),
        )
    )


if __name__ == "__main__":
    main()