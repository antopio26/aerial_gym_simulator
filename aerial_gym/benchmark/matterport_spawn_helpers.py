import os
from typing import Optional

import numpy as np
import trimesh as tm


def _resolve_scene_folder_path(scene_folder: str, scene_root: str = "resources/envs") -> str:
    if os.path.isabs(scene_folder) and os.path.isdir(scene_folder):
        return scene_folder
    if os.path.isdir(scene_folder):
        return scene_folder

    rooted_candidate = os.path.join(scene_root, scene_folder)
    if os.path.isdir(rooted_candidate):
        return rooted_candidate

    raise FileNotFoundError(
        f"Configured scene folder does not exist: {scene_folder} (also checked {rooted_candidate})"
    )


def _find_scene_glb(scene_folder_path: str) -> str:
    glb_candidates = sorted(
        os.path.join(scene_folder_path, fname)
        for fname in os.listdir(scene_folder_path)
        if fname.lower().endswith(".glb")
    )
    if len(glb_candidates) == 0:
        raise FileNotFoundError(f"No .glb file found in scene folder: {scene_folder_path}")
    return glb_candidates[0]


def _find_navmesh_for_scene(scene_file: str) -> Optional[str]:
    scene_folder = os.path.dirname(scene_file)
    scene_stem = os.path.splitext(os.path.basename(scene_file))[0]

    exact_name = os.path.join(scene_folder, f"{scene_stem}.navmesh")
    if os.path.isfile(exact_name):
        return exact_name

    navmesh_candidates = sorted(
        os.path.join(scene_folder, fname)
        for fname in os.listdir(scene_folder)
        if fname.lower().endswith(".navmesh")
    )
    if len(navmesh_candidates) > 0:
        return navmesh_candidates[0]
    return None


def resolve_scene_bundle(
    cfg_cls,
    scene_file: Optional[str] = None,
    scene_folder: Optional[str] = None,
    scene_root: str = "resources/envs",
    logger=None,
) -> dict:
    if scene_folder:
        resolved_scene_folder = _resolve_scene_folder_path(scene_folder, scene_root=scene_root)
        selected_scene_file = _find_scene_glb(resolved_scene_folder)
        cfg_cls.static_scene.file = selected_scene_file
        navmesh_file = _find_navmesh_for_scene(selected_scene_file)
        if logger is not None:
            logger.warning(
                "Resolved Matterport scene folder '%s' -> scene file: %s",
                scene_folder,
                selected_scene_file,
            )
    else:
        selected_scene_file = resolve_scene_file(cfg_cls, scene_file=scene_file, logger=logger)
        resolved_scene_folder = os.path.dirname(selected_scene_file)
        navmesh_file = _find_navmesh_for_scene(selected_scene_file)

    return {
        "scene_file": selected_scene_file,
        "scene_folder": resolved_scene_folder,
        "navmesh_file": navmesh_file,
    }


def parse_vec3(raw: str, name: str) -> np.ndarray:
    vals = [v.strip() for v in raw.split(",") if v.strip() != ""]
    if len(vals) != 3:
        raise ValueError(f"{name} must contain exactly 3 comma-separated values, got: {raw}")
    return np.array([float(vals[0]), float(vals[1]), float(vals[2])], dtype=np.float32)


def resolve_scene_file(cfg_cls, scene_file: Optional[str] = None, logger=None) -> str:
    if scene_file:
        if not os.path.exists(scene_file):
            raise FileNotFoundError(f"Configured scene file does not exist: {scene_file}")
        cfg_cls.static_scene.file = scene_file
        return scene_file

    default_scene = cfg_cls.static_scene.file
    if os.path.exists(default_scene):
        return default_scene

    glb_candidates = []
    for root, _, files in os.walk("resources/envs"):
        for fname in files:
            if fname.endswith(".glb"):
                glb_candidates.append(os.path.join(root, fname))

    if len(glb_candidates) == 0:
        raise FileNotFoundError(
            "No .glb files found under resources/envs and default static scene path is invalid."
        )

    glb_candidates.sort()
    cfg_cls.static_scene.file = glb_candidates[0]
    if logger is not None:
        logger.warning(
            "Default scene file was invalid. Auto-selected scene file: %s",
            cfg_cls.static_scene.file,
        )
    return cfg_cls.static_scene.file


def apply_navmesh_config(
    cfg_cls,
    navmesh_cfg: Optional[dict] = None,
    scene_bundle: Optional[dict] = None,
    logger=None,
) -> Optional[dict]:
    navmesh_cfg = navmesh_cfg or {}
    has_explicit_enabled = "enabled" in navmesh_cfg and navmesh_cfg.get("enabled") is not None

    navmesh_file = navmesh_cfg.get("navmesh_file")
    if (not navmesh_file) and scene_bundle is not None:
        navmesh_file = scene_bundle.get("navmesh_file")

    if has_explicit_enabled:
        enabled = bool(navmesh_cfg.get("enabled"))
    else:
        enabled = navmesh_file is not None

    cfg_cls.navmesh_sampling.enable = enabled
    if not enabled:
        return None

    cfg_cls.navmesh_sampling.navmesh_file = navmesh_file
    cfg_cls.navmesh_sampling.edge_padding = float(navmesh_cfg.get("edge_padding", 0.30))
    cfg_cls.navmesh_sampling.enforce_env_bounds = bool(navmesh_cfg.get("enforce_env_bounds", True))
    cfg_cls.navmesh_sampling.max_bound_resample_rounds = int(
        navmesh_cfg.get("max_bound_resample_rounds", 5)
    )
    spawn_height_range = navmesh_cfg.get("spawn_height_offset_range", [0.15, 0.40])
    if len(spawn_height_range) != 2:
        raise ValueError(
            "navmesh.spawn_height_offset_range must contain exactly two values [min, max]."
        )
    cfg_cls.navmesh_sampling.spawn_height_offset_range = [
        float(spawn_height_range[0]),
        float(spawn_height_range[1]),
    ]
    cfg_cls.navmesh_sampling.zero_velocity_on_spawn = bool(
        navmesh_cfg.get("zero_velocity_on_spawn", True)
    )

    if logger is not None:
        logger.warning(
            "Enabled env-level navmesh spawn sampling: file=%s edge_padding=%.2f",
            cfg_cls.navmesh_sampling.navmesh_file,
            cfg_cls.navmesh_sampling.edge_padding,
        )

    return {
        "enabled": True,
        "navmesh_file": cfg_cls.navmesh_sampling.navmesh_file,
        "edge_padding": cfg_cls.navmesh_sampling.edge_padding,
        "spawn_height_range": cfg_cls.navmesh_sampling.spawn_height_offset_range,
        "enforce_env_bounds": cfg_cls.navmesh_sampling.enforce_env_bounds,
    }


def apply_spawn_region(
    cfg_cls,
    spawn_cfg: Optional[dict] = None,
    logger=None,
    source_name: str = "config",
) -> Optional[dict]:
    spawn_cfg = spawn_cfg or {}
    lower = None
    upper = None

    if bool(spawn_cfg.get("from_scene_bounds", False)):
        scene_file = cfg_cls.static_scene.file
        scene = tm.load(scene_file, force="scene")
        bounds = np.asarray(scene.bounds, dtype=np.float32)
        scale = float(getattr(cfg_cls.static_scene, "scale", 1.0))
        translation = np.asarray(
            getattr(cfg_cls.static_scene, "translation", [0.0, 0.0, 0.0]), dtype=np.float32
        )
        bounds = bounds * scale + translation[None, :]

        xy_margin = float(spawn_cfg.get("scene_xy_margin", 0.4))
        z_min_offset = float(spawn_cfg.get("scene_z_min_offset", 1.0))
        z_max_offset = float(spawn_cfg.get("scene_z_max_offset", 0.6))

        lower = np.array(
            [
                bounds[0, 0] + xy_margin,
                bounds[0, 1] + xy_margin,
                bounds[0, 2] + z_min_offset,
            ],
            dtype=np.float32,
        )
        upper = np.array(
            [
                bounds[1, 0] - xy_margin,
                bounds[1, 1] - xy_margin,
                bounds[1, 2] + z_max_offset,
            ],
            dtype=np.float32,
        )

    center = spawn_cfg.get("center")
    bounds = spawn_cfg.get("bounds", spawn_cfg.get("half_extents"))
    if center is not None and bounds is not None:
        if len(center) != 3 or len(bounds) != 3:
            raise ValueError("spawn_region.center and spawn_region.bounds must be length-3 lists.")
        center_vec = np.array(center, dtype=np.float32)
        half_vec = np.array(bounds, dtype=np.float32)
        lower = center_vec - half_vec
        upper = center_vec + half_vec

    lower_cfg = spawn_cfg.get("lower")
    upper_cfg = spawn_cfg.get("upper")
    if lower_cfg is not None and upper_cfg is not None:
        if len(lower_cfg) != 3 or len(upper_cfg) != 3:
            raise ValueError("spawn_region.lower and spawn_region.upper must be length-3 lists.")
        lower = np.array(lower_cfg, dtype=np.float32)
        upper = np.array(upper_cfg, dtype=np.float32)

    if lower is None or upper is None:
        return None

    if np.any(upper <= lower):
        raise ValueError(
            f"Invalid spawn bounds from {source_name}: lower={lower.tolist()} upper={upper.tolist()}"
        )

    cfg_cls.env.lower_bound_min = lower.tolist()
    cfg_cls.env.lower_bound_max = lower.tolist()
    cfg_cls.env.upper_bound_min = upper.tolist()
    cfg_cls.env.upper_bound_max = upper.tolist()

    if logger is not None:
        logger.warning(
            "Configured spawn region from %s: lower=%s upper=%s",
            source_name,
            np.array2string(lower, precision=3),
            np.array2string(upper, precision=3),
        )

    return {
        "lower": lower.tolist(),
        "upper": upper.tolist(),
        "source": source_name,
    }


def resolve_scene_file_from_env(cfg_cls, prefix: str, logger=None) -> str:
    return resolve_scene_bundle_from_env(cfg_cls, prefix=prefix, logger=logger)["scene_file"]


def resolve_scene_bundle_from_env(cfg_cls, prefix: str, logger=None) -> dict:
    scene_file = os.getenv(f"{prefix}_SCENE_FILE", "") or None
    scene_folder = os.getenv(f"{prefix}_SCENE_FOLDER", "") or None
    scene_root = os.getenv(f"{prefix}_SCENE_ROOT", "resources/envs")

    return resolve_scene_bundle(
        cfg_cls,
        scene_file=scene_file,
        scene_folder=scene_folder,
        scene_root=scene_root,
        logger=logger,
    )


def apply_navmesh_config_from_env(
    cfg_cls,
    prefix: str,
    scene_bundle: Optional[dict] = None,
    logger=None,
) -> Optional[dict]:
    enabled_raw = os.getenv(f"{prefix}_USE_NAVMESH", "")
    if enabled_raw == "":
        enabled_value = None
    else:
        enabled_value = enabled_raw == "1"

    navmesh_cfg = {
        "enabled": enabled_value,
        "navmesh_file": os.getenv(f"{prefix}_NAVMESH_FILE", "") or None,
        "edge_padding": float(os.getenv(f"{prefix}_NAVMESH_EDGE_PADDING", "0.30")),
        "enforce_env_bounds": os.getenv(f"{prefix}_NAVMESH_ENFORCE_ENV_BOUNDS", "1") == "1",
        "max_bound_resample_rounds": int(
            os.getenv(f"{prefix}_NAVMESH_MAX_BOUND_RESAMPLE_ROUNDS", "5")
        ),
        "spawn_height_offset_range": [
            float(os.getenv(f"{prefix}_NAVMESH_SPAWN_HEIGHT_MIN", "0.15")),
            float(os.getenv(f"{prefix}_NAVMESH_SPAWN_HEIGHT_MAX", "0.40")),
        ],
        "zero_velocity_on_spawn": os.getenv(f"{prefix}_NAVMESH_ZERO_VELOCITY_ON_SPAWN", "1")
        == "1",
    }
    return apply_navmesh_config(
        cfg_cls,
        navmesh_cfg=navmesh_cfg,
        scene_bundle=scene_bundle,
        logger=logger,
    )


def apply_spawn_region_from_env(cfg_cls, prefix: str, logger=None) -> Optional[dict]:
    spawn_cfg = {
        "from_scene_bounds": os.getenv(f"{prefix}_SPAWN_FROM_SCENE_BOUNDS", "0") == "1",
        "scene_xy_margin": float(os.getenv(f"{prefix}_SCENE_XY_MARGIN", "0.4")),
        "scene_z_min_offset": float(os.getenv(f"{prefix}_SCENE_Z_MIN_OFFSET", "1.0")),
        "scene_z_max_offset": float(os.getenv(f"{prefix}_SCENE_Z_MAX_OFFSET", "0.6")),
    }

    center_raw = os.getenv(f"{prefix}_SPAWN_CENTER", None)
    bounds_raw = os.getenv(f"{prefix}_SPAWN_BOUNDS", os.getenv(f"{prefix}_SPAWN_HALF_EXTENTS", None))
    if center_raw is not None and bounds_raw is not None:
        spawn_cfg["center"] = parse_vec3(center_raw, f"{prefix}_SPAWN_CENTER").tolist()
        spawn_cfg["bounds"] = parse_vec3(bounds_raw, f"{prefix}_SPAWN_BOUNDS").tolist()

    lower_raw = os.getenv(f"{prefix}_SPAWN_LOWER", None)
    upper_raw = os.getenv(f"{prefix}_SPAWN_UPPER", None)
    if lower_raw is not None and upper_raw is not None:
        spawn_cfg["lower"] = parse_vec3(lower_raw, f"{prefix}_SPAWN_LOWER").tolist()
        spawn_cfg["upper"] = parse_vec3(upper_raw, f"{prefix}_SPAWN_UPPER").tolist()

    return apply_spawn_region(cfg_cls, spawn_cfg=spawn_cfg, logger=logger, source_name=prefix)
