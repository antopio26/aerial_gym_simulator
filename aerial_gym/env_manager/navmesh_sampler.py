import os

import torch

from aerial_gym import AERIAL_GYM_DIRECTORY
from aerial_gym.utils.standalone_navmesh_parser import StandaloneNavMesh


class NavMeshSpawnSampler:
    """Env-level navmesh sampler for safe robot spawn placement."""

    def __init__(self, env_cfg, env_origins, device, logger):
        self.env_cfg = env_cfg
        self.device = device
        self.logger = logger
        self.env_origins = torch.tensor(env_origins, dtype=torch.float32, device=self.device)

        self.nav_cfg = getattr(self.env_cfg, "navmesh_sampling", None)
        self.enabled = self.nav_cfg is not None and bool(getattr(self.nav_cfg, "enable", False))
        self.navmesh = None

        if not self.enabled:
            return

        navmesh_file = self._resolve_navmesh_file()
        if navmesh_file is None:
            self.logger.warning("Navmesh sampling enabled but navmesh file could not be resolved.")
            self.enabled = False
            return

        self.navmesh = StandaloneNavMesh(navmesh_file)
        if self.navmesh.pt_vertices is None or self.navmesh.pt_polygons is None:
            self.logger.warning(f"Navmesh has no valid polygons: {navmesh_file}")
            self.navmesh = None
            self.enabled = False
            return

        self.navmesh.pt_vertices = self.navmesh.pt_vertices.to(self.device)
        self.navmesh.pt_polygons = self.navmesh.pt_polygons.to(self.device)
        self.navmesh.pt_poly_areas = self.navmesh.pt_poly_areas.to(self.device)
        self.logger.info(f"Enabled env navmesh spawn sampler from: {navmesh_file}")

    def _resolve_abs(self, path):
        if os.path.isabs(path):
            return path
        candidate_repo = os.path.join(AERIAL_GYM_DIRECTORY, path)
        if os.path.exists(candidate_repo):
            return candidate_repo
        return os.path.join(os.getcwd(), path)

    def _resolve_navmesh_file(self):
        navmesh_file = getattr(self.nav_cfg, "navmesh_file", None)
        if navmesh_file:
            candidate = self._resolve_abs(navmesh_file)
            if os.path.exists(candidate):
                return candidate
            return None

        static_scene_cfg = getattr(self.env_cfg, "static_scene", None)
        if static_scene_cfg is None:
            return None

        scene_file = getattr(static_scene_cfg, "file", None)
        if scene_file is None:
            return None

        scene_abs = self._resolve_abs(scene_file)
        if not os.path.exists(scene_abs):
            return None

        scene_dir = os.path.dirname(scene_abs)
        scene_stem = os.path.splitext(os.path.basename(scene_abs))[0]
        candidates = [
            os.path.join(scene_dir, f"{scene_stem}.basis.navmesh"),
            os.path.join(scene_dir, f"{scene_stem}.navmesh"),
        ]
        for cand in candidates:
            if os.path.exists(cand):
                return cand

        navmesh_files = sorted(
            [
                os.path.join(scene_dir, fname)
                for fname in os.listdir(scene_dir)
                if fname.endswith(".navmesh")
            ]
        )
        if navmesh_files:
            return navmesh_files[0]
        return None

    def _apply_scene_transform(self, points):
        if getattr(self.nav_cfg, "inherit_scene_transform", True):
            static_scene_cfg = getattr(self.env_cfg, "static_scene", None)
            if static_scene_cfg is not None:
                scale = float(getattr(static_scene_cfg, "scale", 1.0))
                translation = torch.tensor(
                    getattr(static_scene_cfg, "translation", [0.0, 0.0, 0.0]),
                    dtype=torch.float32,
                    device=self.device,
                )
                points = points * scale + translation

        navmesh_scale = float(getattr(self.nav_cfg, "navmesh_scale", 1.0))
        navmesh_translation = torch.tensor(
            getattr(self.nav_cfg, "navmesh_translation", [0.0, 0.0, 0.0]),
            dtype=torch.float32,
            device=self.device,
        )
        return points * navmesh_scale + navmesh_translation

    def sample_world_points(self, env_ids, height_offset_range, bounds_min=None, bounds_max=None):
        if not self.enabled or self.navmesh is None or len(env_ids) == 0:
            return None

        env_ids = env_ids.to(dtype=torch.long, device=self.device)
        edge_padding = float(getattr(self.nav_cfg, "edge_padding", 0.0))
        oversample_factor = int(getattr(self.nav_cfg, "oversample_factor", 4))
        max_resample_rounds = int(getattr(self.nav_cfg, "max_resample_rounds", 8))
        strict_edge_padding = bool(getattr(self.nav_cfg, "strict_edge_padding", True))
        min_edge_padding_ratio = float(getattr(self.nav_cfg, "min_edge_padding_ratio", 0.35))
        padding_relaxation_factor = float(
            getattr(self.nav_cfg, "padding_relaxation_factor", 0.70)
        )
        max_padding_relax_rounds = int(getattr(self.nav_cfg, "max_padding_relax_rounds", 3))
        enforce_bounds = bool(getattr(self.nav_cfg, "enforce_env_bounds", True))
        max_bound_resample_rounds = int(getattr(self.nav_cfg, "max_bound_resample_rounds", 5))

        bounds_min_env = None
        bounds_max_env = None
        if bounds_min is not None and bounds_max is not None:
            bounds_min_env = bounds_min.to(self.device)[env_ids]
            bounds_max_env = bounds_max.to(self.device)[env_ids]

        candidate_count = max(oversample_factor * 4, 8)
        selected_points_world = []
        fallback_count = 0

        for i, env_id in enumerate(env_ids):
            env_origin = self.env_origins[env_id]
            selected = None

            for _ in range(max_bound_resample_rounds):
                points_local = self.navmesh.sample_points_with_padding(
                    count=candidate_count,
                    height_offset=tuple(height_offset_range),
                    edge_padding=edge_padding,
                    oversample_factor=oversample_factor,
                    max_resample_rounds=max_resample_rounds,
                    strict_edge_padding=strict_edge_padding,
                    min_edge_padding_ratio=min_edge_padding_ratio,
                    padding_relaxation_factor=padding_relaxation_factor,
                    max_padding_relax_rounds=max_padding_relax_rounds,
                ).to(self.device)
                points_local_scene = self._apply_scene_transform(points_local)

                if enforce_bounds and bounds_min_env is not None and bounds_max_env is not None:
                    env_bmin = bounds_min_env[i]
                    env_bmax = bounds_max_env[i]

                    points_world = points_local_scene + env_origin.unsqueeze(0)
                    in_world_bounds = torch.logical_and(
                        points_world >= env_bmin.unsqueeze(0),
                        points_world <= env_bmax.unsqueeze(0),
                    ).all(dim=1)

                    in_local_bounds = torch.logical_and(
                        points_local_scene >= env_bmin.unsqueeze(0),
                        points_local_scene <= env_bmax.unsqueeze(0),
                    ).all(dim=1)

                    in_bounds = torch.logical_or(in_world_bounds, in_local_bounds)
                    kept_local = points_local_scene[in_bounds]
                else:
                    kept_local = points_local_scene

                if kept_local.shape[0] > 0:
                    pick = torch.randint(0, kept_local.shape[0], (1,), device=self.device)
                    selected = kept_local[pick[0]] + env_origin
                    break

            if selected is None:
                fallback_count += 1
                points_local = self.navmesh.sample_points_with_padding(
                    count=1,
                    height_offset=tuple(height_offset_range),
                    edge_padding=edge_padding,
                    oversample_factor=oversample_factor,
                    max_resample_rounds=max_resample_rounds,
                    strict_edge_padding=strict_edge_padding,
                    min_edge_padding_ratio=min_edge_padding_ratio,
                    padding_relaxation_factor=padding_relaxation_factor,
                    max_padding_relax_rounds=max_padding_relax_rounds,
                ).to(self.device)
                selected = self._apply_scene_transform(points_local)[0] + env_origin

            selected_points_world.append(selected.unsqueeze(0))

        if fallback_count > 0:
            self.logger.warning(
                "Navmesh spawn sampling fallback used for %d/%d envs.",
                fallback_count,
                len(env_ids),
            )

        return torch.cat(selected_points_world, dim=0)

    def apply_spawn(self, robot_state_tensor, env_ids, env_bounds_min=None, env_bounds_max=None):
        if not self.enabled or self.navmesh is None or len(env_ids) == 0:
            return

        spawn_h = getattr(self.nav_cfg, "spawn_height_offset_range", [0.0, 0.0])
        spawn_points = self.sample_world_points(
            env_ids=env_ids,
            height_offset_range=spawn_h,
            bounds_min=env_bounds_min,
            bounds_max=env_bounds_max,
        )
        if spawn_points is None:
            return

        env_ids_dev = env_ids.to(dtype=torch.long, device=robot_state_tensor.device)
        robot_state_tensor[env_ids_dev, 0:3] = spawn_points.to(robot_state_tensor.device)

        if bool(getattr(self.nav_cfg, "zero_velocity_on_spawn", True)):
            robot_state_tensor[env_ids_dev, 7:13] = 0.0
