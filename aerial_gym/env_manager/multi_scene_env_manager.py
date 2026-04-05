"""Multi-scene environment manager.

Loads N coexisting scenes into a single IsaacGym simulation, placing each at
a world-space offset so they don't overlap.  Envs (drones) are assigned to
scenes round-robin and each env gets per-scene collision, navmesh, and warp
rendering.

All envs have env_spacing=0 (env_origins at the world origin).  Spatial
separation is achieved entirely through scene offsets applied to collision
meshes, warp render meshes, and navmesh sample points.
"""

import numpy as np
import torch

from aerial_gym.env_manager.env_manager import EnvManager
from aerial_gym.env_manager.static_scene import StaticSceneGLB
from aerial_gym.env_manager.scene_pool import SceneInfo
from aerial_gym.utils.standalone_navmesh_parser import StandaloneNavMesh
from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger("multi_scene_env_manager")


class _SceneSlot:
    """Runtime data for one loaded scene."""
    __slots__ = ("info", "static_scene", "navmesh", "offset", "bounds_min", "bounds_max")

    def __init__(self):
        self.info = None          # SceneInfo
        self.static_scene = None  # StaticSceneGLB
        self.navmesh = None       # StandaloneNavMesh or None
        self.offset = None        # (3,) np array — world-space translation
        self.bounds_min = None    # (3,) np array — scene AABB min (with offset)
        self.bounds_max = None    # (3,) np array — scene AABB max (with offset)


class MultiSceneEnvManager(EnvManager):
    """EnvManager that loads multiple coexisting static scenes.

    Args:
        scene_infos: List of SceneInfo — one per coexisting scene.
        scene_spacing_margin: Extra padding (metres) between scenes along the
            layout axis to avoid any overlap.
        All other args are forwarded to EnvManager.
    """

    def __init__(
        self,
        sim_name,
        env_name,
        robot_name,
        controller_name,
        device,
        scene_infos,
        scene_spacing_margin=50.0,
        args=None,
        num_envs=None,
        use_warp=None,
        headless=None,
    ):
        self._scene_infos = scene_infos
        self._scene_spacing_margin = float(scene_spacing_margin)
        self._scene_slots = []       # populated in _load_static_scenes
        self._env_to_scene = []      # populated in _load_static_scenes

        # The parent __init__ will call populate_env → _load_static_scenes
        # and prepare_sim → _setup_navmesh_sampler, both of which we override.
        super().__init__(
            sim_name=sim_name,
            env_name=env_name,
            robot_name=robot_name,
            controller_name=controller_name,
            device=device,
            args=args,
            num_envs=num_envs,
            use_warp=use_warp,
            headless=headless,
        )

    # ------------------------------------------------------------------
    # Properties exposed for the task / navmesh goal sampler
    # ------------------------------------------------------------------

    @property
    def env_to_scene(self):
        return self._env_to_scene

    @property
    def scene_slots(self):
        return self._scene_slots

    # ------------------------------------------------------------------
    # Override: static scene loading
    # ------------------------------------------------------------------

    def _load_static_scenes(self):
        """Load N scenes at world-space offsets and assign envs round-robin."""
        n_scenes = len(self._scene_infos)
        n_envs = self.cfg.env.num_envs

        # --- Build scene configs and load meshes ---
        template_cfg = self.cfg.static_scene if hasattr(self.cfg, "static_scene") else None
        slots = []
        for info in self._scene_infos:
            slot = _SceneSlot()
            slot.info = info
            cfg = self._make_scene_cfg(template_cfg, info)
            slot.static_scene = StaticSceneGLB(cfg)
            slots.append(slot)

        # --- Compute scene offsets along X axis ---
        spacing = 0.0
        for slot in slots:
            extent = float(slot.static_scene.bounds[1, 0] - slot.static_scene.bounds[0, 0])
            spacing = max(spacing, extent)
        spacing += self._scene_spacing_margin

        for i, slot in enumerate(slots):
            slot.offset = np.array([i * spacing, 0.0, 0.0], dtype=np.float32)
            bmin = slot.static_scene.bounds[0] + slot.offset
            bmax = slot.static_scene.bounds[1] + slot.offset
            slot.bounds_min = bmin
            slot.bounds_max = bmax

        self._scene_slots = slots

        # --- Assign envs to scenes (round-robin) ---
        self._env_to_scene = [i % n_scenes for i in range(n_envs)]

        logger.info(
            "MultiScene: %d scenes, %d envs, spacing=%.1f m",
            n_scenes, n_envs, spacing,
        )
        for i, slot in enumerate(slots):
            assigned = sum(1 for e in self._env_to_scene if e == i)
            logger.info(
                "  scene %d: %s | offset=(%.1f, %.1f, %.1f) | %d envs",
                i, slot.info.scene_id,
                slot.offset[0], slot.offset[1], slot.offset[2],
                assigned,
            )

        # --- Add collision meshes (one per scene, NOT per env) ---
        for slot in slots:
            self.IGE_env.add_static_triangle_mesh(
                vertices=slot.static_scene.collision_vertices,
                faces=slot.static_scene.collision_faces,
                translation=tuple(slot.offset),
                static_friction=getattr(self.cfg.static_scene, "collision_static_friction", 1.0),
                dynamic_friction=getattr(self.cfg.static_scene, "collision_dynamic_friction", 1.0),
                restitution=getattr(self.cfg.static_scene, "collision_restitution", 0.0),
                segmentation_id=getattr(self.cfg.static_scene, "segmentation_id", 0),
            )

        # --- Warp rendering (per-env scene assignment) ---
        if self.cfg.env.use_warp:
            static_scenes = [s.static_scene for s in slots]
            offsets = np.stack([s.offset for s in slots])
            self.warp_env.set_multi_scene(static_scenes, self._env_to_scene, offsets)

        # Keep a reference for compatibility (use first scene).
        self.static_scene = slots[0].static_scene if slots else None

    # ------------------------------------------------------------------
    # Override: navmesh sampler setup
    # ------------------------------------------------------------------

    def _setup_navmesh_sampler(self):
        """Create a MultiSceneNavMeshSampler instead of the single-scene one."""
        nav_cfg = getattr(self.cfg, "navmesh_sampling", None)
        if nav_cfg is None or not getattr(nav_cfg, "enable", False):
            self.navmesh_sampler = None
            return

        self.navmesh_sampler = _MultiSceneNavMeshSampler(
            scene_slots=self._scene_slots,
            env_to_scene=self._env_to_scene,
            env_cfg=self.cfg,
            device=self.device,
        )

    # ------------------------------------------------------------------
    # Override: per-env bounds after prepare_sim
    # ------------------------------------------------------------------

    def prepare_sim(self):
        super().prepare_sim()
        self._apply_per_env_bounds()

    def _apply_per_env_bounds(self):
        """Set each env's bounds to its assigned scene's AABB (world-frame)."""
        for env_id in range(self.num_envs):
            sid = self._env_to_scene[env_id]
            slot = self._scene_slots[sid]
            bmin = torch.tensor(slot.bounds_min, device=self.device, dtype=torch.float32)
            bmax = torch.tensor(slot.bounds_max, device=self.device, dtype=torch.float32)
            # Deterministic bounds (min == max → no random sampling on reset).
            self.IGE_env.env_lower_bound_min[env_id] = bmin
            self.IGE_env.env_lower_bound_max[env_id] = bmin
            self.IGE_env.env_upper_bound_min[env_id] = bmax
            self.IGE_env.env_upper_bound_max[env_id] = bmax
            self.IGE_env.env_lower_bound[env_id] = bmin
            self.IGE_env.env_upper_bound[env_id] = bmax

        # Update the global tensor dict references.
        self.global_tensor_dict["env_bounds_min"] = self.IGE_env.env_lower_bound
        self.global_tensor_dict["env_bounds_max"] = self.IGE_env.env_upper_bound

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_scene_cfg(template_cfg, scene_info: SceneInfo):
        """Build a scene config object for StaticSceneGLB from a template + SceneInfo."""
        class _Cfg:
            pass

        cfg = _Cfg()
        # Copy all attributes from template.
        if template_cfg is not None:
            for key in dir(template_cfg):
                if not key.startswith("_"):
                    setattr(cfg, key, getattr(template_cfg, key))
        # Override the file path.
        cfg.file = scene_info.glb_path
        cfg.enable = True
        return cfg


# ======================================================================
# Multi-scene navmesh sampler
# ======================================================================

class _MultiSceneNavMeshSampler:
    """Routes navmesh spawn/goal sampling to the correct per-scene navmesh.

    Exposes the same interface as NavMeshSpawnSampler (enabled, apply_spawn,
    sample_world_points) so the rest of the code works unchanged.
    """

    def __init__(self, scene_slots, env_to_scene, env_cfg, device):
        self.device = device
        self.env_to_scene = env_to_scene
        self.scene_slots = scene_slots
        self.nav_cfg = getattr(env_cfg, "navmesh_sampling", None)
        self.enabled = False

        # Load navmeshes and compute per-scene transforms.
        self._scene_navmeshes = []        # StandaloneNavMesh or None
        self._scene_offsets_torch = []    # (3,) tensors on device
        self._scene_transforms = []       # (scale, translation) tuples

        static_scene_cfg = getattr(env_cfg, "static_scene", None)
        inherit = getattr(self.nav_cfg, "inherit_scene_transform", True) if self.nav_cfg else True
        nav_scale = float(getattr(self.nav_cfg, "navmesh_scale", 1.0)) if self.nav_cfg else 1.0
        nav_trans = getattr(self.nav_cfg, "navmesh_translation", [0.0, 0.0, 0.0]) if self.nav_cfg else [0.0, 0.0, 0.0]
        nav_translation = torch.tensor(nav_trans, dtype=torch.float32, device=device)

        scene_scale_cfg = float(getattr(static_scene_cfg, "scale", 1.0)) if static_scene_cfg else 1.0
        scene_trans_cfg = getattr(static_scene_cfg, "translation", [0.0, 0.0, 0.0]) if static_scene_cfg else [0.0, 0.0, 0.0]
        scene_translation = torch.tensor(scene_trans_cfg, dtype=torch.float32, device=device)

        for slot in scene_slots:
            self._scene_offsets_torch.append(
                torch.tensor(slot.offset, dtype=torch.float32, device=device)
            )

            if slot.info.navmesh_path is None:
                self._scene_navmeshes.append(None)
                self._scene_transforms.append(None)
                continue

            navmesh = StandaloneNavMesh(slot.info.navmesh_path)
            if navmesh.pt_vertices is None or navmesh.pt_polygons is None:
                logger.warning("Scene %s: navmesh has no valid polygons.", slot.info.scene_id)
                self._scene_navmeshes.append(None)
                self._scene_transforms.append(None)
                continue

            navmesh.pt_vertices = navmesh.pt_vertices.to(device)
            navmesh.pt_polygons = navmesh.pt_polygons.to(device)
            navmesh.pt_poly_areas = navmesh.pt_poly_areas.to(device)
            self._scene_navmeshes.append(navmesh)

            # Pre-compute the transform: scene_scale * x + scene_translation,
            # then navmesh_scale * x + navmesh_translation.
            if inherit:
                self._scene_transforms.append((scene_scale_cfg, scene_translation, nav_scale, nav_translation))
            else:
                self._scene_transforms.append((1.0, torch.zeros(3, device=device), nav_scale, nav_translation))

            self.enabled = True

        if not self.enabled:
            logger.warning("MultiSceneNavMeshSampler: no valid navmeshes found.")

    # ------------------------------------------------------------------
    # Public interface (matches NavMeshSpawnSampler)
    # ------------------------------------------------------------------

    def apply_spawn(self, robot_state_tensor, env_ids, env_bounds_min=None, env_bounds_max=None):
        if not self.enabled or len(env_ids) == 0:
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

    def sample_world_points(self, env_ids, height_offset_range, bounds_min=None, bounds_max=None, edge_padding=None):
        if not self.enabled or len(env_ids) == 0:
            return None

        env_ids = env_ids.to(dtype=torch.long, device=self.device)
        if edge_padding is None:
            edge_padding = float(getattr(self.nav_cfg, "edge_padding", 0.0))
        enforce_bounds = bool(getattr(self.nav_cfg, "enforce_env_bounds", True))
        max_resample = int(getattr(self.nav_cfg, "max_bound_resample_rounds", 5))

        results = torch.zeros((len(env_ids), 3), device=self.device, dtype=torch.float32)

        # Group envs by scene for efficient batched sampling.
        scene_groups = {}
        for local_idx, env_id in enumerate(env_ids):
            sid = self.env_to_scene[int(env_id.item())]
            scene_groups.setdefault(sid, []).append((local_idx, env_id))

        for sid, pairs in scene_groups.items():
            navmesh = self._scene_navmeshes[sid]
            if navmesh is None:
                continue
            transform = self._scene_transforms[sid]
            offset = self._scene_offsets_torch[sid]
            slot = self.scene_slots[sid]

            for local_idx, env_id in pairs:
                point = self._sample_one(
                    navmesh, transform, offset, slot,
                    height_offset_range, edge_padding,
                    env_id, bounds_min, bounds_max,
                    enforce_bounds, max_resample,
                )
                results[local_idx] = point

        return results

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _apply_transform(self, points, transform):
        """Apply scene+navmesh transform to raw navmesh points."""
        scene_scale, scene_trans, nav_scale, nav_trans = transform
        points = points * scene_scale + scene_trans
        points = points * nav_scale + nav_trans
        return points

    def _sample_one(self, navmesh, transform, offset, slot,
                     height_offset_range, edge_padding,
                     env_id, bounds_min, bounds_max,
                     enforce_bounds, max_resample):
        """Sample a single valid world-frame point for one env."""
        candidate_count = 8

        bmin_world = None
        bmax_world = None
        if enforce_bounds and bounds_min is not None and bounds_max is not None:
            bmin_world = bounds_min.to(self.device)[env_id]
            bmax_world = bounds_max.to(self.device)[env_id]

        for _ in range(max_resample):
            raw = navmesh.sample_points_with_padding(
                count=candidate_count,
                height_offset=tuple(height_offset_range),
                edge_padding=edge_padding,
            ).to(self.device)

            local = self._apply_transform(raw, transform)
            world = local + offset

            if bmin_world is not None and bmax_world is not None:
                in_bounds = torch.logical_and(
                    world >= bmin_world.unsqueeze(0),
                    world <= bmax_world.unsqueeze(0),
                ).all(dim=1)
                kept = world[in_bounds]
            else:
                kept = world

            if kept.shape[0] > 0:
                pick = torch.randint(0, kept.shape[0], (1,), device=self.device)
                return kept[pick[0]]

        # Fallback: return an unfiltered sample.
        raw = navmesh.sample_points_with_padding(
            count=1,
            height_offset=tuple(height_offset_range),
            edge_padding=edge_padding,
        ).to(self.device)
        return self._apply_transform(raw, transform)[0] + offset
