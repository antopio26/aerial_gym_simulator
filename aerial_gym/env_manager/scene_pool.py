"""Scene discovery and pool management for multi-scene training.

Scans a base folder for Matterport-style scene directories, each containing
a GLB mesh and optionally a navmesh file:

    base_folder/
    ├── 00807-rsggHU7g7dh/
    │   ├── rsggHU7g7dh.glb
    │   └── rsggHU7g7dh.basis.navmesh
    ├── 00808-y9hTuugGdiq/
    │   ├── y9hTuugGdiq.glb
    │   └── y9hTuugGdiq.basis.navmesh
    └── ...
"""

import os
import random
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from aerial_gym import AERIAL_GYM_DIRECTORY
from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger("scene_pool")


@dataclass
class SceneInfo:
    """Metadata for a single discoverable scene."""
    scene_id: str
    glb_path: str
    navmesh_path: Optional[str] = None


class ScenePool:
    """Discovers scenes under *base_folder* and samples subsets for rotation.

    Args:
        base_folder: Directory containing one sub-folder per scene.
        shuffle:     If True, scenes are shuffled on discovery (default True).
    """

    def __init__(self, base_folder: str, shuffle: bool = True):
        self.base_folder = self._resolve(base_folder)
        self.scenes: List[SceneInfo] = self._discover()
        if shuffle:
            random.shuffle(self.scenes)
        if not self.scenes:
            raise FileNotFoundError(
                f"No scenes found in {self.base_folder}. "
                "Each sub-folder must contain at least one .glb file."
            )
        logger.info("ScenePool: discovered %d scenes in %s", len(self.scenes), self.base_folder)
        self._rotation_idx = 0

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self, n: int, exclude: Optional[Sequence[str]] = None) -> List[SceneInfo]:
        """Return *n* scenes, avoiding *exclude* scene_ids when possible.

        Uses round-robin rotation so every scene gets visited before any
        repeats.  Falls back to random if the pool is smaller than *n*.
        """
        pool = self.scenes
        if exclude:
            exclude_set = set(exclude)
            pool = [s for s in self.scenes if s.scene_id not in exclude_set]
            if len(pool) < n:
                pool = self.scenes  # not enough unique scenes; allow repeats

        if n >= len(pool):
            return list(pool)

        selected = []
        while len(selected) < n:
            idx = self._rotation_idx % len(pool)
            selected.append(pool[idx])
            self._rotation_idx += 1
        return selected

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve(path: str) -> str:
        if os.path.isabs(path):
            return path
        candidate = os.path.join(AERIAL_GYM_DIRECTORY, path)
        if os.path.isdir(candidate):
            return candidate
        return os.path.abspath(path)

    def _discover(self) -> List[SceneInfo]:
        scenes = []
        if not os.path.isdir(self.base_folder):
            return scenes

        for entry in sorted(os.listdir(self.base_folder)):
            scene_dir = os.path.join(self.base_folder, entry)
            if not os.path.isdir(scene_dir):
                continue

            glb_files = sorted(
                f for f in os.listdir(scene_dir) if f.lower().endswith(".glb")
            )
            if not glb_files:
                continue

            glb_path = os.path.join(scene_dir, glb_files[0])
            scene_id = os.path.splitext(glb_files[0])[0]

            # Resolve navmesh: try scene_id.basis.navmesh, scene_id.navmesh,
            # then first .navmesh in directory.
            navmesh_path = None
            candidates = [
                os.path.join(scene_dir, f"{scene_id}.basis.navmesh"),
                os.path.join(scene_dir, f"{scene_id}.navmesh"),
            ]
            for cand in candidates:
                if os.path.exists(cand):
                    navmesh_path = cand
                    break
            if navmesh_path is None:
                nav_files = sorted(
                    f for f in os.listdir(scene_dir) if f.endswith(".navmesh")
                )
                if nav_files:
                    navmesh_path = os.path.join(scene_dir, nav_files[0])

            scenes.append(SceneInfo(
                scene_id=scene_id,
                glb_path=glb_path,
                navmesh_path=navmesh_path,
            ))

        return scenes
