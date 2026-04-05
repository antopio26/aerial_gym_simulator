"""Environment configs for multi-scene Matterport training.

These extend the single-scene MatterportGLBEnvCfg with a ``scene_pool``
section that tells MultiSceneEnvManager where to find scenes and how many to
load simultaneously.

The ``static_scene`` section acts as a **template**: its material/friction
properties are applied to every loaded scene, but ``file`` is ignored
(overridden per scene by the scene pool).

Variants:
    MultiSceneMatterportEnvCfg           — full textures (needed for ViT pipeline)
    MultiSceneMatterportEnvNoTexturesCfg — no textures (lighter, for VAE depth pipeline)
"""

from aerial_gym.config.env_config.matterport_glb_env import (
    MatterportGLBEnvCfg,
    MatterportGLBEnvNoTexturesCfg,
)


class MultiSceneMatterportEnvCfg(MatterportGLBEnvCfg):

    class env(MatterportGLBEnvCfg.env):
        # Disable auto bounds — MultiSceneEnvManager computes per-env bounds
        # from each scene's AABB after prepare_sim().  These large placeholder
        # values satisfy IGE_env_manager's requirement for explicit bounds.
        auto_env_bounds_from_static_scene = False
        lower_bound_min = [-10000.0, -10000.0, -10000.0]
        lower_bound_max = [-10000.0, -10000.0, -10000.0]
        upper_bound_min = [10000.0, 10000.0, 10000.0]
        upper_bound_max = [10000.0, 10000.0, 10000.0]

    class scene_pool:
        # Root folder containing one sub-directory per scene.
        base_folder = "resources/envs/train"
        # How many distinct scenes are loaded into the sim at once.
        num_coexisting_scenes = 4
        # Extra spacing (metres) between scene collision meshes along the
        # layout axis to guarantee no overlap.
        scene_spacing_margin = 50.0

    class static_scene(MatterportGLBEnvCfg.static_scene):
        # ``file`` is unused — overridden per scene by the pool.
        # All other fields (scale, friction, texture settings) serve as the
        # template applied to every scene.
        pass

    class navmesh_sampling(MatterportGLBEnvCfg.navmesh_sampling):
        pass


class MultiSceneMatterportEnvNoTexturesCfg(MultiSceneMatterportEnvCfg):
    class static_scene(MatterportGLBEnvNoTexturesCfg.static_scene):
        pass
