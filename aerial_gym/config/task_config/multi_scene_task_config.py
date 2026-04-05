"""Task configs for multi-scene DCE RL navigation.

Extends the single-scene MatterportVAETaskConfig with a ``scene_rotation``
section that controls whether and when to swap the active set of scenes.

Variants:
    MultiSceneVAETaskConfig   — VAE depth pipeline, no textures env.
    MultiSceneViTTaskConfig   — ViT RGB pipeline, textured env.
"""

from aerial_gym.config.task_config.matterport_dce_task_config import (
    MatterportVAETaskConfig,
    MatterportViTTaskConfig,
)


class MultiSceneVAETaskConfig(MatterportVAETaskConfig):
    env_name = "multi_scene_matterport_env_no_textures"

    class scene_rotation:
        # Set to False (or omit) to load N scenes once and never swap.
        enable = False
        # Swap trigger — exactly one of these should be set (the other None).
        switch_after_steps = None       # task-level steps (each = one policy call)
        switch_after_rollouts = None    # completed episodes across all envs
        # sync: wait for all episodes to finish, destroy sim, rebuild with new scenes.
        mode = "sync"

    class navmesh_sampling(MatterportVAETaskConfig.navmesh_sampling):
        pass

    class navmesh_curriculum(MatterportVAETaskConfig.navmesh_curriculum):
        pass

    class vae_config(MatterportVAETaskConfig.vae_config):
        pass

    class curriculum(MatterportVAETaskConfig.curriculum):
        pass


class MultiSceneViTTaskConfig(MatterportViTTaskConfig):
    env_name = "multi_scene_matterport_env"

    class scene_rotation(MultiSceneVAETaskConfig.scene_rotation):
        pass

    class vae_config(MatterportViTTaskConfig.vae_config):
        pass

    class vit_config(MatterportViTTaskConfig.vit_config):
        pass

    class curriculum(MatterportViTTaskConfig.curriculum):
        pass
