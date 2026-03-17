class MatterportGLBEnvCfg:
    class env:
        num_envs = 1
        num_env_actions = 0
        env_spacing = 20.0

        num_physics_steps_per_env_step_mean = 1
        num_physics_steps_per_env_step_std = 0

        render_viewer_every_n_steps = 10
        collision_force_threshold = 0.010
        reset_on_collision = True
        create_ground_plane = False
        sample_timestep_for_latency = True
        perturb_observations = True
        keep_same_env_for_num_episodes = 1
        write_to_sim_at_every_timestep = False

        use_warp = True

        lower_bound_min = [-11.0, -0.5, 0.4]
        lower_bound_max = [-8.0, 1.5, 1.2]
        upper_bound_min = [1.0, 8.5, 2.0]
        upper_bound_max = [3.0, 9.5, 3.0]

    class env_config:
        include_asset_type = {}
        asset_type_to_dict_map = {}

    class static_scene:
        enable = True
        file = "resources/envs/HaxA7YrQdEC.glb"
        collision_file = None
        scale = 1.0
        translation = [0.0, 0.0, 0.0]
        enable_texture_rendering = True
        # Resolution per unique texture tile in the deduplicated atlas.
        # 41 unique tiles at 2048px → 7×6 atlas ~14336×12288 (~2 GB float32 GPU).
        # Lower to 1024 to reduce to ~530 MB, or 512 for ~132 MB.
        texture_atlas_tile_size = 2048
        texture_atlas_tile_padding = 2
        collision_static_friction = 1.0
        collision_dynamic_friction = 1.0
        collision_restitution = 0.0
        segmentation_id = 0
