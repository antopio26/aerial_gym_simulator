class MatterportGLBEnvCfg:
    # Global scene scale factor applied uniformly to the mesh and navmesh.
    # Scales the static scene geometry and the navmesh (via inherit_scene_transform),
    # but does NOT affect the robot size or dynamics.
    # This multiplies static_scene.scale, so the effective mesh scale is
    #   static_scene.scale * scene_scale.
    scene_scale = 1.5

    class env:
        num_envs = 1
        num_env_actions = 0
        env_spacing = 0.0

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

        # By default, infer env bounds directly from the static scene mesh AABB.
        # To override manually, define lower_bound_min/lower_bound_max and
        # upper_bound_min/upper_bound_max explicitly in this class.
        auto_env_bounds_from_static_scene = True
        auto_env_bounds_padding = [0.0, 0.0, 0.0]

    class env_config:
        include_asset_type = {}
        asset_type_to_dict_map = {}

    class static_scene:
        enable = True
        file = "resources/envs/00807-rsggHU7g7dh/rsggHU7g7dh.glb"
        collision_file = None
        scale = 1.0
        translation = [5.0, -5.0, 0.0]
        enable_texture_rendering = True
        # Resolution per unique texture tile in the deduplicated atlas.
        # 41 unique tiles at 2048px → 7x6 atlas ~14336x12288 (~2 GB float32 GPU).
        # Lower to 1024 to reduce to ~530 MB, or 512 for ~132 MB.
        texture_atlas_tile_size = 2048
        texture_atlas_tile_padding = 2
        collision_static_friction = 1.0
        collision_dynamic_friction = 1.0
        collision_restitution = 0.0
        segmentation_id = 0

    class navmesh_sampling:
        enable = False
        # Optional explicit navmesh path. If None, it is auto-resolved from static_scene.file.
        navmesh_file = None

        # Optional transform layer for navmesh coordinates.
        inherit_scene_transform = True
        navmesh_scale = 1.0
        navmesh_translation = [0.0, 0.0, 0.0]

        # Safe spawn configuration.
        spawn_height_offset_range = [0.15, 0.40]
        edge_padding = 0.30
        enforce_env_bounds = True
        max_bound_resample_rounds = 40
        zero_velocity_on_spawn = True
