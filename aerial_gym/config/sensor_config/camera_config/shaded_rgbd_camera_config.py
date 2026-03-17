from aerial_gym.config.sensor_config.camera_config.base_depth_camera_config import BaseDepthCameraConfig


class ShadedRGBDCameraConfig(BaseDepthCameraConfig):
    # Distinct sensor type so existing camera paths remain untouched.
    sensor_type = "shaded_rgbd_camera"

    height = 270
    width = 480

    # Shaded RGBD currently returns depth in depth_range_pixels and color in rgb_pixels.
    segmentation_camera = False
    return_pointcloud = False
    calculate_depth = True

    # Simple directional lighting controls for the first implementation.
    enable_lighting = True
    ambient_strength = 0.2
    light_direction = [0.3, 0.4, 0.85]

    enable_textures = True
    texture_filter_mode = "bilinear"
    debug_uv_checker = False

    # UV debug controls:
    # uv_bary_mode permutations (weights assigned to uv0, uv1, uv2):
    # 0: (w, u, v)
    # 1: (w, v, u)
    # 2: (u, w, v)
    # 3: (u, v, w)
    # 4: (v, w, u)
    # 5: (v, u, w)
    # b3 was validated on Matterport GLB scenes as the correct mapping.
    uv_bary_mode = 3
    # uv_transform_mode (D4 symmetry family on UV square):
    # 0: (u, v)
    # 1: (1-u, v)
    # 2: (u, 1-v)
    # 3: (1-u, 1-v)
    # 4: (v, u)
    # 5: (1-v, u)
    # 6: (v, 1-u)
    # 7: (1-v, 1-u)
    uv_transform_mode = 0
