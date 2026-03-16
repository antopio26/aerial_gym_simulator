from aerial_gym.config.sensor_config.camera_config.base_depth_camera_config import BaseDepthCameraConfig


class ShadedRGBDCameraConfig(BaseDepthCameraConfig):
    # Distinct sensor type so existing camera paths remain untouched.
    sensor_type = "shaded_rgbd_camera"

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
