from aerial_gym.config.sensor_config.camera_config.base_depth_camera_config import BaseDepthCameraConfig


class ShadedRGBDCameraConfig(BaseDepthCameraConfig):
    # Distinct sensor type so existing camera paths remain untouched.
    sensor_type = "shaded_rgbd_camera"

    # Match base depth camera geometry as closely as possible.
    height = 240
    width = 320

    max_range = 10.0

    # Shaded RGBD currently returns depth in depth_range_pixels and color in rgb_pixels.
    segmentation_camera = False
    return_pointcloud = False
    calculate_depth = True

    # Simple directional lighting controls.
    enable_lighting = False
    ambient_strength = 0.2
    light_direction = [0.3, 0.4, 0.85]

    enable_textures = True
    debug_uv_checker = False
