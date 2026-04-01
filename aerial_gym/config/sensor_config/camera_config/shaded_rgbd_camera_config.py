from aerial_gym.config.sensor_config.camera_config.base_depth_camera_config import BaseDepthCameraConfig


class RGBDCameraConfig(BaseDepthCameraConfig):
    # Distinct sensor type so existing camera paths remain untouched.
    sensor_type = "rgbd_camera"

    # Match base depth camera geometry as closely as possible.
    height = 240
    width = 320

    # Depth and RGB channels can have independent max ranges.
    # depth_max_range: ray hits beyond this distance report depth = depth_max_range.
    # rgb_max_range:   ray hits beyond this distance produce a black pixel (background).
    # The actual ray cast distance is max(depth_max_range, rgb_max_range).
    depth_max_range = 10.0
    rgb_max_range = 20.0

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
