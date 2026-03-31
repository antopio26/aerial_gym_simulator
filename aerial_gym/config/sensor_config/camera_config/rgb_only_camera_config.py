from aerial_gym.config.sensor_config.camera_config.shaded_rgbd_camera_config import (
    ShadedRGBDCameraConfig,
)


class RGBOnlyCameraConfig(ShadedRGBDCameraConfig):
    """Shaded RGB camera without depth output.

    Inherits all geometry and lighting settings from ShadedRGBDCameraConfig but
    sets calculate_depth=False so the warp renderer does not populate
    depth_range_pixels, saving the associated tensor memory and compute.

    Use this for the ViT DCE pipeline where only rgb_pixels are consumed.
    """

    calculate_depth = False
