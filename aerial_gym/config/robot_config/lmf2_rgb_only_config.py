from aerial_gym.config.robot_config.lmf2_config import LMF2Cfg
from aerial_gym.config.sensor_config.camera_config.rgb_only_camera_config import (
    RGBOnlyCameraConfig,
)


class LMF2RGBOnlyCfg(LMF2Cfg):
    """LMF2 with RGB-only camera.

    Identical to lmf2_with_rgbd_camera but with calculate_depth=False,
    so the warp renderer skips populating depth_range_pixels.  Use this for
    the ViT DCE pipeline where only rgb_pixels are consumed, to save tensor
    memory and avoid unnecessary depth computation.
    """

    class sensor_config(LMF2Cfg.sensor_config):
        enable_camera = True
        camera_config = RGBOnlyCameraConfig
