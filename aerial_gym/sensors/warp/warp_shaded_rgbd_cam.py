import warp as wp
import math
from aerial_gym.sensors.warp.warp_kernels.shaded_rgbd_camera_kernels import ShadedRGBCameraWarpKernels


class WarpShadedRGBCam:
    def __init__(
        self,
        num_envs,
        config,
        mesh_ids_array,
        vertex_colors_array=None,
        vertex_color_offsets=None,
        device="cuda:0",
    ):
        self.cfg = config
        self.num_envs = num_envs
        self.num_sensors = self.cfg.num_sensors
        self.mesh_ids_array = mesh_ids_array
        self.vertex_colors_array = None
        self.vertex_color_offsets = None
        self.vertex_uvs = None
        self.texture_image = None
        self.texture_width = 0
        self.texture_height = 0
        self.base_color_factor = wp.vec3(1.0, 1.0, 1.0)
        self.width = self.cfg.width
        self.height = self.cfg.height
        self.horizontal_fov = math.radians(self.cfg.horizontal_fov_deg)
        self.far_plane = self.cfg.max_range
        self.ambient_strength = float(getattr(self.cfg, "ambient_strength", 0.2))
        self.light_dir_world = wp.vec3(*getattr(self.cfg, "light_direction", [0.0, 0.0, 1.0]))
        self.device = device
        self.camera_position_array = None
        self.camera_orientation_array = None
        self.graph = None
        self.initialize_camera_matrices()

        if vertex_colors_array is not None and vertex_color_offsets is not None:
            self.set_vertex_color_buffers(vertex_colors_array, vertex_color_offsets)

    def initialize_camera_matrices(self):
        W = self.width
        H = self.height
        (u_0, v_0) = (W / 2, H / 2)
        f = W / 2 * 1 / math.tan(self.horizontal_fov / 2)
        vertical_fov = 2 * math.atan(H / (2 * f))
        alpha_u = u_0 / math.tan(self.horizontal_fov / 2)
        alpha_v = v_0 / math.tan(vertical_fov / 2)
        self.K = wp.mat44(
            alpha_u, 0.0, u_0, 0.0,
            0.0, alpha_v, v_0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        )
        self.K_inv = wp.inverse(self.K)
        self.c_x = int(u_0)
        self.c_y = int(v_0)

    def set_image_tensors(self, rgb_pixels, depth_pixels):
        self.rgb_pixels = wp.from_torch(rgb_pixels, dtype=wp.vec3)
        self.depth_pixels = wp.from_torch(depth_pixels, dtype=wp.float32)

    def set_vertex_color_buffers(self, vertex_colors_array, vertex_color_offsets):
        self.vertex_colors_array = wp.from_torch(vertex_colors_array, dtype=wp.vec3)
        self.vertex_color_offsets = wp.from_torch(vertex_color_offsets, dtype=wp.int32)

    def set_texture_buffers(self, vertex_uvs, texture_image, base_color_factor):
        self.vertex_uvs = wp.from_torch(vertex_uvs, dtype=wp.vec2)
        self.texture_image = wp.from_torch(texture_image, dtype=wp.vec3)
        self.texture_height = int(texture_image.shape[0])
        self.texture_width = int(texture_image.shape[1])
        self.base_color_factor = wp.vec3(*base_color_factor.tolist())

    def set_pose_tensor(self, positions, orientations):
        self.camera_position_array = wp.from_torch(positions, dtype=wp.vec3)
        self.camera_orientation_array = wp.from_torch(orientations, dtype=wp.quat)

    def create_render_graph(self, debug=False):
        if not debug:
            wp.capture_begin(device=self.device)

        if getattr(self.cfg, "enable_textures", False) and self.vertex_uvs is not None and self.texture_image is not None:
            wp.launch(
                kernel=ShadedRGBCameraWarpKernels.draw_textured_rgbd_kernel,
                dim=(self.num_envs, self.num_sensors, self.width, self.height),
                inputs=[
                    self.mesh_ids_array,
                    self.camera_position_array,
                    self.camera_orientation_array,
                    self.K_inv,
                    self.far_plane,
                    self.rgb_pixels,
                    self.depth_pixels,
                    self.vertex_uvs,
                    self.texture_image,
                    self.texture_width,
                    self.texture_height,
                    self.base_color_factor,
                    self.ambient_strength,
                    self.light_dir_world,
                    self.c_x,
                    self.c_y,
                ],
                device=self.device,
            )
        else:
            wp.launch(
                kernel=ShadedRGBCameraWarpKernels.draw_shaded_rgbd_kernel,
                dim=(self.num_envs, self.num_sensors, self.width, self.height),
                inputs=[
                    self.mesh_ids_array,
                    self.camera_position_array,
                    self.camera_orientation_array,
                    self.K_inv,
                    self.far_plane,
                    self.rgb_pixels,
                    self.depth_pixels,
                    self.vertex_colors_array,
                    self.vertex_color_offsets,
                    self.ambient_strength,
                    self.light_dir_world,
                    self.c_x,
                    self.c_y,
                ],
                device=self.device,
            )

        if not debug:
            self.graph = wp.capture_end(device=self.device)

    def capture(self, debug=False):
        if debug:
            self.create_render_graph(debug=True)
            return

        if self.graph is None:
            self.create_render_graph(debug=debug)

        wp.capture_launch(self.graph)
