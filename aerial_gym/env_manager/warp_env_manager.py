from aerial_gym.env_manager.base_env_manager import BaseManager

import warp as wp
import numpy as np
import torch

import trimesh as tm

from aerial_gym.utils.math import tf_apply

# intialize warp
wp.init()

from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger("warp_env_manager")


class WarpEnv(BaseManager):
    def __init__(self, global_sim_dict, device):
        logger.debug("Initializing WarpEnv")
        super().__init__(global_sim_dict["env_cfg"], device)
        self.num_envs = global_sim_dict["num_envs"]
        self.env_meshes = []
        self.warp_mesh_id_list = []
        self.warp_mesh_per_env = []
        self.global_vertex_to_asset_index_tensor = None
        self.vertex_maps_per_env_original = None
        self.global_env_mesh_list = []
        self.global_vertex_counter = 0
        self.global_vertex_segmentation_list = []
        self.global_vertex_to_asset_index_map = []

        self.CONST_WARP_MESH_ID_LIST = None
        self.CONST_WARP_MESH_PER_ENV = None
        self.CONST_GLOBAL_VERTEX_TO_ASSET_INDEX_TENSOR = None
        self.VERTEX_MAPS_PER_ENV_ORIGINAL = None
        self.static_scene = None
        self.env_origins = None
        logger.debug("[DONE] Initializing WarpEnv")

    def set_static_scene(self, static_scene, env_origins):
        self.static_scene = static_scene
        self.env_origins = np.asarray(env_origins, dtype=np.float32)

    def set_multi_scene(self, scenes, env_to_scene, scene_offsets):
        """Configure per-env static scene assignment for multi-scene rendering.

        Args:
            scenes:        list of StaticSceneGLB instances (one per coexisting scene).
            env_to_scene:  list[int] of length num_envs mapping env_id → scene index.
            scene_offsets:  (N_scenes, 3) numpy array of world-space translations.
        """
        self.multi_scenes = scenes
        self.env_to_scene = env_to_scene
        self.scene_offsets = np.asarray(scene_offsets, dtype=np.float32)
        # Flag that prepare_for_simulation should use multi-scene path.
        self.static_scene = None  # disable single-scene path

    def reset_idx(self, env_ids):
        if self.global_vertex_counter == 0:
            return
        # logger.debug("Updating vertex maps per env")
        self.vertex_maps_per_env_updated[:] = tf_apply(
            self.unfolded_env_vec_root_tensor[self.CONST_GLOBAL_VERTEX_TO_ASSET_INDEX_TENSOR, 3:7],
            self.unfolded_env_vec_root_tensor[self.CONST_GLOBAL_VERTEX_TO_ASSET_INDEX_TENSOR, 0:3],
            self.VERTEX_MAPS_PER_ENV_ORIGINAL[:],
        )
        # logger.debug("[DONE] Updating vertex maps per env")

        # logger.debug("Refitting warp meshes")
        for i in env_ids:
            self.warp_mesh_per_env[i].refit()
        # logger.debug("[DONE] Refitting warp meshes")

    def pre_physics_step(self, action):
        pass

    def post_physics_step(self):
        pass

    def step(self, action):
        pass

    def reset(self):
        return self.reset_idx(torch.arange(self.num_envs, device=self.device))

    def create_env(self, env_id):
        if len(self.env_meshes) <= env_id:
            self.env_meshes.append([])
        else:
            raise ValueError("Environment already exists")

    def add_asset_to_env(self, asset_info_dict, env_id, global_asset_counter, segmentation_counter):
        warp_asset = asset_info_dict["warp_asset"]
        # use the variable segmentation mask to set the segmentation id for each vertex
        updated_vertex_segmentation = (
            warp_asset.asset_vertex_segmentation_value
            + segmentation_counter * warp_asset.variable_segmentation_mask
        )
        logger.debug(
            f"Asset {asset_info_dict['filename']} has {len(warp_asset.asset_unified_mesh.vertices)} vertices. Segmentation mask: {warp_asset.variable_segmentation_mask} and updated segmentation: {updated_vertex_segmentation}"
        )
        self.env_meshes[env_id].append(warp_asset.asset_unified_mesh)

        self.global_vertex_to_asset_index_map += [global_asset_counter] * len(
            warp_asset.asset_unified_mesh.vertices
        )
        self.global_vertex_counter += len(warp_asset.asset_unified_mesh.vertices)
        self.global_vertex_segmentation_list += updated_vertex_segmentation.tolist()
        return None, len(
            np.unique(
                warp_asset.asset_vertex_segmentation_value * warp_asset.variable_segmentation_mask
            )
        )

    def _prepare_multi_scene(self):
        """Build per-env warp meshes using multi-scene assignment.

        Each env gets the render mesh of its assigned scene, translated by the
        scene's world-space offset.  Texture data is taken from the first scene
        that has textures (all scenes share the same atlas pipeline so this is
        fine for single-material Matterport scenes).
        """
        scenes = self.multi_scenes
        env_to_scene = self.env_to_scene
        scene_offsets = self.scene_offsets

        # --- Texture atlas (shared across envs) ---
        # Use the first scene with a texture atlas.  For no-texture mode all
        # scenes return None and the sensor falls back to vertex colours.
        ref_scene = None
        for s in scenes:
            if s.texture_image is not None:
                ref_scene = s
                break
        if ref_scene is None:
            ref_scene = scenes[0]

        texture_image = ref_scene.texture_image
        if texture_image is not None:
            self.global_tensor_dict["CONST_WARP_TEXTURE_IMAGE_TENSOR"] = torch.tensor(
                texture_image, device=self.device, dtype=torch.uint8, requires_grad=False,
            )
        else:
            self.global_tensor_dict["CONST_WARP_TEXTURE_IMAGE_TENSOR"] = None

        self.global_tensor_dict["CONST_WARP_TEXTURE_UV_TENSOR"] = torch.tensor(
            ref_scene.render_vertex_uvs, device=self.device, dtype=torch.float32, requires_grad=False,
        ).contiguous()
        self.global_tensor_dict["CONST_WARP_TEXTURE_VERTEX_NORMAL_TENSOR"] = torch.tensor(
            ref_scene.render_vertex_normals, device=self.device, dtype=torch.float32, requires_grad=False,
        ).contiguous()
        self.global_tensor_dict["CONST_WARP_TEXTURE_BASE_COLOR_FACTOR"] = torch.tensor(
            ref_scene.base_color_factor, device=self.device, dtype=torch.float32, requires_grad=False,
        )

        # --- Per-env warp meshes ---
        # Pre-build GPU tensors per scene to avoid redundant uploads.
        scene_gpu = {}
        for idx, scene in enumerate(scenes):
            offset = scene_offsets[idx]
            translated = scene.render_vertices + offset[None, :]
            scene_gpu[idx] = {
                "vertices": torch.tensor(translated, device=self.device, dtype=torch.float32, requires_grad=False),
                "faces": torch.tensor(scene.render_faces, device=self.device, dtype=torch.int32, requires_grad=False),
                "uv": torch.tensor(scene.render_vertex_uvs, device=self.device, dtype=torch.float32, requires_grad=False).contiguous(),
                "normals": torch.tensor(scene.render_vertex_normals, device=self.device, dtype=torch.float32, requires_grad=False).contiguous(),
            }

        self.warp_mesh_per_env = []
        self.warp_mesh_id_list = []

        # Per-env UV and normal tensors — needed when scenes differ in topology.
        per_env_uv_list = []
        per_env_normal_list = []

        for env_id in range(self.num_envs):
            sid = env_to_scene[env_id]
            gpu = scene_gpu[sid]
            verts = gpu["vertices"]
            faces = gpu["faces"]
            velocity = torch.zeros((verts.shape[0], 3), device=self.device, dtype=torch.float32, requires_grad=False)

            wp_mesh = wp.Mesh(
                points=wp.from_torch(verts, dtype=wp.vec3),
                indices=wp.from_torch(faces.flatten(), dtype=wp.int32),
                velocities=wp.from_torch(velocity, dtype=wp.vec3),
            )
            self.warp_mesh_per_env.append(wp_mesh)
            self.warp_mesh_id_list.append(wp_mesh.id)
            per_env_uv_list.append(gpu["uv"])
            per_env_normal_list.append(gpu["normals"])

        # Store per-env UV/normal lists for sensors that index by env.
        self.global_tensor_dict["CONST_WARP_TEXTURE_UV_PER_ENV"] = per_env_uv_list
        self.global_tensor_dict["CONST_WARP_TEXTURE_NORMAL_PER_ENV"] = per_env_normal_list

        self.CONST_WARP_MESH_ID_LIST = self.warp_mesh_id_list
        self.CONST_WARP_MESH_PER_ENV = self.warp_mesh_per_env
        self.global_tensor_dict["CONST_WARP_MESH_ID_LIST"] = self.CONST_WARP_MESH_ID_LIST
        self.global_tensor_dict["CONST_WARP_MESH_PER_ENV"] = self.CONST_WARP_MESH_PER_ENV
        self.global_tensor_dict["CONST_GLOBAL_VERTEX_TO_ASSET_INDEX_TENSOR"] = None
        self.global_tensor_dict["VERTEX_MAPS_PER_ENV_ORIGINAL"] = None
        self.global_tensor_dict["CONST_GLOBAL_VERTEX_COLOR_TENSOR"] = None
        self.global_tensor_dict["CONST_GLOBAL_VERTEX_COLOR_OFFSETS"] = None
        return 1

    def prepare_for_simulation(self, global_tensor_dict):
        logger.debug("Preparing for simulation")
        self.global_tensor_dict = global_tensor_dict
        if self.static_scene is not None and self.global_vertex_counter > 0:
            raise NotImplementedError("Combining static GLB scenes with dynamic warp-managed assets is not implemented yet.")

        # Multi-scene path: per-env scene assignment.
        if getattr(self, "multi_scenes", None) is not None:
            return self._prepare_multi_scene()

        # Static scene path: populates CONST_WARP_TEXTURE_* keys for the sensor.
        # Dynamic asset path (below): populates CONST_GLOBAL_VERTEX_COLOR_* keys instead.
        if self.static_scene is not None:
            texture_image = self.static_scene.texture_image
            if texture_image is not None:
                # Store atlas as uint8 to save ~4x GPU memory (~505 MB vs ~2023 MB).
                # The Warp kernel converts to float inline at sample time.
                self.global_tensor_dict["CONST_WARP_TEXTURE_IMAGE_TENSOR"] = torch.tensor(
                    texture_image,
                    device=self.device,
                    dtype=torch.uint8,
                    requires_grad=False,
                )
            else:
                self.global_tensor_dict["CONST_WARP_TEXTURE_IMAGE_TENSOR"] = None

            self.global_tensor_dict["CONST_WARP_TEXTURE_UV_TENSOR"] = torch.tensor(
                self.static_scene.render_vertex_uvs,
                device=self.device,
                dtype=torch.float32,
                requires_grad=False,
            ).contiguous()
            self.global_tensor_dict["CONST_WARP_TEXTURE_VERTEX_NORMAL_TENSOR"] = torch.tensor(
                self.static_scene.render_vertex_normals,
                device=self.device,
                dtype=torch.float32,
                requires_grad=False,
            ).contiguous()
            self.global_tensor_dict["CONST_WARP_TEXTURE_BASE_COLOR_FACTOR"] = torch.tensor(
                self.static_scene.base_color_factor,
                device=self.device,
                dtype=torch.float32,
                requires_grad=False,
            )

            self.warp_mesh_per_env = []
            self.warp_mesh_id_list = []
            for env_origin in self.env_origins:
                translated_vertices = self.static_scene.render_vertices + env_origin[None, :]
                vertex_tensor = torch.tensor(
                    translated_vertices,
                    device=self.device,
                    dtype=torch.float32,
                    requires_grad=False,
                )
                face_tensor = torch.tensor(
                    self.static_scene.render_faces,
                    device=self.device,
                    dtype=torch.int32,
                    requires_grad=False,
                )
                velocity_tensor = torch.zeros(
                    (translated_vertices.shape[0], 3),
                    device=self.device,
                    dtype=torch.float32,
                    requires_grad=False,
                )

                wp_mesh = wp.Mesh(
                    points=wp.from_torch(vertex_tensor, dtype=wp.vec3),
                    indices=wp.from_torch(face_tensor.flatten(), dtype=wp.int32),
                    velocities=wp.from_torch(velocity_tensor, dtype=wp.vec3),
                )
                self.warp_mesh_per_env.append(wp_mesh)
                self.warp_mesh_id_list.append(wp_mesh.id)

            self.CONST_WARP_MESH_ID_LIST = self.warp_mesh_id_list
            self.CONST_WARP_MESH_PER_ENV = self.warp_mesh_per_env
            self.global_tensor_dict["CONST_WARP_MESH_ID_LIST"] = self.CONST_WARP_MESH_ID_LIST
            self.global_tensor_dict["CONST_WARP_MESH_PER_ENV"] = self.CONST_WARP_MESH_PER_ENV
            self.global_tensor_dict["CONST_GLOBAL_VERTEX_TO_ASSET_INDEX_TENSOR"] = None
            self.global_tensor_dict["VERTEX_MAPS_PER_ENV_ORIGINAL"] = None
            self.global_tensor_dict["CONST_GLOBAL_VERTEX_COLOR_TENSOR"] = None
            self.global_tensor_dict["CONST_GLOBAL_VERTEX_COLOR_OFFSETS"] = None
            return 1

        if self.global_vertex_counter == 0:
            logger.warning(
                "No assets have been added to the environment. Skipping preparation for simulation"
            )
            self.global_tensor_dict["CONST_WARP_MESH_ID_LIST"] = None
            self.global_tensor_dict["CONST_WARP_MESH_PER_ENV"] = None
            self.global_tensor_dict["CONST_GLOBAL_VERTEX_TO_ASSET_INDEX_TENSOR"] = None
            self.global_tensor_dict["VERTEX_MAPS_PER_ENV_ORIGINAL"] = None
            self.global_tensor_dict["CONST_GLOBAL_VERTEX_COLOR_TENSOR"] = None
            self.global_tensor_dict["CONST_GLOBAL_VERTEX_COLOR_OFFSETS"] = None
            self.global_tensor_dict["CONST_WARP_TEXTURE_IMAGE_TENSOR"] = None
            self.global_tensor_dict["CONST_WARP_TEXTURE_UV_TENSOR"] = None
            self.global_tensor_dict["CONST_WARP_TEXTURE_VERTEX_NORMAL_TENSOR"] = None
            self.global_tensor_dict["CONST_WARP_TEXTURE_BASE_COLOR_FACTOR"] = None
            return 1

        self.global_vertex_to_asset_index_tensor = torch.tensor(
            self.global_vertex_to_asset_index_map,
            device=self.device,
            requires_grad=False,
        )
        self.vertex_maps_per_env_original = torch.zeros(
            (self.global_vertex_counter, 3), device=self.device, requires_grad=False
        )
        # updated vertex maps are used for the warp environment
        self.vertex_maps_per_env_updated = self.vertex_maps_per_env_original.clone()

        ## unify env meshes
        logger.debug("Unifying environment meshes")
        for i in range(len(self.env_meshes)):
            self.global_env_mesh_list.append(tm.util.concatenate(self.env_meshes[i]))
        logger.debug("[DONE] Unifying environment meshes")

        # Build one contiguous RGB buffer aligned with the per-env unified mesh vertices.
        global_vertex_color_list = []
        global_vertex_color_offsets = []
        color_vertex_offset = 0
        for env_mesh in self.global_env_mesh_list:
            global_vertex_color_offsets.append(color_vertex_offset)
            vertex_colors = None
            if hasattr(env_mesh, "visual") and hasattr(env_mesh.visual, "vertex_colors"):
                vertex_colors = env_mesh.visual.vertex_colors

            if vertex_colors is not None and len(vertex_colors) == len(env_mesh.vertices):
                colors = np.asarray(vertex_colors[:, :3], dtype=np.float32) / 255.0
            else:
                colors = np.ones((len(env_mesh.vertices), 3), dtype=np.float32)

            global_vertex_color_list.append(colors)
            color_vertex_offset += len(env_mesh.vertices)

        self.global_vertex_color_tensor = torch.tensor(
            np.concatenate(global_vertex_color_list, axis=0),
            device=self.device,
            dtype=torch.float32,
            requires_grad=False,
        )
        self.global_vertex_color_offsets = torch.tensor(
            global_vertex_color_offsets,
            device=self.device,
            dtype=torch.int32,
            requires_grad=False,
        )
        self.global_tensor_dict["CONST_GLOBAL_VERTEX_COLOR_TENSOR"] = self.global_vertex_color_tensor
        self.global_tensor_dict["CONST_GLOBAL_VERTEX_COLOR_OFFSETS"] = self.global_vertex_color_offsets

        # prepare warp meshes
        logger.debug("Creating warp meshes")
        vertex_iterator = 0
        for env_mesh in self.global_env_mesh_list:
            self.vertex_maps_per_env_original[
                vertex_iterator : vertex_iterator + len(env_mesh.vertices)
            ] = torch.tensor(env_mesh.vertices, device=self.device, requires_grad=False)
            faces_tensor = torch.tensor(
                env_mesh.faces,
                device=self.device,
                requires_grad=False,
                dtype=torch.int32,
            )
            vertex_velocities = torch.zeros(
                len(env_mesh.vertices), 3, device=self.device, requires_grad=False
            )
            segmentation_tensor = torch.tensor(
                self.global_vertex_segmentation_list[
                    vertex_iterator : vertex_iterator + len(env_mesh.vertices)
                ],
                device=self.device,
                requires_grad=False,
            )
            # we hijack this field and use it for segmentation
            vertex_velocities[:, 0] = segmentation_tensor

            vertex_vec3_array = wp.from_torch(
                self.vertex_maps_per_env_updated[
                    vertex_iterator : vertex_iterator + len(env_mesh.vertices)
                ],
                dtype=wp.vec3,
            )
            faces_wp_int32_array = wp.from_torch(faces_tensor.flatten(), dtype=wp.int32)
            velocities_vec3_array = wp.from_torch(vertex_velocities, dtype=wp.vec3)

            wp_mesh = wp.Mesh(
                points=vertex_vec3_array,
                indices=faces_wp_int32_array,
                velocities=velocities_vec3_array,
            )

            self.warp_mesh_per_env.append(wp_mesh)
            self.warp_mesh_id_list.append(wp_mesh.id)
            vertex_iterator += len(env_mesh.vertices)

        logger.debug("[DONE] Creating warp meshes")
        # define consts so that they can be accessed only after the environment has been prepared
        self.CONST_WARP_MESH_ID_LIST = self.warp_mesh_id_list
        self.CONST_WARP_MESH_PER_ENV = self.warp_mesh_per_env
        self.CONST_GLOBAL_VERTEX_TO_ASSET_INDEX_TENSOR = self.global_vertex_to_asset_index_tensor
        self.VERTEX_MAPS_PER_ENV_ORIGINAL = self.vertex_maps_per_env_original

        self.global_tensor_dict["CONST_WARP_MESH_ID_LIST"] = self.CONST_WARP_MESH_ID_LIST
        self.global_tensor_dict["CONST_WARP_MESH_PER_ENV"] = self.CONST_WARP_MESH_PER_ENV
        self.global_tensor_dict["CONST_GLOBAL_VERTEX_TO_ASSET_INDEX_TENSOR"] = (
            self.CONST_GLOBAL_VERTEX_TO_ASSET_INDEX_TENSOR
        )
        self.global_tensor_dict["VERTEX_MAPS_PER_ENV_ORIGINAL"] = self.VERTEX_MAPS_PER_ENV_ORIGINAL

        self.unfolded_env_vec_root_tensor = self.global_tensor_dict[
            "unfolded_env_asset_state_tensor"
        ]
        return 1
