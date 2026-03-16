import numpy as np
import trimesh as tm
import math

from PIL import Image


class StaticSceneGLB:
    def __init__(self, scene_config):
        self.cfg = scene_config
        self.file_path = self.cfg.file
        self.texture_max_resolution = getattr(self.cfg, "texture_max_resolution", 4096)
        self.atlas_tile_resolution = getattr(self.cfg, "atlas_tile_resolution", 256)
        self.enable_texture_rendering = getattr(self.cfg, "enable_texture_rendering", True)

        render_scene = self._load_scene(self.file_path)
        (
            self.render_vertices,
            self.render_faces,
            self.render_vertex_uvs,
            self.vertex_colors,
            self.texture_image,
        ) = self._build_render_buffers(render_scene)
        self.base_color_factor = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        collision_file = getattr(self.cfg, "collision_file", None)
        collision_scene = render_scene if collision_file is None else self._load_scene(collision_file)
        collision_mesh = self._build_collision_mesh(collision_scene)
        self.collision_vertices = np.asarray(collision_mesh.vertices, dtype=np.float32)
        self.collision_faces = np.asarray(collision_mesh.faces, dtype=np.uint32)

        self.bounds = np.asarray(collision_mesh.bounds, dtype=np.float32)

    def _load_scene(self, file_path):
        scene = tm.load(file_path, force="scene")
        return scene

    def _get_scene_meshes(self, scene):
        meshes = []
        for mesh in scene.dump(concatenate=False):
            if not isinstance(mesh, tm.Trimesh):
                continue

            mesh = mesh.copy()
            mesh.remove_unreferenced_vertices()

            scale = float(getattr(self.cfg, "scale", 1.0))
            if scale != 1.0:
                mesh.vertices *= scale

            translation = np.asarray(
                getattr(self.cfg, "translation", [0.0, 0.0, 0.0]), dtype=np.float32
            )
            if np.linalg.norm(translation) > 0.0:
                mesh.vertices += translation

            meshes.append(mesh)

        if len(meshes) == 0:
            raise ValueError(f"No triangle meshes found in static GLB scene: {self.file_path}")

        return meshes

    def _build_collision_mesh(self, scene):
        return tm.util.concatenate(self._get_scene_meshes(scene))

    def _build_render_buffers(self, scene):
        meshes = self._get_scene_meshes(scene)
        atlas_cols = int(math.ceil(math.sqrt(len(meshes))))
        atlas_rows = int(math.ceil(len(meshes) / atlas_cols))
        tile_resolution = int(self.atlas_tile_resolution)

        atlas_image = None
        if self.enable_texture_rendering:
            atlas_image = np.zeros(
                (atlas_rows * tile_resolution, atlas_cols * tile_resolution, 3),
                dtype=np.uint8,
            )

        merged_vertices = []
        merged_faces = []
        merged_uvs = []
        merged_vertex_colors = []
        vertex_offset = 0

        for mesh_index, mesh in enumerate(meshes):
            visual = getattr(mesh, "visual", None)
            material = getattr(visual, "material", None)
            base_color_factor = self._extract_base_color_factor(material)
            vertex_colors = self._extract_vertex_colors(mesh, base_color_factor)
            local_uvs = self._extract_uvs(mesh)

            if atlas_image is not None:
                tile = self._extract_texture_tile(material, base_color_factor, tile_resolution)
                atlas_row = mesh_index // atlas_cols
                atlas_col = mesh_index % atlas_cols
                row_start = atlas_row * tile_resolution
                col_start = atlas_col * tile_resolution
                atlas_image[
                    row_start : row_start + tile_resolution,
                    col_start : col_start + tile_resolution,
                ] = tile

                remapped_uvs = np.empty_like(local_uvs)
                remapped_uvs[:, 0] = (atlas_col + np.clip(local_uvs[:, 0], 0.0, 1.0)) / atlas_cols
                remapped_uvs[:, 1] = (
                    atlas_rows - atlas_row - 1 + np.clip(local_uvs[:, 1], 0.0, 1.0)
                ) / atlas_rows
            else:
                remapped_uvs = np.zeros((len(mesh.vertices), 2), dtype=np.float32)

            merged_vertices.append(np.asarray(mesh.vertices, dtype=np.float32))
            merged_faces.append(np.asarray(mesh.faces, dtype=np.uint32) + vertex_offset)
            merged_uvs.append(remapped_uvs)
            merged_vertex_colors.append(vertex_colors)
            vertex_offset += len(mesh.vertices)

        return (
            np.concatenate(merged_vertices, axis=0),
            np.concatenate(merged_faces, axis=0),
            np.concatenate(merged_uvs, axis=0),
            np.concatenate(merged_vertex_colors, axis=0),
            atlas_image,
        )

    def _extract_uvs(self, mesh):
        visual = getattr(mesh, "visual", None)
        uv = getattr(visual, "uv", None)
        if uv is None or len(uv) != len(mesh.vertices):
            return np.zeros((len(mesh.vertices), 2), dtype=np.float32)
        return np.asarray(uv, dtype=np.float32)

    def _extract_vertex_colors(self, mesh, base_color_factor):
        visual = getattr(mesh, "visual", None)
        vertex_colors = getattr(visual, "vertex_colors", None)
        if vertex_colors is not None and len(vertex_colors) == len(mesh.vertices):
            return np.asarray(vertex_colors[:, :3], dtype=np.float32) / 255.0

        return np.repeat(base_color_factor[None, :], len(mesh.vertices), axis=0).astype(np.float32)

    def _extract_base_color_factor(self, material):
        factor = getattr(material, "baseColorFactor", None)
        if factor is None:
            return np.array([1.0, 1.0, 1.0], dtype=np.float32)
        if np.max(factor[:3]) > 1.0:
            return np.asarray(factor[:3], dtype=np.float32) / 255.0
        return np.asarray(factor[:3], dtype=np.float32)

    def _extract_texture_tile(self, material, base_color_factor, tile_resolution):
        texture = getattr(material, "baseColorTexture", None)
        if texture is None:
            tile = np.broadcast_to(
                np.round(255.0 * base_color_factor).astype(np.uint8),
                (tile_resolution, tile_resolution, 3),
            ).copy()
            return tile

        image = (
            texture.convert("RGB")
            if isinstance(texture, Image.Image)
            else Image.fromarray(np.asarray(texture)).convert("RGB")
        )
        if image.size != (tile_resolution, tile_resolution):
            image = image.resize((tile_resolution, tile_resolution), Image.BILINEAR)

        tile = np.asarray(image, dtype=np.float32)
        tile *= base_color_factor[None, None, :]
        return np.clip(tile, 0.0, 255.0).astype(np.uint8)
