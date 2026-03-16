import hashlib
import math
import numpy as np
import trimesh as tm
from PIL import Image


class StaticSceneGLB:
    """Load a GLB/OBJ static scene for Warp rendering and Isaac Gym collision.

    Unique textures are identified by content hash so the GPU atlas stores no
    duplicate tiles.  Each unique texture is packed at
    ``texture_atlas_tile_size x texture_atlas_tile_size`` pixels (default 2048).

    Memory budget for a Matterport scene (41 unique textures in a 7x6 grid):
        tile_size   atlas pixels        float32 GPU memory
        2048        14336 x 12288       ~ 2.1 GB
        1024         7168 x  6144       ~530 MB
         512         3584 x  3072       ~132 MB

    UV remapping accounts for the Warp kernel's (1-v) V-flip so that GLTF
    textures render correctly without any extra flipping step.
    """

    def __init__(self, scene_config):
        self.cfg = scene_config
        self.file_path = self.cfg.file
        self.enable_texture_rendering = getattr(self.cfg, "enable_texture_rendering", True)
        self.tile_size = int(getattr(self.cfg, "texture_atlas_tile_size", 2048))

        render_verts, render_faces, render_uvs, atlas, vcols = self._load_and_build(
            self.file_path
        )
        self.render_vertices = render_verts                # (V, 3) float32
        self.render_faces = render_faces.astype(np.int32)  # (F, 3) int32 for Warp
        self.render_vertex_uvs = render_uvs                # (V, 2) float32
        self.texture_image = atlas                         # (H, W, 3) uint8 or None
        self.base_color_factor = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self.vertex_colors = vcols                         # (V, 3) float32

        collision_file = getattr(self.cfg, "collision_file", None)
        if collision_file is not None:
            c_v, c_f, _, _, _ = self._load_and_build(collision_file)
        else:
            c_v, c_f = render_verts, render_faces
        self.collision_vertices = c_v.astype(np.float32)
        self.collision_faces = c_f.astype(np.uint32)
        self.bounds = np.array(
            [self.collision_vertices.min(0), self.collision_vertices.max(0)],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Top-level loader
    # ------------------------------------------------------------------

    def _load_and_build(self, file_path):
        scene = tm.load(file_path, force="scene")
        geoms = [m for m in scene.dump(concatenate=False) if isinstance(m, tm.Trimesh)]
        if not geoms:
            raise ValueError(f"No triangle meshes found in {file_path}")

        scale = float(getattr(self.cfg, "scale", 1.0))
        translation = np.asarray(
            getattr(self.cfg, "translation", [0.0, 0.0, 0.0]), dtype=np.float32
        )
        for g in geoms:
            g.remove_unreferenced_vertices()
            if scale != 1.0:
                g.vertices = (g.vertices * scale).astype(np.float32)
            if np.any(translation != 0.0):
                g.vertices = (g.vertices + translation).astype(np.float32)

        if self.enable_texture_rendering:
            return self._build_textured(geoms)
        return self._build_vertex_color(geoms)

    # ------------------------------------------------------------------
    # Textured path: dedup -> atlas -> UV remap
    # ------------------------------------------------------------------

    def _build_textured(self, geoms):
        tile_size = self.tile_size

        # Pass 1: collect unique textures keyed by MD5 content hash
        hash_to_idx = {}
        unique_tiles = []   # each (tile_size, tile_size, 3) uint8
        submesh_tile = []   # atlas tile index per geom, -1 if untextured

        for g in geoms:
            mat = getattr(getattr(g, "visual", None), "material", None)
            tex = getattr(mat, "baseColorTexture", None)
            if tex is None:
                submesh_tile.append(-1)
                continue
            arr = np.asarray(tex)
            key = hashlib.md5(arr.tobytes()).hexdigest()
            if key not in hash_to_idx:
                hash_to_idx[key] = len(unique_tiles)
                img = (
                    tex if isinstance(tex, Image.Image) else Image.fromarray(arr)
                ).convert("RGB")
                if img.size != (tile_size, tile_size):
                    img = img.resize((tile_size, tile_size), Image.LANCZOS)
                unique_tiles.append(np.asarray(img, dtype=np.uint8))
            submesh_tile.append(hash_to_idx[key])

        # Pack tiles into a square-ish grid
        n = len(unique_tiles)
        atlas, cols, rows = None, 1, 1
        if n > 0:
            cols = int(math.ceil(math.sqrt(n)))
            rows = int(math.ceil(n / cols))
            atlas = np.zeros((rows * tile_size, cols * tile_size, 3), dtype=np.uint8)
            for idx, tile in enumerate(unique_tiles):
                r, c = divmod(idx, cols)
                atlas[r * tile_size:(r + 1) * tile_size,
                      c * tile_size:(c + 1) * tile_size] = tile
            print(
                f"[StaticSceneGLB] {len(geoms)} submeshes -> {n} unique textures "
                f"-> {cols}x{rows} grid at {tile_size} px/tile "
                f"= {atlas.shape[1]}x{atlas.shape[0]} px "
                f"({atlas.nbytes // (1024 * 1024)} MB uint8, "
                f"~{atlas.nbytes * 4 // (1024 * 1024)} MB float32 on GPU)"
            )

        # Pass 2: merge vertices and remap UVs.
        # The Warp kernel reads:  pixel_row = (1 - atlas_v) * atlas_H
        # Tile idx stored at grid row r occupies atlas rows [r*T, (r+1)*T).
        # For GLTF v_orig (v=0 -> bottom, v=1 -> top of visible image):
        #   desired pixel_row = r*T + (1 - v_orig)*T
        #   (1 - atlas_v) * rows*T = (r + 1 - v_orig)*T
        #   atlas_v = (rows - 1 - r + v_orig) / rows
        all_v, all_f, all_uv = [], [], []
        v_off = 0
        for i, g in enumerate(geoms):
            v = np.asarray(g.vertices, dtype=np.float32)
            f = np.asarray(g.faces, dtype=np.int32) + v_off
            uv = self._geom_uvs(g)
            tidx = submesh_tile[i]
            if tidx >= 0 and n > 0:
                r, c = divmod(tidx, cols)
                u_a = (c + uv[:, 0] % 1.0) / cols
                v_a = (rows - 1 - r + uv[:, 1] % 1.0) / rows
                uv_out = np.stack([u_a, v_a], axis=1).astype(np.float32)
            else:
                uv_out = np.zeros((len(v), 2), dtype=np.float32)
            all_v.append(v)
            all_f.append(f)
            all_uv.append(uv_out)
            v_off += len(v)

        verts = np.concatenate(all_v)
        faces = np.concatenate(all_f)
        uvs   = np.concatenate(all_uv)
        vcols = np.ones((len(verts), 3), dtype=np.float32)
        return verts, faces, uvs, atlas, vcols

    # ------------------------------------------------------------------
    # Vertex-colour fallback (no textures)
    # ------------------------------------------------------------------

    def _build_vertex_color(self, geoms):
        all_v, all_f, all_vc = [], [], []
        v_off = 0
        for g in geoms:
            v  = np.asarray(g.vertices, dtype=np.float32)
            f  = np.asarray(g.faces, dtype=np.int32) + v_off
            vc = self._geom_vcols(g)
            all_v.append(v)
            all_f.append(f)
            all_vc.append(vc)
            v_off += len(v)
        verts = np.concatenate(all_v)
        faces = np.concatenate(all_f)
        uvs   = np.zeros((len(verts), 2), dtype=np.float32)
        vcols = np.concatenate(all_vc)
        return verts, faces, uvs, None, vcols

    # ------------------------------------------------------------------
    # Per-geometry helpers
    # ------------------------------------------------------------------

    def _geom_uvs(self, g):
        uv = getattr(getattr(g, "visual", None), "uv", None)
        if uv is not None and len(uv) == len(g.vertices):
            return np.asarray(uv, dtype=np.float32)
        return np.zeros((len(g.vertices), 2), dtype=np.float32)

    def _geom_vcols(self, g):
        vis = getattr(g, "visual", None)
        vc  = getattr(vis, "vertex_colors", None)
        if vc is not None and len(vc) == len(g.vertices):
            return np.asarray(vc[:, :3], dtype=np.float32) / 255.0
        mat = getattr(vis, "material", None)
        fac = getattr(mat, "baseColorFactor", None)
        if fac is not None:
            f = np.asarray(fac[:3], dtype=np.float32)
            if f.max() > 1.0:
                f /= 255.0
            return np.broadcast_to(f, (len(g.vertices), 3)).copy()
        return np.ones((len(g.vertices), 3), dtype=np.float32)
