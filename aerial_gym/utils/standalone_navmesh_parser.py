import struct
import math
import numpy as np

try:
    import torch
except ImportError:
    print("Warning: PyTorch is not installed. Will fallback to NumPy logic if needed, but PyTorch is recommended.")
    torch = None

class NavMeshSettings:
    def __init__(self, data_bytes=None):
        if data_bytes and len(data_bytes) >= 56:
            # 13 floats (52 bytes) + 4 bools (4 bytes) = 56 bytes
            unpacked = struct.unpack('<13f 4b', data_bytes[:56])
            self.cell_size = unpacked[0]
            self.cell_height = unpacked[1]
            self.agent_height = unpacked[2]
            self.agent_radius = unpacked[3]
            self.agent_max_climb = unpacked[4]
            self.agent_max_slope = unpacked[5]
            self.region_min_size = unpacked[6]
            self.region_merge_size = unpacked[7]
            self.edge_max_len = unpacked[8]
            self.edge_max_error = unpacked[9]
            self.verts_per_poly = unpacked[10]
            self.detail_sample_dist = unpacked[11]
            self.detail_sample_max_error = unpacked[12]
            
            # The last 4 are booleans
            self.filter_low_hanging_obstacles = bool(unpacked[13])
            self.filter_ledge_spans = bool(unpacked[14])
            self.filter_walkable_low_height_spans = bool(unpacked[15])
            self.include_static_objects = bool(unpacked[16])
        else:
            self.cell_size = 0.05
            self.cell_height = 0.2
            self.agent_height = 1.5
            self.agent_radius = 0.1
            self.agent_max_climb = 0.2

class StandaloneNavMesh:
    def __init__(self, filepath):
        self.filepath = filepath
        self.header = {}
        self.settings = NavMeshSettings()
        self.tiles = []
        
        # Extracted geometry
        self.vertices = []
        self.polygons = []
        
        # PyTorch Tensors
        self.pt_vertices = None
        self.pt_polygons = None
        self.pt_poly_areas = None
        self.height_axis = 2
        self.planar_axes = (0, 1)

        self._load_from_file(filepath)
        self._build_pytorch_tensors()

    def _load_from_file(self, path):
        """
        Deserializes the binary NavMesh format defined in Habitat-Sim (Recast/Detour).
        """
        with open(path, 'rb') as f:
            # 1. Read NavMeshSetHeader (40 bytes)
            # int magic, int version, int numTiles
            # dtNavMeshParams (origX, origY, origZ, tileW, tileH, maxTiles, maxPolys)
            header_data = f.read(40)
            if len(header_data) < 40:
                raise ValueError("Invalid NavMesh file: too short")
            
            magic, version, numTiles = struct.unpack('<3i', header_data[:12])
            
            # Magic 'MSET' in little endian is 0x5445534D (1413830228) or 0x4D534554 (1297302868)
            if magic not in (1413830228, 1297302356, 1297302868):
                raise ValueError(f"Invalid NavMesh magic number: {magic}")
            
            params = struct.unpack('<5f 2i', header_data[12:40])
            self.header = {
                'magic': magic,
                'version': version,
                'numTiles': numTiles,
                'orig': (params[0], params[1], params[2]),
                'tile_width': params[3],
                'tile_height': params[4],
                'max_tiles': params[5],
                'max_polys': params[6]
            }
            
            # 2. Read NavMeshSettings if version >= 2 (56 bytes)
            if version >= 2:
                settings_data = f.read(56)
                self.settings = NavMeshSettings(settings_data)
            
            # 3. Read Tiles
            for _ in range(numTiles):
                tile_header_data = f.read(8)
                if len(tile_header_data) < 8:
                    break
                tile_ref, data_size = struct.unpack('<I i', tile_header_data)
                
                # Read raw tile data chunk
                tile_data = f.read(data_size)
                self._parse_tile_data(tile_data)

    def _parse_tile_data(self, tile_data):
        """
        Parses a raw Detour Tile block to extract vertices and polygons.
        Depending on the Detour compilation, dtMeshHeader is about 100 bytes.
        """
        # dtMeshHeader (approx 100 bytes)
        # 15 ints and 10 floats -> 100 bytes. Let's unpack to safely find counts
        if len(tile_data) < 100:
            return
            
        header_vals = struct.unpack('<15i 10f', tile_data[:100])
        magic = header_vals[0]
        vertCount = header_vals[7]
        polyCount = header_vals[6]
        detailMeshCount = header_vals[9]
        detailVertCount = header_vals[10]
        detailTriCount = header_vals[11]

        # In Detour, 'verts' directly follows the header.
        # Vertices are typically represented as 3 floats = 12 bytes each
        offset = 100
        
        # Read vertices
        tile_verts = []
        for v in range(vertCount):
            xyz = struct.unpack('<3f', tile_data[offset:offset+12])
            
            # Rotate 90 degrees on X-axis during extraction
            rotated_xyz = (xyz[0], -xyz[2], xyz[1])
            tile_verts.append(rotated_xyz)
            
            offset += 12
            
        # Add to global vertex array, recording offset
        v_offset = len(self.vertices)
        self.vertices.extend(tile_verts)
        
        # dtPoly is typically 1 (firstLink) + 6*2 (verts) + 6*2 (neis) + 2 (flags) + 1 (area) + 1 (type) = 44 bytes ?
        # Actually standard DT_VERTS_PER_POLYGON is 6 so: 
        # unsigned int firstLink = 4 bytes
        # unsigned short verts[6] = 12 bytes
        # unsigned short neis[6] = 12 bytes
        # unsigned short flags = 2 bytes
        # unsigned char area = 1 byte
        # unsigned char getType() = 1 byte
        # Total = 32 bytes exactly
        
        poly_size = 32
        for p in range(polyCount):
            poly_data = tile_data[offset:offset+poly_size]
            firstLink = struct.unpack('<I', poly_data[:4])[0]
            v_indices = struct.unpack('<6H', poly_data[4:16])
            
            # Extract how many vertices actually belong to this polygon
            vert_count = poly_data[30]
            
            # Collect valid vertices
            poly_verts = []
            for i in range(vert_count):
                vi = v_indices[i]
                poly_verts.append(v_offset + vi)
                    
            if len(poly_verts) >= 3:
                self.polygons.append(poly_verts)
                
            offset += poly_size
        
        # (Remaining arrays like links, detail meshes, BV tree, etc. are skipped because 
        # we only need raw polygons for position validation and sampling)

    def _build_pytorch_tensors(self):
        """
        Convert lists to PyTorch tensors for fast vectorized sampling and checking.
        """
        if torch is None or len(self.vertices) == 0:
            return
            
        # Convert vertices to a FloatTensor
        self.pt_vertices = torch.tensor(self.vertices, dtype=torch.float32)
        
        # Since polygons can be multi-sided (up to 6), we triangulate them
        # (Fan triangulation: v0, v1, v2 / v0, v2, v3 etc.)
        triangles = []
        for poly in self.polygons:
            v0 = poly[0]
            for i in range(1, len(poly) - 1):
                triangles.append([v0, poly[i], poly[i+1]])
                
        if len(triangles) == 0:
            return
            
        self.pt_polygons = torch.tensor(triangles, dtype=torch.long)
        
        # Compute areas of all triangles via cross product
        v0s = self.pt_vertices[self.pt_polygons[:, 0]]
        v1s = self.pt_vertices[self.pt_polygons[:, 1]]
        v2s = self.pt_vertices[self.pt_polygons[:, 2]]
        
        # Cross product magnitude
        cross = torch.cross(v1s - v0s, v2s - v0s, dim=1)
        self.pt_poly_areas = 0.5 * torch.norm(cross, dim=1)

        # Infer up-axis from dominant component of area-weighted triangle normals.
        if self.pt_poly_areas.numel() > 0:
            abs_cross = torch.abs(cross)
            weighted = abs_cross * self.pt_poly_areas.unsqueeze(1)
            dominant_axis = int(torch.argmax(torch.sum(weighted, dim=0)).item())
            self.height_axis = dominant_axis
            self.planar_axes = tuple(ax for ax in (0, 1, 2) if ax != self.height_axis)
        
    def sample_points(self, count, height_offset=None):
        """
        Sample `count` points uniformly from the NavMesh using PyTorch.
        If height_offset is an (min_h, max_h) tuple, uniform random vertical offsets within this range are added.
        """
        if torch is None or self.pt_polygons is None:
            raise RuntimeError("PyTorch is not available or Navmesh is empty.")
            
        # 1. Pick `count` triangles weighted by their area
        # Use multinomial sampling
        tri_indices = torch.multinomial(self.pt_poly_areas, count, replacement=True)
        
        # 2. Get the vertices of selected triangles
        selected_tris = self.pt_polygons[tri_indices]
        v0s = self.pt_vertices[selected_tris[:, 0]]
        v1s = self.pt_vertices[selected_tris[:, 1]]
        v2s = self.pt_vertices[selected_tris[:, 2]]
        
        # 3. Generate random Barycentric coordinates for each sample
        sample_device = self.pt_vertices.device
        u = torch.rand(count, 1, device=sample_device)
        v = torch.rand(count, 1, device=sample_device)
        
        # If u + v > 1, reflect it back to stay inside the triangle
        mask = (u + v) > 1.0
        u[mask] = 1.0 - u[mask]
        v[mask] = 1.0 - v[mask]
        
        # 4. Compute final 3D coordinates
        w = 1.0 - u - v
        sampled_points = (w * v0s) + (u * v1s) + (v * v2s)
        
        if height_offset is not None and isinstance(height_offset, tuple):
            min_h, max_h = height_offset
            # Add uniform random offsets in [min_h, max_h] to the height axis.
            offsets = (max_h - min_h) * torch.rand(count, device=sample_device) + min_h
            sampled_points[:, self.height_axis] += offsets
            
        return sampled_points

    def _compute_edge_padding_mask(self, sampled_points, selected_tris, edge_padding):
        """
        Returns a boolean mask marking points whose 2D distance to each selected
        triangle edge is at least `edge_padding`.
        """
        if edge_padding <= 0.0:
            return torch.ones(sampled_points.shape[0], dtype=torch.bool, device=sampled_points.device)

        planar = list(self.planar_axes)
        pts = sampled_points[:, planar]
        v0 = self.pt_vertices[selected_tris[:, 0]][:, planar]
        v1 = self.pt_vertices[selected_tris[:, 1]][:, planar]
        v2 = self.pt_vertices[selected_tris[:, 2]][:, planar]

        edges_a = torch.stack((v0, v1, v2), dim=1)
        edges_b = torch.stack((v1, v2, v0), dim=1)

        ab = edges_b - edges_a
        ap = pts.unsqueeze(1) - edges_a
        ab_sq = torch.sum(ab * ab, dim=2)

        # Handle potential degenerate edges safely.
        ab_sq = torch.clamp(ab_sq, min=1.0e-12)
        t = torch.sum(ap * ab, dim=2) / ab_sq
        t = torch.clamp(t, 0.0, 1.0)
        closest = edges_a + t.unsqueeze(-1) * ab
        dists = torch.norm(pts.unsqueeze(1) - closest, dim=2)
        min_dist = torch.min(dists, dim=1)[0]
        return min_dist >= edge_padding

    def sample_points_with_padding(
        self,
        count,
        height_offset=None,
        edge_padding=0.0,
        oversample_factor=4,
        max_resample_rounds=8,
        strict_edge_padding=True,
        min_edge_padding_ratio=0.35,
        padding_relaxation_factor=0.70,
        max_padding_relax_rounds=3,
    ):
        """
        Sample points from the NavMesh while enforcing a 2D minimum distance from
        triangle edges, useful for obstacle-safe spawn/goal placement.
        """
        if torch is None or self.pt_polygons is None:
            raise RuntimeError("PyTorch is not available or Navmesh is empty.")

        sample_device = self.pt_vertices.device

        if count <= 0:
            return torch.zeros((0, 3), dtype=torch.float32, device=sample_device)

        if edge_padding <= 0.0:
            return self.sample_points(count=count, height_offset=height_offset)

        accepted = torch.zeros((0, 3), dtype=torch.float32, device=sample_device)
        candidate_count = max(int(count * oversample_factor), count)

        min_edge_padding_ratio = float(np.clip(min_edge_padding_ratio, 0.0, 1.0))
        padding_relaxation_factor = float(np.clip(padding_relaxation_factor, 0.05, 0.99))
        target_min_padding = edge_padding * min_edge_padding_ratio if strict_edge_padding else 0.0
        curr_padding = float(edge_padding)

        for relax_round in range(max_padding_relax_rounds + 1):
            accepted_chunks = []
            accepted_count = 0

            for _ in range(max_resample_rounds):
                tri_indices = torch.multinomial(self.pt_poly_areas, candidate_count, replacement=True)

                selected_tris = self.pt_polygons[tri_indices]
                v0s = self.pt_vertices[selected_tris[:, 0]]
                v1s = self.pt_vertices[selected_tris[:, 1]]
                v2s = self.pt_vertices[selected_tris[:, 2]]

                u = torch.rand(candidate_count, 1, device=sample_device)
                v = torch.rand(candidate_count, 1, device=sample_device)
                mask_uv = (u + v) > 1.0
                u[mask_uv] = 1.0 - u[mask_uv]
                v[mask_uv] = 1.0 - v[mask_uv]
                w = 1.0 - u - v

                candidates = (w * v0s) + (u * v1s) + (v * v2s)

                if height_offset is not None and isinstance(height_offset, tuple):
                    min_h, max_h = height_offset
                    offsets = (max_h - min_h) * torch.rand(candidate_count, device=sample_device) + min_h
                    candidates[:, self.height_axis] += offsets

                keep_mask = self._compute_edge_padding_mask(candidates, selected_tris, curr_padding)
                kept = candidates[keep_mask]
                if kept.shape[0] > 0:
                    accepted_chunks.append(kept)
                    accepted_count += kept.shape[0]
                if accepted_count >= count:
                    break

            if accepted_count > 0:
                accepted = torch.cat(accepted_chunks, dim=0)
                if accepted.shape[0] >= count:
                    return accepted[:count]

            if relax_round >= max_padding_relax_rounds:
                break

            next_padding = curr_padding * padding_relaxation_factor
            curr_padding = max(target_min_padding, next_padding)

        if accepted.shape[0] > 0:
            # Keep the same safety distribution when we are slightly short.
            shortfall = count - accepted.shape[0]
            if shortfall > 0:
                replay_ids = torch.randint(0, accepted.shape[0], (shortfall,), device=sample_device)
                accepted = torch.cat((accepted, accepted[replay_ids]), dim=0)
            return accepted[:count]

        print(
            f"Warning: could not find any valid navmesh samples with edge_padding={edge_padding:.3f}. "
            "Falling back to unpadded sampling. Consider reducing edge padding."
        )
        return self.sample_points(count=count, height_offset=height_offset)

    def is_navigable(self, points, height_tol=0.5):
        """
        Checks if a batched set of PyTorch positions (N x 3) is navigable.
        Checks footprint inside Triangles + Height tolerance constraint.
        """
        if torch is None or self.pt_polygons is None:
            return [False] * points.shape[0]

        # Extract planar coordinates in the walkable plane.
        planar = list(self.planar_axes)
        pts_2d = points[:, planar]
        
        # Extract triangle vertices in 2D
        v0s = self.pt_vertices[self.pt_polygons[:, 0]][:, planar]
        v1s = self.pt_vertices[self.pt_polygons[:, 1]][:, planar]
        v2s = self.pt_vertices[self.pt_polygons[:, 2]][:, planar]
        
        # We need a robust vectorized approach. 
        # For simplicity in this script, we can do pairwise broadcasting or KD-Tree based.
        # This naive O(N*Tri) approach verifies if points land inside a face footprint.
        
        # [N, num_triangles, 2]
        pts_expanded = pts_2d.unsqueeze(1) 
        
        # Edge vectors
        e0 = v1s - v0s  # [num_triangles, 2]
        e1 = v2s - v1s
        e2 = v0s - v2s
        
        # Point vectors to edge starts
        # [N, num_triangles, 2]
        v0_to_p = pts_expanded - v0s.unsqueeze(0)
        v1_to_p = pts_expanded - v1s.unsqueeze(0)
        v2_to_p = pts_expanded - v2s.unsqueeze(0)
        
        # 2D cross products to check interior (all must have same sign)
        # cp1: e0 x v0p
        cp1 = e0[..., 0] * v0_to_p[..., 1] - e0[..., 1] * v0_to_p[..., 0]
        cp2 = e1[..., 0] * v1_to_p[..., 1] - e1[..., 1] * v1_to_p[..., 0]
        cp3 = e2[..., 0] * v2_to_p[..., 1] - e2[..., 1] * v2_to_p[..., 0]
        
        inside_mask = (cp1 >= 0) & (cp2 >= 0) & (cp3 >= 0)
        inside_mask = inside_mask | ((cp1 <= 0) & (cp2 <= 0) & (cp3 <= 0))
        
        navigable = torch.zeros(points.shape[0], dtype=torch.bool)
        
        # Now, for points that fall inside the footprint, check clearance along height axis.
        for i in range(points.shape[0]):
            valid_tris = inside_mask[i].nonzero(as_tuple=True)[0]
            if len(valid_tris) == 0:
                navigable[i] = False
                continue
                
            # Grab heights on these triangles and verify bounding 
            # (Just taking max vertex height for demonstration. True validation uses barycentric Y interp)
            h_max = torch.max(self.pt_vertices[self.pt_polygons[valid_tris], self.height_axis], dim=1)[0]
            h_min = torch.min(self.pt_vertices[self.pt_polygons[valid_tris], self.height_axis], dim=1)[0]
            
            p_h = points[i, self.height_axis]
            if torch.any((p_h >= h_min - height_tol) & (p_h <= h_max + height_tol)):
                navigable[i] = True

        return navigable

if __name__ == '__main__':
    print("Standalone NavMesh Parser is ready.")
    print("Example usage:")
    print("  navmesh = StandaloneNavMesh('path_to_scene.navmesh')")
    print("  points = navmesh.sample_points(100)")
    print("  valid_mask = navmesh.is_navigable(points)")
