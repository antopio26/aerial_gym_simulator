import warp as wp


class RGBCameraWarpKernels:
    @staticmethod
    @wp.kernel
    def draw_textured_rgbd_kernel(
        mesh_ids: wp.array(dtype=wp.uint64),
        cam_poss: wp.array(dtype=wp.vec3, ndim=2),
        cam_quats: wp.array(dtype=wp.quat, ndim=2),
        K_inv: wp.mat44,
        depth_far_plane: float,
        rgb_far_plane: float,
        rgb_pixels: wp.array(dtype=wp.vec3, ndim=4),
        depth_pixels: wp.array(dtype=float, ndim=4),
        vertex_uvs: wp.array(dtype=wp.vec2),
        vertex_normals: wp.array(dtype=wp.vec3),
        texture_image: wp.array(dtype=wp.uint8, ndim=3),
        texture_width: int,
        texture_height: int,
        base_color_factor: wp.vec3,
        enable_lighting: int,
        debug_uv_checker: int,
        ambient_strength: float,
        light_dir_world: wp.vec3,
        c_x: int,
        c_y: int,
    ):
        env_id, cam_id, x, y = wp.tid()
        mesh = mesh_ids[env_id]
        cam_pos = cam_poss[env_id, cam_id]
        cam_quat = cam_quats[env_id, cam_id]
        cam_coords = wp.vec3(float(x), float(y), 1.0)
        uv_ray = wp.normalize(wp.transform_vector(K_inv, cam_coords))
        ro = cam_pos
        rd = wp.normalize(wp.quat_rotate(cam_quat, uv_ray))
        t = float(0.0)
        u = float(0.0)
        v = float(0.0)
        sign = float(0.0)
        n = wp.vec3()
        f = int(0)
        far_plane = depth_far_plane
        if rgb_far_plane > depth_far_plane:
            far_plane = rgb_far_plane
        dist = depth_far_plane
        color = wp.vec3(0.0, 0.0, 0.0)

        if wp.mesh_query_ray(mesh, ro, rd, far_plane, t, u, v, sign, n, f):
            if t < depth_far_plane:
                dist = t
            if t <= rgb_far_plane:
                mesh_obj = wp.mesh_get(mesh)
                idx0 = mesh_obj.indices[f * 3 + 0]
                idx1 = mesh_obj.indices[f * 3 + 1]
                idx2 = mesh_obj.indices[f * 3 + 2]

                uv0 = vertex_uvs[idx0]
                uv1 = vertex_uvs[idx1]
                uv2 = vertex_uvs[idx2]
                n0 = vertex_normals[idx0]
                n1 = vertex_normals[idx1]
                n2 = vertex_normals[idx2]
                w = 1.0 - u - v
                # Warp barycentrics (u, v, w) map to (uv0, uv1, uv2) for Matterport GLB.
                uv = u * uv0 + v * uv1 + w * uv2
                shading_normal = wp.normalize(w * n0 + u * n1 + v * n2)

                # Bilinear texture sampling with V-flip (GLTF V-axis convention).
                fu = wp.clamp(uv[0], 0.0, 1.0) * float(texture_width - 1)
                fv = wp.clamp(1.0 - uv[1], 0.0, 1.0) * float(texture_height - 1)
                x0 = int(wp.floor(fu))
                y0 = int(wp.floor(fv))
                x1 = wp.min(x0 + 1, texture_width - 1)
                y1 = wp.min(y0 + 1, texture_height - 1)
                tx = fu - float(x0)
                ty = fv - float(y0)
                # uint8 -> float32 inline: 0.003921569 = 1.0 / 255.0
                # Avoids storing a 4x larger float atlas on the GPU.
                c00 = wp.vec3(float(texture_image[y0, x0, 0]) * 0.003921569,
                              float(texture_image[y0, x0, 1]) * 0.003921569,
                              float(texture_image[y0, x0, 2]) * 0.003921569)
                c10 = wp.vec3(float(texture_image[y0, x1, 0]) * 0.003921569,
                              float(texture_image[y0, x1, 1]) * 0.003921569,
                              float(texture_image[y0, x1, 2]) * 0.003921569)
                c01 = wp.vec3(float(texture_image[y1, x0, 0]) * 0.003921569,
                              float(texture_image[y1, x0, 1]) * 0.003921569,
                              float(texture_image[y1, x0, 2]) * 0.003921569)
                c11 = wp.vec3(float(texture_image[y1, x1, 0]) * 0.003921569,
                              float(texture_image[y1, x1, 1]) * 0.003921569,
                              float(texture_image[y1, x1, 2]) * 0.003921569)
                albedo = (
                    (1.0 - tx) * (1.0 - ty) * c00
                    + tx * (1.0 - ty) * c10
                    + (1.0 - tx) * ty * c01
                    + tx * ty * c11
                )
                albedo = wp.cw_mul(albedo, base_color_factor)

                if debug_uv_checker != 0:
                    checker_u = int(wp.floor(uv[0] * 24.0))
                    checker_v = int(wp.floor(uv[1] * 24.0))
                    if (checker_u + checker_v) % 2 == 0:
                        color = wp.vec3(0.95, 0.95, 0.95)
                    else:
                        color = wp.vec3(0.10, 0.10, 0.10)
                elif enable_lighting != 0:
                    l_hat = wp.normalize(light_dir_world)
                    lambert = wp.max(wp.dot(shading_normal, l_hat), 0.0)
                    lit = ambient_strength + (1.0 - ambient_strength) * lambert
                    color = lit * albedo
                else:
                    color = albedo

        rgb_pixels[env_id, cam_id, y, x] = color
        depth_pixels[env_id, cam_id, y, x] = dist

    @staticmethod
    @wp.kernel
    def draw_rgbd_kernel(
        mesh_ids: wp.array(dtype=wp.uint64),
        cam_poss: wp.array(dtype=wp.vec3, ndim=2),
        cam_quats: wp.array(dtype=wp.quat, ndim=2),
        K_inv: wp.mat44,
        depth_far_plane: float,
        rgb_far_plane: float,
        rgb_pixels: wp.array(dtype=wp.vec3, ndim=4),
        depth_pixels: wp.array(dtype=float, ndim=4),
        vertex_colors: wp.array(dtype=wp.vec3),
        vertex_color_offsets: wp.array(dtype=wp.int32),
        enable_lighting: int,
        ambient_strength: float,
        light_dir_world: wp.vec3,
        c_x: int,
        c_y: int,
    ):
        env_id, cam_id, x, y = wp.tid()
        mesh = mesh_ids[env_id]
        cam_pos = cam_poss[env_id, cam_id]
        cam_quat = cam_quats[env_id, cam_id]
        cam_coords = wp.vec3(float(x), float(y), 1.0)
        uv = wp.normalize(wp.transform_vector(K_inv, cam_coords))
        ro = cam_pos
        rd = wp.normalize(wp.quat_rotate(cam_quat, uv))
        t = float(0.0)
        u = float(0.0)
        v = float(0.0)
        sign = float(0.0)
        n = wp.vec3()
        f = int(0)
        far_plane = depth_far_plane
        if rgb_far_plane > depth_far_plane:
            far_plane = rgb_far_plane
        dist = depth_far_plane
        color = wp.vec3(0.0, 0.0, 0.0)

        if wp.mesh_query_ray(mesh, ro, rd, far_plane, t, u, v, sign, n, f):
            if t < depth_far_plane:
                dist = t
            if t <= rgb_far_plane:
                mesh_obj = wp.mesh_get(mesh)
                env_vertex_offset = vertex_color_offsets[env_id]

                # Barycentric interpolation of per-vertex color.
                idx0 = mesh_obj.indices[f * 3 + 0]
                idx1 = mesh_obj.indices[f * 3 + 1]
                idx2 = mesh_obj.indices[f * 3 + 2]
                c0 = vertex_colors[env_vertex_offset + idx0]
                c1 = vertex_colors[env_vertex_offset + idx1]
                c2 = vertex_colors[env_vertex_offset + idx2]

                w = 1.0 - u - v
                albedo = w * c0 + u * c1 + v * c2

                if enable_lighting != 0:
                    n_hat = wp.normalize(n)
                    l_hat = wp.normalize(light_dir_world)
                    lambert = wp.max(wp.dot(n_hat, l_hat), 0.0)
                    lit = ambient_strength + (1.0 - ambient_strength) * lambert
                    color = lit * albedo
                else:
                    color = albedo

        rgb_pixels[env_id, cam_id, y, x] = color
        depth_pixels[env_id, cam_id, y, x] = dist
