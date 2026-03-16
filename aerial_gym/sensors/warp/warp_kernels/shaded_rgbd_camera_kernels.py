import warp as wp


class ShadedRGBCameraWarpKernels:
    @staticmethod
    @wp.kernel
    def draw_textured_rgbd_kernel(
        mesh_ids: wp.array(dtype=wp.uint64),
        cam_poss: wp.array(dtype=wp.vec3, ndim=2),
        cam_quats: wp.array(dtype=wp.quat, ndim=2),
        K_inv: wp.mat44,
        far_plane: float,
        rgb_pixels: wp.array(dtype=wp.vec3, ndim=4),
        depth_pixels: wp.array(dtype=float, ndim=4),
        vertex_uvs: wp.array(dtype=wp.vec2),
        texture_image: wp.array(dtype=wp.vec3, ndim=2),
        texture_width: int,
        texture_height: int,
        base_color_factor: wp.vec3,
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
        dist = far_plane
        color = wp.vec3(0.0, 0.0, 0.0)

        if wp.mesh_query_ray(mesh, ro, rd, far_plane, t, u, v, sign, n, f):
            dist = t
            mesh_obj = wp.mesh_get(mesh)
            idx0 = mesh_obj.indices[f * 3 + 0]
            idx1 = mesh_obj.indices[f * 3 + 1]
            idx2 = mesh_obj.indices[f * 3 + 2]

            uv0 = vertex_uvs[idx0]
            uv1 = vertex_uvs[idx1]
            uv2 = vertex_uvs[idx2]
            w = 1.0 - u - v
            uv = w * uv0 + u * uv1 + v * uv2

            # Bilinear texture sampling with V-flip (GLTF V-axis convention).
            fu = wp.clamp(uv[0], 0.0, 1.0) * float(texture_width - 1)
            fv = wp.clamp(1.0 - uv[1], 0.0, 1.0) * float(texture_height - 1)
            x0 = int(wp.floor(fu))
            y0 = int(wp.floor(fv))
            x1 = wp.min(x0 + 1, texture_width - 1)
            y1 = wp.min(y0 + 1, texture_height - 1)
            tx = fu - float(x0)
            ty = fv - float(y0)
            c00 = texture_image[y0, x0]
            c10 = texture_image[y0, x1]
            c01 = texture_image[y1, x0]
            c11 = texture_image[y1, x1]
            albedo = (
                (1.0 - tx) * (1.0 - ty) * c00
                + tx * (1.0 - ty) * c10
                + (1.0 - tx) * ty * c01
                + tx * ty * c11
            )
            albedo = wp.cw_mul(albedo, base_color_factor)

            n_hat = wp.normalize(n)
            l_hat = wp.normalize(light_dir_world)
            lambert = wp.max(wp.dot(n_hat, l_hat), 0.0)
            lit = ambient_strength + (1.0 - ambient_strength) * lambert
            color = lit * albedo

        rgb_pixels[env_id, cam_id, y, x] = color
        depth_pixels[env_id, cam_id, y, x] = dist

    @staticmethod
    @wp.kernel
    def draw_shaded_rgbd_kernel(
        mesh_ids: wp.array(dtype=wp.uint64),
        cam_poss: wp.array(dtype=wp.vec3, ndim=2),
        cam_quats: wp.array(dtype=wp.quat, ndim=2),
        K_inv: wp.mat44,
        far_plane: float,
        rgb_pixels: wp.array(dtype=wp.vec3, ndim=4),
        depth_pixels: wp.array(dtype=float, ndim=4),
        vertex_colors: wp.array(dtype=wp.vec3),
        vertex_color_offsets: wp.array(dtype=wp.int32),
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
        dist = far_plane
        color = wp.vec3(0.0, 0.0, 0.0)

        if wp.mesh_query_ray(mesh, ro, rd, far_plane, t, u, v, sign, n, f):
            dist = t
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

            n_hat = wp.normalize(n)
            l_hat = wp.normalize(light_dir_world)
            lambert = wp.max(wp.dot(n_hat, l_hat), 0.0)
            lit = ambient_strength + (1.0 - ambient_strength) * lambert
            color = lit * albedo

        rgb_pixels[env_id, cam_id, y, x] = color
        depth_pixels[env_id, cam_id, y, x] = dist
