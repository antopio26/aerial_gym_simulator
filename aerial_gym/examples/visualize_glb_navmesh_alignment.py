import argparse
import os

import numpy as np
import trimesh as tm

from aerial_gym.utils.standalone_navmesh_parser import StandaloneNavMesh


def _parse_vec3(raw):
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError(f"Expected 3 comma-separated values, got: {raw}")
    return np.array([float(parts[0]), float(parts[1]), float(parts[2])], dtype=np.float32)


def _load_glb_meshes(glb_path):
    loaded = tm.load(glb_path, force="scene")
    if isinstance(loaded, tm.Trimesh):
        return [loaded]
    meshes = [m for m in loaded.dump(concatenate=False) if isinstance(m, tm.Trimesh)]
    if not meshes:
        raise ValueError(f"No mesh geometry found in GLB: {glb_path}")
    return meshes


def _build_navmesh_trimesh(navmesh_path):
    nav = StandaloneNavMesh(navmesh_path)
    verts = np.asarray(nav.vertices, dtype=np.float32)
    if verts.shape[0] == 0:
        raise ValueError(f"No vertices parsed from navmesh: {navmesh_path}")

    tri_faces = []
    for poly in nav.polygons:
        if len(poly) < 3:
            continue
        root = poly[0]
        for i in range(1, len(poly) - 1):
            tri_faces.append([root, poly[i], poly[i + 1]])

    if not tri_faces:
        raise ValueError(f"No polygon faces parsed from navmesh: {navmesh_path}")

    faces = np.asarray(tri_faces, dtype=np.int64)
    return tm.Trimesh(vertices=verts, faces=faces, process=False)


def _apply_transform(mesh, scale, translation):
    mesh.vertices = mesh.vertices * scale + translation[None, :]


def _collect_bounds(meshes):
    mins = []
    maxs = []
    for m in meshes:
        mins.append(m.bounds[0])
        maxs.append(m.bounds[1])
    return np.min(np.asarray(mins), axis=0), np.max(np.asarray(maxs), axis=0)


def _safe_show_overlay(overlay, explicit_export_path=""):
    try:
        overlay.show()
        return
    except Exception as exc:
        # In Docker/headless setups trimesh viewer may fail due to missing GLU/GLX.
        print(f"Viewer unavailable ({type(exc).__name__}): {exc}")
        if explicit_export_path:
            print("Overlay has already been exported via --export.")
            return

        fallback_export = "navmesh_overlay_fallback.glb"
        overlay.export(fallback_export)
        print(
            "Saved fallback overlay scene to: "
            f"{fallback_export} (use --no-show to suppress viewer attempts)"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Overlay a GLB scene and navmesh to visually inspect alignment."
    )
    parser.add_argument("--glb", required=True, help="Path to GLB file")
    parser.add_argument("--navmesh", required=True, help="Path to .navmesh file")
    parser.add_argument(
        "--scene-scale",
        type=float,
        default=1.0,
        help="Scale applied to both GLB and navmesh (default: 1.0)",
    )
    parser.add_argument(
        "--scene-translation",
        type=str,
        default="0,0,0",
        help="Translation applied to both GLB and navmesh, format x,y,z",
    )
    parser.add_argument(
        "--navmesh-scale",
        type=float,
        default=1.0,
        help="Extra scale applied to navmesh only (default: 1.0)",
    )
    parser.add_argument(
        "--navmesh-translation",
        type=str,
        default="0,0,0",
        help="Extra translation applied to navmesh only, format x,y,z",
    )
    parser.add_argument(
        "--export",
        type=str,
        default="",
        help="Optional path to export merged overlay scene (.glb/.ply)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open interactive viewer; only print alignment stats/export",
    )

    args = parser.parse_args()

    if not os.path.exists(args.glb):
        raise FileNotFoundError(f"GLB file not found: {args.glb}")
    if not os.path.exists(args.navmesh):
        raise FileNotFoundError(f"Navmesh file not found: {args.navmesh}")

    scene_translation = _parse_vec3(args.scene_translation)
    navmesh_translation = _parse_vec3(args.navmesh_translation)

    glb_meshes = _load_glb_meshes(args.glb)
    navmesh_mesh = _build_navmesh_trimesh(args.navmesh)

    for i, glb_mesh in enumerate(glb_meshes):
        _apply_transform(glb_mesh, args.scene_scale, scene_translation)
        glb_mesh.visual.face_colors = np.array([180, 180, 180, 90], dtype=np.uint8)
        glb_mesh.metadata["name"] = f"glb_{i}"

    _apply_transform(navmesh_mesh, args.scene_scale, scene_translation)
    _apply_transform(navmesh_mesh, args.navmesh_scale, navmesh_translation)
    navmesh_mesh.visual.face_colors = np.array([220, 40, 40, 170], dtype=np.uint8)

    glb_min, glb_max = _collect_bounds(glb_meshes)
    nav_min, nav_max = navmesh_mesh.bounds

    print("=== Alignment Stats ===")
    print(f"GLB bounds min: {glb_min}")
    print(f"GLB bounds max: {glb_max}")
    print(f"Nav bounds min: {nav_min}")
    print(f"Nav bounds max: {nav_max}")
    print(f"Bounds center delta: {((glb_min + glb_max) * 0.5) - ((nav_min + nav_max) * 0.5)}")

    overlay = tm.Scene()
    for i, glb_mesh in enumerate(glb_meshes):
        overlay.add_geometry(glb_mesh, geom_name=f"glb_{i}")
    overlay.add_geometry(navmesh_mesh, geom_name="navmesh")

    if args.export:
        overlay.export(args.export)
        print(f"Exported overlay scene to: {args.export}")

    if not args.no_show:
        _safe_show_overlay(overlay, explicit_export_path=args.export)


if __name__ == "__main__":
    main()
