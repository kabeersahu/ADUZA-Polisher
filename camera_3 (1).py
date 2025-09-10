#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mesh-only segmentation (Stages 35) + Stage 5b top-patch extraction.

Pipeline:
 3) RANSAC plane segmentation on mesh vertices ’ remove base plate.
 4) DBSCAN on remaining vertices (with eps, min_points) ’ cluster clamps/object.
    Optional: 2-of-3 smoothing on DBSCAN labels (neighbor vote).
 5) Select object cluster ’ extract as submesh ’ save as object_surface_mesh.ply.
 5b) Extract ONLY the top concave/convex patch (normals-up + height band +
     region grow + concavity-aware filtering) ’ save as object_top_patch_mesh.ply.

Notes:
- Only CLI behavior changed: defaults are provided and parse_args([]) is used, so you can run directly:
    python mesh_seg_ransac_dbscan_2of3_top.py
- All core logic (RANSAC, DBSCAN, 2-of-3 voting, selection, top-patch) remains intact.
"""

import os
import sys
import argparse
from collections import Counter

import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN


# -------------------- Core utilities --------------------

def segment_plane_on_mesh_vertices(mesh: o3d.geometry.TriangleMesh,
                                   dist_thresh: float,
                                   num_iterations: int = 1000):
    """RANSAC plane on mesh vertices; return (plane_model, inlier_idx, outlier_idx)."""
    verts = np.asarray(mesh.vertices)
    if verts.size == 0:
        raise RuntimeError("Mesh has no vertices.")

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(verts))
    plane_model, inliers = pcd.segment_plane(distance_threshold=dist_thresh,
                                             ransac_n=3,
                                             num_iterations=num_iterations)
    inliers = np.asarray(inliers, dtype=np.int64)
    all_idx = np.arange(len(verts), dtype=np.int64)
    outliers = np.setdiff1d(all_idx, inliers, assume_unique=False)
    return plane_model, inliers, outliers


def dbscan_on_subset_vertices(mesh: o3d.geometry.TriangleMesh,
                              subset_idx: np.ndarray,
                              eps: float,
                              min_points: int):
    """Run DBSCAN on mesh vertices limited to subset_idx; return labels per subset vertex."""
    verts = np.asarray(mesh.vertices)
    if verts.size == 0:
        return np.array([], dtype=int)
    pts = verts[subset_idx]
    if len(pts) == 0:
        return np.array([], dtype=int)
    labels = DBSCAN(eps=eps, min_samples=min_points).fit(pts).labels_
    return labels


def two_of_three_neighbor_vote(points: np.ndarray,
                               labels: np.ndarray,
                               k: int = 3) -> np.ndarray:
    """
    Safe 2-of-3 smoothing:
      - removes non-finite points before building KDTree
      - clamps k to [1, N]
      - skips search if < 2 valid neighbors
    """
    if points.size == 0 or labels.size == 0:
        return labels

    # Filter out NaN/Inf rows
    finite_mask = np.isfinite(points).all(axis=1)
    if not finite_mask.all():
        # operate only on the valid subset, then copy back
        valid_idx = np.where(finite_mask)[0]
        pts_valid = points[finite_mask]
        lbl_valid = labels[finite_mask].copy()
        if pts_valid.shape[0] == 0:
            return labels  # nothing we can smooth safely

        tree = o3d.geometry.KDTreeFlann(o3d.utility.Vector3dVector(pts_valid))
        safe_k = max(1, min(k, pts_valid.shape[0]))
        for i in range(pts_valid.shape[0]):
            if lbl_valid[i] == -1:
                continue
            try:
                _, idx, _ = tree.search_knn_vector_3d(pts_valid[i], safe_k)
            except Exception:
                continue
            neigh = lbl_valid[idx]
            cnt = Counter(neigh[neigh >= 0])
            if cnt:
                most, c = cnt.most_common(1)[0]
                if c >= 2:
                    lbl_valid[i] = most
        # write back only to finite positions
        out = labels.copy()
        out[finite_mask] = lbl_valid
        return out

    # All points are finite
    tree = o3d.geometry.KDTreeFlann(o3d.utility.Vector3dVector(points))
    new_labels = labels.copy()
    safe_k = max(1, min(k, points.shape[0]))
    for i in range(points.shape[0]):
        if labels[i] == -1:
            continue
        try:
            _, idx, _ = tree.search_knn_vector_3d(points[i], safe_k)
        except Exception:
            continue
        neigh = labels[idx]
        cnt = Counter(neigh[neigh >= 0])
        if cnt:
            most, c = cnt.most_common(1)[0]
            if c >= 2:
                new_labels[i] = most
    return new_labels


def select_cluster(subset_idx: np.ndarray,
                   labels: np.ndarray,
                   mesh: o3d.geometry.TriangleMesh,
                   plane_model=None,
                   mode: str = "largest") -> np.ndarray:
    """
    Choose which cluster is the object of interest.
    Modes:
      - "largest": cluster with most vertices.
      - "highest_above_plate": highest mean height along plane normal (needs plane_model).
      - "centered": closest cluster centroid to overall centroid of remaining verts.
    Returns absolute vertex indices of the chosen cluster.
    """
    valid_mask = labels >= 0
    if not np.any(valid_mask):
        return np.array([], dtype=np.int64)

    verts = np.asarray(mesh.vertices)
    subset_pts = verts[subset_idx]
    cluster_ids = np.unique(labels[valid_mask])

    if mode == "largest":
        best = max(cluster_ids, key=lambda cid: np.sum(labels == cid))
        return subset_idx[labels == best]

    if mode == "highest_above_plate" and plane_model is not None:
        a, b, c, _ = plane_model
        n = np.array([a, b, c], dtype=float)
        n /= (np.linalg.norm(n) + 1e-12)
        best = None
        best_val = -1e18
        for cid in cluster_ids:
            pts = subset_pts[labels == cid]
            mean_h = np.mean(pts @ n)
            if mean_h > best_val:
                best_val = mean_h
                best = cid
        return subset_idx[labels == best]

    if mode == "centered":
        scene_centroid = np.mean(subset_pts, axis=0)
        best = None
        best_dist = 1e18
        for cid in cluster_ids:
            pts = subset_pts[labels == cid]
            d = np.linalg.norm(np.mean(pts, axis=0) - scene_centroid)
            if d < best_dist:
                best_dist = d
                best = cid
        return subset_idx[labels == best]

    # fallback
    best = max(cluster_ids, key=lambda cid: np.sum(labels == cid))
    return subset_idx[labels == best]


def extract_submesh_by_vertex_indices(mesh: o3d.geometry.TriangleMesh,
                                      keep_idx_abs: np.ndarray) -> o3d.geometry.TriangleMesh:
    """Extract submesh with triangles whose all 3 vertices are in keep_idx_abs."""
    if keep_idx_abs.size == 0:
        return o3d.geometry.TriangleMesh()

    keep_mask = np.zeros(len(mesh.vertices), dtype=bool)
    keep_mask[keep_idx_abs] = True

    tris = np.asarray(mesh.triangles)
    if tris.size == 0:
        return o3d.geometry.TriangleMesh()

    face_keep = np.all(keep_mask[tris], axis=1)
    kept_tris = tris[face_keep]

    if kept_tris.size == 0:
        return o3d.geometry.TriangleMesh()  # empty

    used_old = np.unique(kept_tris.reshape(-1))
    remap = -np.ones(len(mesh.vertices), dtype=int)
    remap[used_old] = np.arange(len(used_old))

    new_tris = remap[kept_tris]
    out = o3d.geometry.TriangleMesh()
    out.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices)[used_old])
    out.triangles = o3d.utility.Vector3iVector(new_tris)

    # Colors preserved if present
    if mesh.has_vertex_colors():
        out.vertex_colors = o3d.utility.Vector3dVector(np.asarray(mesh.vertex_colors)[used_old])

    # Recompute normals for robustness
    out.compute_vertex_normals()
    return out


# -------------------- Stage 5b: top patch extraction --------------------

def compute_vertex_curvature(points: np.ndarray, k: int = 30) -> np.ndarray:
    """
    Safe PCA curvature estimator:
      curvature_i = lambda_min / (lambda1 + lambda2 + lambda3)
    - removes non-finite points
    - clamps k to [3, N]; if <3 neighbors, returns 0.0
    """
    n = points.shape[0]
    if n == 0:
        return np.array([], dtype=float)

    finite_mask = np.isfinite(points).all(axis=1)
    curv = np.zeros(n, dtype=float)
    if not finite_mask.all():
        # Build tree only on finite subset
        valid_idx = np.where(finite_mask)[0]
        pts_valid = points[finite_mask]
        if pts_valid.shape[0] < 3:
            return curv  # nothing to do; keep zeros
        tree = o3d.geometry.KDTreeFlann(o3d.utility.Vector3dVector(pts_valid))
        for i_global in range(n):
            if not finite_mask[i_global]:
                curv[i_global] = 0.0
                continue
            # index in the compacted array
            i = np.searchsorted(valid_idx, i_global)
            safe_k = max(3, min(k, pts_valid.shape[0]))
            try:
                _, idx, _ = tree.search_knn_vector_3d(pts_valid[i], safe_k)
            except Exception:
                curv[i_global] = 0.0
                continue
            nn = pts_valid[idx]
            if nn.shape[0] < 3:
                curv[i_global] = 0.0
                continue
            C = np.cov(nn.T)
            try:
                w, _ = np.linalg.eigh(C)
                w = np.sort(np.abs(w))
                denom = (np.sum(w) + 1e-12)
                curv[i_global] = (w[0] / denom) if denom > 0 else 0.0
            except np.linalg.LinAlgError:
                curv[i_global] = 0.0
        return curv

    # All finite
    if n < 3:
        return curv
    tree = o3d.geometry.KDTreeFlann(o3d.utility.Vector3dVector(points))
    safe_k = max(3, min(k, n))
    for i in range(n):
        try:
            _, idx, _ = tree.search_knn_vector_3d(points[i], safe_k)
        except Exception:
            curv[i] = 0.0
            continue
        nn = points[idx]
        if nn.shape[0] < 3:
            curv[i] = 0.0
            continue
        C = np.cov(nn.T)
        try:
            w, _ = np.linalg.eigh(C)
            w = np.sort(np.abs(w))
            denom = (np.sum(w) + 1e-12)
            curv[i] = (w[0] / denom) if denom > 0 else 0.0
        except np.linalg.LinAlgError:
            curv[i] = 0.0
    return curv


def extract_top_patch(mesh_obj: o3d.geometry.TriangleMesh,
                      plate_normal_world: np.ndarray,
                      angle_deg: float = 55.0,
                      height_quantile: float = 0.60,
                      concavity_percentile: float = 75.0,
                      knn_curv: int = 30) -> np.ndarray:
    """
    Keep only vertices forming the TOP polishing patch:
      1) normals roughly "up" (<= angle_deg from plate normal)
      2) vertices in the upper height band (>= height_quantile)
      3) concavity-aware: suppress sharp corners/sides using curvature percentile
      4) region-grow from highest seed with normal smoothness (<= 25°)
    Returns a boolean vertex mask over mesh_obj.vertices.
    """
    v = np.asarray(mesh_obj.vertices)
    if v.size == 0:
        return np.array([], dtype=bool)

    if not mesh_obj.has_vertex_normals():
        mesh_obj.compute_vertex_normals()
    n = np.asarray(mesh_obj.vertex_normals)

    up = plate_normal_world / (np.linalg.norm(plate_normal_world) + 1e-12)

    # Signed "height" along the up direction relative to centroid
    height = v @ up   # absolute height along camera's "up"

    # Angle between vertex normals and 'up'
    cosang = np.clip(n @ up, -1.0, 1.0)
    ang = np.degrees(np.arccos(cosang))
    ang_mask = (ang <= angle_deg)

    # Upper height band
    h_thr = np.quantile(height, height_quantile)
    h_mask = (height >= h_thr)

    # Concavity-aware filter: compute curvature and drop high-curvature spikes (corners)
    curv = compute_vertex_curvature(v, k=knn_curv)
    if curv.size == 0:
        concave_mask = np.ones_like(ang_mask, dtype=bool)
    else:
        curv_thr = np.percentile(curv, concavity_percentile)
        concave_mask = curv <= curv_thr  # keep smoother patch, reject spiky edges

    keep0 = ang_mask & h_mask & concave_mask
    if not np.any(keep0):
        return keep0

    # 1-ring adjacency
    tris = np.asarray(mesh_obj.triangles)
    adj = [[] for _ in range(len(v))]
    for a, b, c3 in tris:
        adj[a].extend([b, c3])
        adj[b].extend([a, c3])
        adj[c3].extend([a, b])
    for i in range(len(v)):
        if adj[i]:
            adj[i] = list(set(adj[i]))

    # Region-grow from highest vertex that passes keep0
    seed_candidates = np.where(keep0)[0]
    if seed_candidates.size == 0:
        return keep0  # nothing passes; return initial mask (possibly empty)
    seed_idx = seed_candidates[np.argmax(height[seed_candidates])]

    grown = np.zeros(len(v), dtype=bool)
    stack = [int(seed_idx)]
    grown[seed_idx] = True
    while stack:
        u = stack.pop()
        for w in adj[u]:
            if grown[w] or not keep0[w]:
                continue
            # normal smoothness d 25°
            dot = float(np.clip(np.dot(n[u], n[w]), -1.0, 1.0))
            if np.degrees(np.arccos(dot)) <= 25.0:
                grown[w] = True
                stack.append(w)

    final_mask = grown if grown.any() else keep0
    return final_mask


def submesh_from_vertex_mask(mesh_in: o3d.geometry.TriangleMesh,
                             mask: np.ndarray) -> o3d.geometry.TriangleMesh:
    """Build a submesh keeping only faces with all 3 vertices inside mask."""
    if mask.size == 0:
        return o3d.geometry.TriangleMesh()

    tris = np.asarray(mesh_in.triangles)
    verts = np.asarray(mesh_in.vertices)
    if tris.size == 0 or verts.size == 0:
        return o3d.geometry.TriangleMesh()

    keep_face = np.all(mask[tris], axis=1)
    kept_tris = tris[keep_face]
    if kept_tris.size == 0:
        return o3d.geometry.TriangleMesh()

    used = np.unique(kept_tris.reshape(-1))
    remap = -np.ones(len(verts), dtype=int)
    remap[used] = np.arange(len(used))
    new_tris = remap[kept_tris]
    out = o3d.geometry.TriangleMesh()
    out.vertices = o3d.utility.Vector3dVector(verts[used])
    out.triangles = o3d.utility.Vector3iVector(new_tris)
    if mesh_in.has_vertex_colors():
        out.vertex_colors = o3d.utility.Vector3dVector(np.asarray(mesh_in.vertex_colors)[used])
    out.compute_vertex_normals()
    return out


# -------------------- Main --------------------

def main():
    # We keep argparse (for flexibility) but force defaults if user runs without args.
    ap = argparse.ArgumentParser(description="Mesh-only RANSAC + DBSCAN + 2-of-3 + top-patch extractor")

    # Arguments with defaults so direct run works
    ap.add_argument("--mesh", type=str, default=r"F:\RealSenseData\Captures\capture_20250824_233841.ply", help="Input mesh (.ply/.obj/.stl)")
    ap.add_argument("--save_dir", type=str, default=r"F:\RealSenseData\yoyo", help="Output folder")
    ap.add_argument("--plane_dist", type=float, default=0.002, help="RANSAC distance threshold (meters)")
    ap.add_argument("--plane_iter", type=int, default=1500, help="RANSAC iterations")
    ap.add_argument("--dbscan_eps", type=float, default=0.01, help="DBSCAN eps (meters)")
    ap.add_argument("--dbscan_min", type=int, default=50, help="DBSCAN min points")
    ap.add_argument("--use_2of3", action="store_true", default=True,
                    help="Apply 2-of-3 neighbor vote to DBSCAN labels (default ON)")
    ap.add_argument("--select_mode", type=str, default="largest",
                    choices=["largest", "highest_above_plate", "centered"],
                    help="How to pick the object cluster")
    ap.add_argument("--top_angle", type=float, default=55.0, help="Max normal angle from 'up' for top patch (deg)")
    ap.add_argument("--top_height_q", type=float, default=0.65, help="Height quantile for top band [0..1]")
    ap.add_argument("--curv_percentile", type=float, default=75.0,
                    help="Concavity filter: keep curvature <= percentile")
    ap.add_argument("--show", action="store_true", default=True, help="Show Open3D viewers")

    # Force using defaults if user double-clicks / no CLI args provided
    args = ap.parse_args([])

    # Ensure output folder
    os.makedirs(args.save_dir, exist_ok=True)

    # ----- Load mesh BEFORE Stage 3 (fixes NameError: mesh not defined) -----
    mesh_path = args.mesh
    # Normalize Windows backslashes to forward slashes to avoid unicode-escape issues in prints
    mesh_path_print = mesh_path.replace("\\", "/")
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if len(mesh.vertices) == 0:
        raise RuntimeError(f"Empty mesh or unsupported file format: {mesh_path_print}")

    # --- 3) Plane segmentation on vertices ---
    print(f"[Stage 3] Plane RANSAC (dist={args.plane_dist:.4f}, iters={args.plane_iter})")
    plane_model, inliers_plate, outliers_rest = segment_plane_on_mesh_vertices(
        mesh, args.plane_dist, num_iterations=args.plane_iter
    )
    print(" Plane model a,b,c,d:", plane_model)
    print(f" Plate verts: {inliers_plate.size} | Remaining verts: {outliers_rest.size}")

    # Save quick colored visualization (plate=green, rest=red)
    colors = np.tile(np.array([[0.7, 0.1, 0.1]]), (len(mesh.vertices), 1))
    colors[inliers_plate] = np.array([0.0, 0.7, 0.2])
    mesh_col = o3d.geometry.TriangleMesh(mesh)
    mesh_col.vertex_colors = o3d.utility.Vector3dVector(colors)
    out_plate_vs_rest = os.path.join(args.save_dir, "mesh_plate_vs_rest.ply")
    o3d.io.write_triangle_mesh(out_plate_vs_rest, mesh_col)
    if args.show:
        try:
            o3d.visualization.draw_geometries(
                [mesh_col],
                 window_name="Green=Plate, Red=Rest"
                 
            ) 
        except Exception as e:
           print(" Viewer warning:", e)

    # --- 4) DBSCAN on remaining vertices ---
    print(f"[Stage 4] DBSCAN (eps={args.dbscan_eps:.4f}, min={args.dbscan_min}) on remaining vertices")
    raw_labels = dbscan_on_subset_vertices(mesh, outliers_rest,
                                           eps=args.dbscan_eps, min_points=args.dbscan_min)

    # Optional 2-of-3 smoothing on vertex labels
    labels = raw_labels
    if args.use_2of3 and len(raw_labels) > 0:
        verts_rest = np.asarray(mesh.vertices)[outliers_rest]
        labels = two_of_three_neighbor_vote(verts_rest, raw_labels, k=3)
        print(" Applied 2-of-3 neighbor vote smoothing.")

    n_clusters = int(labels.max() + 1) if labels.size and labels.max() >= 0 else 0
    n_noise = int(np.sum(labels == -1)) if labels.size else 0
    print(f" Clusters found: {n_clusters}; Noise verts: {n_noise}")

    # Colorize clusters for QA
    vert_colors = np.zeros((len(mesh.vertices), 3)) + 0.6
    if n_clusters > 0:
        rng = np.random.default_rng(42)
        palette = rng.random((n_clusters, 3))
        for k in range(n_clusters):
            vert_colors[outliers_rest[labels == k]] = palette[k]
    mesh_clusters = o3d.geometry.TriangleMesh(mesh)
    mesh_clusters.vertex_colors = o3d.utility.Vector3dVector(vert_colors)
    out_clusters = os.path.join(args.save_dir, "mesh_clusters_vertices.ply")
    o3d.io.write_triangle_mesh(out_clusters, mesh_clusters)
    if args.show:
        try:
            o3d.visualization.draw_geometries([mesh_clusters], window_name="Vertex Clusters (after plate removal)")
        except Exception as e:
            print(" Viewer warning:", e)

    # --- 5) Select object cluster and extract submesh ---
    print(f"[Stage 5] Selecting object cluster (mode={args.select_mode})")
    keep_vertex_idx = select_cluster(outliers_rest, labels, mesh=mesh,
                                     plane_model=plane_model, mode=args.select_mode)
    print(f" Selected cluster vertex count: {keep_vertex_idx.shape[0]}")

    submesh = extract_submesh_by_vertex_indices(mesh, keep_vertex_idx)
    
    out_surface_path = os.path.join(args.save_dir, "object_surface_mesh.ply")
    o3d.io.write_triangle_mesh(out_surface_path, submesh)

    # --- 5b) Extract ONLY the TOP patch of the object (concave/convex) ---
    print("[Stage 5b] Extracting top patch (normals-up + height band + region-grow + concavity filter)")
    a, b, c, _ = plane_model
    plate_normal = np.array([a, b, c], dtype=float)
    plate_normal /= (np.linalg.norm(plate_normal) + 1e-12)

    top_mask = extract_top_patch(
        submesh,
        plate_normal_world=plate_normal,
        angle_deg=float(args.top_angle),
        height_quantile=float(args.top_height_q),
        concavity_percentile=float(args.curv_percentile),
        knn_curv=30
    )
    top_mesh = submesh_from_vertex_mask(submesh, top_mask)
    out_top_path = os.path.join(args.save_dir, "object_top_patch_mesh.ply")
    o3d.io.write_triangle_mesh(out_top_path, top_mesh)

    if args.show:
        try:
            # Show top patch overlaid on whole object
            submesh_vis = o3d.geometry.TriangleMesh(submesh)
            submesh_vis.paint_uniform_color([0.75, 0.75, 0.75])
            top_vis = o3d.geometry.TriangleMesh(top_mesh)
            top_vis.paint_uniform_color([0.1, 0.8, 0.1])
            o3d.visualization.draw_geometries([submesh_vis, top_vis],
                                              window_name="Top Patch (green) over Object")
        except Exception as e:
            print(" Viewer warning:", e)

    print("\nSaved:")
    print(" "", out_plate_vs_rest, "(quick QA color)")
    print(" "", out_clusters, "(cluster QA)")
    print(" "", out_surface_path, "(whole object)")
    print(" "", out_top_path, "(only the concave/convex top to polish)")
    print("\n--- Mesh Segmentation (35 + 5b) Complete ---")


if __name__ == "__main__":
    # Make numpy print more compact
    np.set_printoptions(precision=4, suppress=True)
    try:
        main()
    except FileNotFoundError as e:
        print("ERROR: Could not open input mesh file. Tip: set --mesh to a valid .ply/.obj/.stl path.")
        print("Details:", e)
        sys.exit(1)
    except Exception as e:
        print("Unhandled error:", e)
        sys.exit(1)
