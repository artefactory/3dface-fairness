import os
from tqdm import tqdm

import trimesh
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from utils.face_region_extraction import extract_region_mesh


N_SCANS = 100
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REALY_PATH = os.path.join(PROJECT_ROOT, "data", "REALY", "REALY_scan_region")


def k_ring_neighbors(mesh, vertex_index, k):
    """Return indices belonging to the k-ring neighborhood of a vertex."""
    visited = {vertex_index}
    frontier = {vertex_index}

    for _ in range(k):
        new_frontier = set()
        for v in frontier:
            for nbr in mesh.vertex_neighbors[v]:
                if nbr not in visited:
                    new_frontier.add(nbr)
        visited.update(new_frontier)
        frontier = new_frontier

    return np.array(list(visited), dtype=int)


def compute_tangent_frame(v, neighbors, normal):
    """Compute a local tangent-bitangent-normal frame at a vertex."""
    n = normal / np.linalg.norm(normal)
    projections = neighbors - normal * np.sum((neighbors - v) * normal, axis=1, keepdims=True) - v

    t = projections[np.argmax(np.linalg.norm(projections, axis=1))]
    t /= np.linalg.norm(t)
    b = np.cross(n, t)
    b /= np.linalg.norm(b)

    return t, b, n


def project_points_to_tangent_plane(p, neighbors, frame):
    """Project neighbor points to local UVN coordinates centered at p."""
    t, b, n = frame

    u_list, v_list, w_list = [], [], []

    for q in neighbors:
        d = q - p
        u_list.append(np.dot(d, t))
        v_list.append(np.dot(d, b))
        w_list.append(np.dot(d, n))

    return np.array(u_list), np.array(v_list), np.array(w_list)


def compute_lstsq_matrix_quadratic_surface(u, v):
    """Build least-squares operator for quadratic surface fitting."""
    A = np.column_stack([u ** 2, u * v, v ** 2, u, v, np.ones_like(u)])
    M = np.linalg.inv(A.T @ A) @ A.T
    return M


def _build_delta_matrix(mesh, n_neighbors):
    """Compute the discrete operator used to estimate mean curvature."""
    n_vertices = len(mesh.vertices)
    delta_matrix = np.zeros((n_vertices, n_vertices))

    for i in range(n_vertices):
        ring1_indices = k_ring_neighbors(mesh, vertex_index=i, k=1)
        ringk_indices = k_ring_neighbors(mesh, vertex_index=i, k=n_neighbors)

        frame = compute_tangent_frame(
            mesh.vertices[i],
            mesh.vertices[ring1_indices],
            mesh.vertex_normals[i],
        )
        points = mesh.vertices[ringk_indices]
        u, v, _ = project_points_to_tangent_plane(mesh.vertices[i], points, frame)
        M = compute_lstsq_matrix_quadratic_surface(u, v)
        coeffs_by_neighbors = M[0] + M[2]
        delta_matrix[i, ringk_indices] = coeffs_by_neighbors

    return delta_matrix


def _compute_mean_curvature(mesh, n_neighbors):
    """Compute per-vertex mean curvature from local quadratic fits."""
    delta_matrix = _build_delta_matrix(mesh, n_neighbors)
    curvature_vectors = delta_matrix @ mesh.vertices
    return (curvature_vectors * mesh.vertex_normals).sum(axis=1)


def _interpolate_gt_values_to_reconstruction(mesh_gt, values_gt, mesh_rec):
    """Interpolate GT scalar values to reconstruction vertices via closest points."""
    proximity = trimesh.proximity.ProximityQuery(mesh_gt)
    closest_points, _, face_ids = proximity.on_surface(mesh_rec.vertices)

    triangles = mesh_gt.vertices[mesh_gt.faces[face_ids]]
    bary = trimesh.triangles.points_to_barycentric(triangles, closest_points)
    vertex_scalars = values_gt[mesh_gt.faces[face_ids]]
    return np.sum(bary * vertex_scalars, axis=1)


def _save_curvature_plot(
    mesh_gt,
    mesh_rec,
    gt_curvature_map,
    rec_curvature_map,
    diff_curvature_map,
    face_region,
    subject_id,
    save_path,
):
    """Save side-by-side GT / reconstruction / difference curvature visualizations."""
    combined = np.concatenate([gt_curvature_map, rec_curvature_map])
    abs_limit = np.percentile(np.abs(combined), 98)
    vmin, vmax = -abs_limit, abs_limit

    diff_limit = np.percentile(np.abs(diff_curvature_map), 98)
    dvmin, dvmax = -diff_limit, diff_limit

    def get_mesh_assets(base_mesh, curvature, val_min, val_max, cmap_name="jet"):
        norm = plt.Normalize(vmin=val_min, vmax=val_max)
        cmap = plt.get_cmap(cmap_name)
        clipped_curv = np.clip(curvature, val_min, val_max)
        face_curvature = clipped_curv[base_mesh.faces].mean(axis=1)
        face_colors = cmap(norm(face_curvature))
        poly_verts = base_mesh.vertices[base_mesh.faces]
        return poly_verts, face_colors

    fig = plt.figure(figsize=(30, 10))
    configs = [
        (mesh_gt, gt_curvature_map, "Ground truth, $H^{GT}$", vmin, vmax, "jet"),
        (mesh_rec, rec_curvature_map, "Reconstruction, $H^{R}$", vmin, vmax, "jet"),
        (mesh_rec, diff_curvature_map, "Difference, $H^{R} - H^{GT}$", dvmin, dvmax, "seismic"),
    ]

    zoom_factor = 0.7

    for i, (mesh, data, title, c_min, c_max, cmap_name) in enumerate(configs):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        poly_verts, face_colors = get_mesh_assets(mesh, data, c_min, c_max, cmap_name)
        collection = Poly3DCollection(poly_verts, facecolors=face_colors, edgecolors="none")
        ax.add_collection3d(collection)

        v = mesh.vertices
        center = v.mean(axis=0)
        max_range = np.max(np.abs(v - center))

        ax.set_xlim(center[0] - max_range * zoom_factor, center[0] + max_range * zoom_factor)
        ax.set_ylim(center[1] - max_range * zoom_factor, center[1] + max_range * zoom_factor)
        ax.set_zlim(center[2] - max_range * zoom_factor, center[2] + max_range * zoom_factor)

        ax.view_init(elev=90, azim=-90)
        ax.set_box_aspect([1, 1, 1])
        ax.set_axis_off()
        ax.set_title(title, fontsize=22)

        sm = plt.cm.ScalarMappable(cmap=cmap_name, norm=plt.Normalize(vmin=c_min, vmax=c_max))
        plt.colorbar(sm, ax=ax, shrink=0.5, aspect=15)

    fig.suptitle("Mean Curvature Comparison", fontsize=26, y=0.88)

    img_path = os.path.join(save_path, f"plots/{face_region}")
    os.makedirs(img_path, exist_ok=True)
    plt.savefig(os.path.join(img_path, f"{subject_id}.jpg"), bbox_inches="tight", dpi=100)
    plt.close(fig)


def calculate_curvature_maps(
    face_region,
    template_name,
    reconstruction_path,
    save_path,
    n_neighbors=3,
    curvature_plot_save=False,
):
    """Compute GT, reconstruction, and difference mean-curvature maps for all scans."""
    mean_curvature_maps = []
    mean_rec_curvature_maps = []
    diff_mean_curvature_maps = []

    for subject_id in tqdm(range(1, N_SCANS + 1)):
        mesh_rec_path = os.path.join(reconstruction_path, str(subject_id), "final_face.obj")
        mesh_gt_path = os.path.join(REALY_PATH, str(subject_id), f"{face_region}.obj")

        mesh_rec = trimesh.load(mesh_rec_path, process=False)
        mesh_region, _ = extract_region_mesh(mesh_rec, face_region, template_name=template_name)
        mesh_gt = trimesh.load(mesh_gt_path, process=False)

        H = _compute_mean_curvature(mesh_gt, n_neighbors)
        H_rec = _compute_mean_curvature(mesh_region, n_neighbors)
        interpolated_values = _interpolate_gt_values_to_reconstruction(mesh_gt, H, mesh_region)
        diff = H_rec - interpolated_values

        if curvature_plot_save:
            _save_curvature_plot(mesh_gt, mesh_region, H, H_rec, diff, face_region, subject_id, save_path)

        mean_curvature_maps.append(interpolated_values)
        mean_rec_curvature_maps.append(H_rec)
        diff_mean_curvature_maps.append(diff)

    mean_curvature_maps = np.array(mean_curvature_maps)
    mean_rec_curvature_maps = np.array(mean_rec_curvature_maps)
    diff_mean_curvature_maps = np.array(diff_mean_curvature_maps)
    
    os.makedirs(save_path, exist_ok=True)
    np.save(os.path.join(save_path, f"{face_region}_curvature_maps_gt.npy"), mean_curvature_maps)
    np.save(os.path.join(save_path, f"{face_region}_curvature_maps_rec.npy"), mean_rec_curvature_maps)
    np.save(os.path.join(save_path, f"{face_region}_curvature_maps_diff.npy"), diff_mean_curvature_maps)

    return mean_curvature_maps, mean_rec_curvature_maps, diff_mean_curvature_maps
