import os
from tqdm import tqdm

import trimesh
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from utils.basel_face_region_extraction import extract_region_mesh


def k_ring_neighbors(mesh, vertex_index, k):
    """
    Returns the set of vertex indices that form the k-ring neighborhood of the given vertex.
    
    Parameters:
        mesh: trimesh.Trimesh object
        vertex_index: int, index of the central vertex
        k: int, the "ring" number

    Returns:
        neighbors: set of vertex indices (including vertex_index optionally)
    """
    visited = set([vertex_index])
    frontier = set([vertex_index])
    
    for _ in range(k):
        new_frontier = set()
        for v in frontier:
            for nbr in mesh.vertex_neighbors[v]:
                if nbr not in visited:
                    new_frontier.add(nbr)
        visited.update(new_frontier)
        frontier = new_frontier
    
    # visited.remove(vertex_index)  # optional: remove the central vertex
    return np.array(list(visited), dtype='int')


def compute_tangent_frame(v, neighbors, normal):
    """
    Given a unit normal vector n and neighbors of a vertex v, compute tangent and bitangent vectors.
    """
    n = normal / np.linalg.norm(normal)

    projections = neighbors - normal*np.sum((neighbors - v) * normal, axis=1, keepdims=True) - v

    t = projections[np.argmax(np.linalg.norm(projections, axis=1))]
    t /= np.linalg.norm(t)

    # bitangent
    b = np.cross(n, t)
    b /= np.linalg.norm(b)

    return t, b, n


def project_points_to_tangent_plane(p, neighbors, frame):
    """
    Returns arrays u, v, w containing local coordinates of neighbor points.
    """
    t, b, n = frame[0], frame[1], frame[2]

    u_list, v_list, w_list = [], [], []

    for q in neighbors:
        d = q - p
        u_list.append(np.dot(d, t))
        v_list.append(np.dot(d, b))
        w_list.append(np.dot(d, n))

    return np.array(u_list), np.array(v_list), np.array(w_list)


def compute_lstsq_matrix_quadratic_surface(u, v):
    """
    Fit w = a u^2 + b u v + c v^2 + d u + e v + f
    using least squares.
    Returns coefficients [a, b, c, d, e, f].
    """
    A = np.column_stack([u**2, u*v, v**2, u, v, np.ones_like(u)])
    M = np.linalg.inv(A.T @ A) @ A.T
    return M


def save_curvature_plot(mesh_gt, mesh_rec, gt_curvature_map, rec_curvature_map, diff_curvature_map, face_region, img_name, method_name='basel'):
    
    # --- ROBUST SYMMETRIC CLIPPING ---
    # Combine all maps to find a global scale for the first two plots
    combined = np.concatenate([gt_curvature_map, rec_curvature_map])
    
    # Use 98th percentile of absolute values to define a symmetric limit
    # This ensures that if the range is -5 to +2, we clip to -5 to +5
    abs_limit = np.percentile(np.abs(combined), 98)
    vmin, vmax = -abs_limit, abs_limit

    # Separate symmetric limit for the Difference map (usually smaller range)
    diff_limit = np.percentile(np.abs(diff_curvature_map), 98)
    dvmin, dvmax = -diff_limit, diff_limit
    # ---------------------------------

    def get_mesh_assets(base_mesh, curvature, val_min, val_max, cmap_name='jet'):
        norm = plt.Normalize(vmin=val_min, vmax=val_max)
        cmap = plt.get_cmap(cmap_name)
        
        # Clip values to the robust range
        clipped_curv = np.clip(curvature, val_min, val_max)
        
        # Calculate face colors (average of vertex curvatures)
        face_curvature = clipped_curv[base_mesh.faces].mean(axis=1)
        face_colors = cmap(norm(face_curvature))
        poly_verts = base_mesh.vertices[base_mesh.faces]
        return poly_verts, face_colors

    fig = plt.figure(figsize=(30, 10))
    
    # Configuration: (mesh, data, title, vmin, vmax, colormap)
    configs = [
        (mesh_gt, gt_curvature_map, 'Ground truth, $H^{GT}$', vmin, vmax, 'jet'),
        (mesh_rec, rec_curvature_map, 'Reconstruction, $H^{R}$', vmin, vmax, 'jet'),
        (mesh_rec, diff_curvature_map, 'Difference, $H^{R} - H^{GT}$', dvmin, dvmax, 'seismic')
    ]

    zoom_factor = 0.7 

    for i, (mesh, data, title, c_min, c_max, cmap_name) in enumerate(configs):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        
        poly_verts, face_colors = get_mesh_assets(mesh, data, c_min, c_max, cmap_name)
        
        collection = Poly3DCollection(poly_verts, facecolors=face_colors, edgecolors='none')
        ax.add_collection3d(collection)

        # Set bounds and view
        v = mesh.vertices
        center = v.mean(axis=0)
        max_range = np.max(np.abs(v - center))
        
        ax.set_xlim(center[0] - max_range * zoom_factor, center[0] + max_range * zoom_factor)
        ax.set_ylim(center[1] - max_range * zoom_factor, center[1] + max_range * zoom_factor)
        ax.set_zlim(center[2] - max_range * zoom_factor, center[2] + max_range * zoom_factor)
        
        ax.view_init(elev=90, azim=-90) # Front view for Z-aligned meshes
        ax.set_box_aspect([1, 1, 1])
        ax.set_axis_off()
        ax.set_title(title, fontsize=22)

        # Colorbar specific to the subplot's range and colormap
        sm = plt.cm.ScalarMappable(cmap=cmap_name, norm=plt.Normalize(vmin=c_min, vmax=c_max))
        plt.colorbar(sm, ax=ax, shrink=0.5, aspect=15)

    fig.suptitle('Symmetric Mean Curvature Comparison', fontsize=26, y=0.88)

    img_path = f'data/results/{method_name}/curvature_maps/{face_region}/plots'
    os.makedirs(img_path, exist_ok=True)
    plt.savefig(os.path.join(img_path, f'{img_name}.jpg'), bbox_inches='tight', dpi=100)
    plt.close(fig)


def calculate_curvature_maps(face_region, n_neighbors=3, reconstruction_data_path='data/direct_beta_id_optimisation', 
                             realy_data_path='/Users/veronika.shilova/Desktop/REALY/REALY_benchmark/REALY_scan_region', method_name='basel'):
    n_scans = 100
    mean_curvature_maps = []
    mean_rec_curvature_maps = []
    diff_mean_curvature_maps = []

    for subject_id in tqdm(range(1, n_scans+1)):
        mesh_rec_path = os.path.join(reconstruction_data_path, str(subject_id), 'final_face.obj') # 3DMM reconstruction
        mesh_gt_path = os.path.join(realy_data_path, str(subject_id), f'{face_region}.obj') # ground truth

        mesh_rec = trimesh.load(mesh_rec_path, process=False)
        mesh_region, _ = extract_region_mesh(mesh_rec, face_region, model_name=method_name)
        mesh_gt = trimesh.load(mesh_gt_path, process=False)

        delta_matrix = np.zeros((len(mesh_gt.vertices), len(mesh_gt.vertices)))
        for i in range(len(mesh_gt.vertices)):
            indices = k_ring_neighbors(mesh_gt, vertex_index=i, k=n_neighbors)
            frame = compute_tangent_frame(mesh_gt.vertices[i], mesh_gt.vertices[k_ring_neighbors(mesh_gt, vertex_index=i, k=1)], mesh_gt.vertex_normals[i])
            points = mesh_gt.vertices[k_ring_neighbors(mesh_gt, vertex_index=i, k=n_neighbors)]
            u, v, _ = project_points_to_tangent_plane(mesh_gt.vertices[i], points, frame)
            M = compute_lstsq_matrix_quadratic_surface(u, v)
            coeffs_by_neighbors = M[0] + M[2]
            delta_matrix[i, indices] = coeffs_by_neighbors

        curvatures_vectors = delta_matrix @ mesh_gt.vertices
        H = (curvatures_vectors * mesh_gt.vertex_normals).sum(axis=1)

        delta_matrix_rec = np.zeros((len(mesh_region.vertices), len(mesh_region.vertices)))
        for i in range(len(mesh_region.vertices)):
            indices = k_ring_neighbors(mesh_region, vertex_index=i, k=n_neighbors)
            frame = compute_tangent_frame(mesh_region.vertices[i], mesh_region.vertices[k_ring_neighbors(mesh_region, vertex_index=i, k=1)], mesh_region.vertex_normals[i])
            points = mesh_region.vertices[k_ring_neighbors(mesh_region, vertex_index=i, k=n_neighbors)]
            u, v, _ = project_points_to_tangent_plane(mesh_region.vertices[i], points, frame)
            M = compute_lstsq_matrix_quadratic_surface(u, v)
            coeffs_by_neighbors = M[0] + M[2]
            delta_matrix_rec[i, indices] = coeffs_by_neighbors

        curvatures_vectors_rec = delta_matrix_rec @ mesh_region.vertices
        H_rec = (curvatures_vectors_rec * mesh_region.vertex_normals).sum(axis=1)

        # For evety point on reconstructed mesh find the nearest point (!not necessarily a vertex!) on the ground truth mesh
        pq = trimesh.proximity.ProximityQuery(mesh_gt)

        closest_points, _, face_ids = pq.on_surface(mesh_region.vertices)

        # Gather triangles for each closest point
        triangles = mesh_gt.vertices[mesh_gt.faces[face_ids]]  # (N, 3, 3)

        # Compute barycentric coordinates (vectorized)
        bary = trimesh.triangles.points_to_barycentric(triangles, closest_points)  # (N, 3)

        # Gather per-vertex scalar values
        vertex_scalars = H[mesh_gt.faces[face_ids]]  # (N, 3)

        # Linear interpolation
        interpolated_values = np.sum(bary * vertex_scalars, axis=1)

        # Difference between ground truth and reconstruction mean curvatures
        diff = H_rec - interpolated_values
        
        save_curvature_plot(mesh_gt, mesh_region, H, H_rec, diff, face_region, subject_id, method_name)

        mean_curvature_maps.append(interpolated_values)
        mean_rec_curvature_maps.append(H_rec)
        diff_mean_curvature_maps.append(diff)

    mean_curvature_maps = np.array(mean_curvature_maps)
    mean_rec_curvature_maps = np.array(mean_rec_curvature_maps)
    diff_mean_curvature_maps = np.array(diff_mean_curvature_maps)

    path = f'data/results/{method_name}/curvature_maps/{face_region}/'
    os.makedirs(path, exist_ok=True)
    np.save(os.path.join(path, f'{face_region}_curvature_maps_gt.npy'), mean_curvature_maps)
    np.save(os.path.join(path, f'{face_region}_curvature_maps_rec.npy'), mean_rec_curvature_maps)
    np.save(os.path.join(path, f'{face_region}_curvature_maps_diff.npy'), diff_mean_curvature_maps)

    return mean_curvature_maps, mean_rec_curvature_maps, diff_mean_curvature_maps
