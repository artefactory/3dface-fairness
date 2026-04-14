import os
import trimesh
import numpy as np

from utils.rigid_icp import fit_icp_RT


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def barycentric_coordinates_to_keypoints(mesh, bary_coordinates):
    triangles = mesh.faces[bary_coordinates[:, 0].astype('int')]

    v0 = mesh.vertices[triangles[:, 0]]
    v1 = mesh.vertices[triangles[:, 1]]
    v2 = mesh.vertices[triangles[:, 2]]

    keypoints = (bary_coordinates[:, 1:2] * v0 + 
                 bary_coordinates[:, 2:3] * v1 + 
                 bary_coordinates[:, 3:4] * v2)
    return keypoints


def align_shape(shape_verts, shape_keypoints, realy_keypoints, with_scale=True):
    trans, scale, R, t = fit_icp_RT(shape_keypoints.T, realy_keypoints.T, with_scale=with_scale)

    if with_scale:
        shape_aligned = np.matmul(shape_verts * scale, R.T) + t
    else:
        shape_aligned = np.matmul(shape_verts, R.T) + t
    
    return shape_aligned, scale, R, t


def extract_region_mesh(mesh_rec, face_region, template_name):
    """Extract a named facial region from a reconstructed full-face mesh.

    Returns a mesh built from region vertices and region-local faces,
    together with the vertex index mask used for extraction.
    """
    region_dir = os.path.join(PROJECT_ROOT, "data", "3dmm_regions", template_name)
    mask_verts = np.load(os.path.join(region_dir, f"{face_region}_mask_verts.npy"))
    face_region_faces = np.load(os.path.join(region_dir, f"{face_region}_faces.npy"))
    region_mesh = trimesh.Trimesh(
        vertices=mesh_rec.vertices[mask_verts],
        faces=face_region_faces,
    )
    return region_mesh, mask_verts
