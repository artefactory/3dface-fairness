import os
import trimesh
import numpy as np


def extract_region_mesh(mesh_rec, face_region, model_name="basel"):
    """Extract a named facial region from a reconstructed full-face mesh.

    Returns a mesh built from region vertices and region-local faces,
    together with the vertex index mask used for extraction.
    """
    region_dir = os.path.join("data/3dmm_regions", model_name)
    mask_verts = np.load(os.path.join(region_dir, f"{face_region}_mask_verts.npy"))
    face_region_faces = np.load(os.path.join(region_dir, f"{face_region}_faces.npy"))
    region_mesh = trimesh.Trimesh(
        vertices=mesh_rec.vertices[mask_verts],
        faces=face_region_faces,
    )
    return region_mesh, mask_verts
