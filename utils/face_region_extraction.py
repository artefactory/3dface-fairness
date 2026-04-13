import sys
sys.path.append('..')

import os
import trimesh
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


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
