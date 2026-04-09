import os
import trimesh
import numpy as np


def extract_region_mesh(mesh_rec, face_region, model_name='basel'):
    mask_verts = np.load(os.path.join('data/3dmm_regions', f'{model_name}/{face_region}_mask_verts.npy'))
    face_region_faces = np.load(os.path.join('data/3dmm_regions', f'{model_name}/{face_region}_faces.npy'))
    region_mesh = trimesh.Trimesh(vertices=mesh_rec.vertices[mask_verts], faces=face_region_faces)
    return region_mesh, mask_verts
