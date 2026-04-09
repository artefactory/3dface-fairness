# face-recon-fairness

Utilities for analyzing fairness-related error patterns in 3D face reconstruction using curvature maps, spectral decomposition, and clustering.

## What This Repository Does

This project provides a compact pipeline to:

- extract specific face regions from reconstructed 3D meshes,
- compute and compare mean-curvature maps between reconstruction and ground truth,
- project curvature-error maps into a spectral basis,
- cluster reconstruction error patterns and inspect demographic correlations.

The current codebase focuses on analysis utilities and notebooks rather than a packaged CLI.

## Repository Structure

- `utils/basel_face_region_extraction.py`  
  Extracts a face-region mesh (e.g. forehead, nose, etc.) from a full reconstruction.

- `utils/curvature_calculation.py`  
  Computes region-level mean-curvature maps for ground truth and reconstruction and saves visual comparisons.

- `utils/spectral_clustering.py`  
  Performs Laplacian spectral decomposition, clustering, and correlation analysis with attributes.

- `utils/visualisation.py`  
  Plots clustered subject images in grid layouts.

- `notebooks/basel_error_clustering.ipynb`  
  Example notebook workflow.

## Data Assumptions

The utilities assume the following assets exist:

- region masks and templates in `data/3dmm_regions/<method_name>/`,
- reconstruction meshes in paths like:
  - `data/direct_beta_id_optimisation/<subject_id>/final_face.obj`,
- ground-truth region meshes in a REALY benchmark directory,
- demographic attributes CSV (`data/realy_attributes.csv` by default).

Some defaults include absolute local paths (from the original development machine).  
Override function arguments when running on your environment.

## Installation

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy scipy pandas trimesh matplotlib seaborn scikit-learn tqdm robust-laplacian Pillow
```

## Quick Usage

Example: compute curvature maps for a region:

```python
from utils.curvature_calculation import calculate_curvature_maps

H_gt, H_rec, H_diff = calculate_curvature_maps(
    face_region="nose",
    n_neighbors=3,
    reconstruction_data_path="data/direct_beta_id_optimisation",
    realy_data_path="/path/to/REALY_scan_region",
    method_name="basel",
)
```

Example: spectral decomposition + clustering:

```python
from utils.spectral_clustering import (
    calculate_spectral_decomposition,
    cluster_curvature_maps,
)

spectral_coefs, evecs = calculate_spectral_decomposition(
    diff_error_maps=H_diff,
    face_region="nose",
    n_eig=512,
    method_name="basel",
)

labels = cluster_curvature_maps(
    n_clusters=4,
    diff_curvature_maps=H_diff,
    spectral_coefs=spectral_coefs,
    evecs=evecs,
    face_region="nose",
)
```

## Outputs

Generated artifacts are written under:

- `data/results/<method_name>/curvature_maps/<face_region>/`
  - `*_curvature_maps_gt.npy`
  - `*_curvature_maps_rec.npy`
  - `*_curvature_maps_diff.npy`
  - per-subject curvature comparison plots
- `data/results/<method_name>/`
  - cluster label arrays.

## Notes

- This repository is currently research-oriented and assumes a specific dataset layout.
- If you plan to reuse it across environments, consider centralizing paths into a config file.
