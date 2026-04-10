This repository contains the code and experiments accompanying our paper:

***"Discovering Geometric Biases in 3D Face Reconstruction: A Curvature-Aware Spectral Framework for Fairness Evaluation"***

## Overview

This project provides a compact pipeline to:

- extract specific face regions from reconstructed 3D meshes,
- compute and compare mean-curvature maps between reconstruction and ground truth,
- project curvature-error maps into a spectral basis,
- cluster reconstruction error patterns and inspect demographic correlations.

## Installation

1. **Clone the repository**

```bash
git clone git@github.com:artefactory/face-recon-fairness.git
cd face-recon-fairness
```

2. **Create and activate your virtual environment**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

## Data Preparation

1. In order to use our evaluation pipeline, you need an access to REALY benchmark. Please sign the [Agreement](https://www-users.york.ac.uk/~np7/research/Headspace/) and indicate the usage of REALY benchmark according to their guideline, then you will get the permission to download the data.

2. Unzip the benchmark file and put "REALY_scan_region/", "REALY_image/" and "REALY_HIFI3D_keypoints/" folders into "data/REALY".

3. Use the images in the "REALY_image/" folder to reconstruct 3D meshes with your method(s).

4. Save reconstructed meshes as "\*.obj", where "\*" should have the same name as input images from REALY benchmark.

## Quick Usage

Example: compute curvature maps for a given face region:

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
