import os

import numpy as np
import pandas as pd
import trimesh
from collections import Counter

import robust_laplacian
import scipy.sparse.linalg as sla

from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import seaborn as sns

from utils.basel_face_region_extraction import extract_region_mesh
from utils.visualisation import plot_clusters_from_path


N_SCANS = 100


def _template_region_path(method_name, face_region):
    return f"data/3dmm_regions/{method_name}/template_{face_region}.obj"


def calculate_spectral_decomposition(diff_error_maps, face_region, n_eig=512, method_name='basel'):
    """Project curvature-error maps into Laplace-Beltrami spectral space."""
    template_region = trimesh.load(_template_region_path(method_name, face_region), process=False)
    vertices = np.array(template_region.vertices)
    faces = np.array(template_region.faces)

    L, M = robust_laplacian.mesh_laplacian(vertices, faces)

    # Compute eigenvectors
    _, evecs = sla.eigsh(L, n_eig+1, M, sigma=1e-8)
    evecs = evecs[:, 1:]

    evecs = evecs / (evecs.T @ M @ evecs).diagonal()
    spectral_coefs = (diff_error_maps * M.diagonal()) @ evecs

    return spectral_coefs, evecs


def mean_curvature_integral(H, M):
    """Compute area-weighted absolute mean curvature integral."""
    area = M.diagonal()
    H_mean = np.sum(np.abs(H) * area) / np.sum(area)
    return H_mean


def calculate_mean_curvature_integral(diff_curvature_maps, evecs, spectral_coefs, face_region, 
                                      method_name='basel'):
    """Compute per-scan curvature integral in spatial and spectral spaces."""
    reconstruction_data_path = f'data/{method_name}/reconstructions'
    fname = f'data/results/{method_name}/curvature_maps/{face_region}/{face_region}_h_mean.npy'
    fname_sp = f'data/results/{method_name}/curvature_maps/{face_region}/{face_region}_h_mean_sp.npy'

    if os.path.isfile(fname) and os.path.isfile(fname_sp):
        h_mean = np.load(f'data/results/{method_name}/curvature_maps/{face_region}/{face_region}_h_mean.npy')
        h_mean_sp = np.load(f'data/results/{method_name}/curvature_maps/{face_region}/{face_region}_h_mean_sp.npy')
        return h_mean, h_mean_sp

    h_mean = []
    h_mean_sp = []

    for subject_id in range(1, N_SCANS + 1):
        mesh_rec_path = os.path.join(reconstruction_data_path, str(subject_id), 'final_face.obj') # 3DMM reconstruction
        mesh_rec = trimesh.load(mesh_rec_path, process=False)
        region_mesh, _ = extract_region_mesh(mesh_rec, face_region, model_name=method_name)

        _, M = robust_laplacian.mesh_laplacian(np.array(region_mesh.vertices), np.array(region_mesh.faces))

        h_mean_tmp = mean_curvature_integral(diff_curvature_maps[subject_id-1], M)
        h_mean_tmp_sp = mean_curvature_integral(evecs @ spectral_coefs[subject_id-1], M)
        h_mean.append(h_mean_tmp) 
        h_mean_sp.append(h_mean_tmp_sp) 

    h_mean = np.array(h_mean)
    h_mean_sp = np.array(h_mean_sp)

    np.save(f'data/results/{method_name}/curvature_maps/{face_region}/{face_region}_h_mean.npy', h_mean)
    np.save(f'data/results/{method_name}/curvature_maps/{face_region}/{face_region}_h_mean_sp.npy', h_mean_sp)

    return h_mean, h_mean_sp


def get_age_error_correlation_statistics(diff_curvature_maps, evecs, spectral_coefs, face_region, 
                                         method_name='basel', df_attributes_path='data/realy_attributes.csv'):
    """Print/plot age correlation statistics for curvature-based reconstruction error."""
    reconstruction_data_path = f'data/{method_name}/reconstructions'
    h_mean, h_mean_sp = calculate_mean_curvature_integral(diff_curvature_maps, evecs, spectral_coefs, face_region, reconstruction_data_path, method_name)

    df = pd.read_csv(df_attributes_path)
    mask_male = (df.gender == 'male')

    rho_p_all, p_p_all = pearsonr(df.age_lyhm.values, h_mean)
    rho_p_all_sp, p_p_all_sp = pearsonr(df.age_lyhm.values, h_mean_sp)

    rho_p_male, p_p_male = pearsonr(df[(df.gender == 'male')].age_lyhm.values, h_mean[mask_male])
    rho_p_male_sp, p_p_male_sp = pearsonr(df[(df.gender == 'male')].age_lyhm.values, h_mean_sp[mask_male])

    rho_p_female, p_p_female = pearsonr(df[(df.gender != 'male')].age_lyhm.values, h_mean[~mask_male])
    rho_p_female_sp, p_p_female_sp = pearsonr(df[(df.gender != 'male')].age_lyhm.values, h_mean_sp[~mask_male])

    spe_s_all, p_s_all = spearmanr(df.age_lyhm.values, h_mean)
    spe_s_all_sp, p_s_all_sp = spearmanr(df.age_lyhm.values, h_mean_sp)

    spe_s_male, p_s_male = spearmanr(df[(df.gender == 'male')].age_lyhm.values, h_mean[mask_male])
    spe_s_male_sp, p_s_male_sp = spearmanr(df[(df.gender == 'male')].age_lyhm.values, h_mean_sp[mask_male])

    spe_s_female, p_s_female = spearmanr(df[(df.gender != 'male')].age_lyhm.values, h_mean[~mask_male])
    spe_s_female_sp, p_s_female_sp = spearmanr(df[(df.gender != 'male')].age_lyhm.values, h_mean_sp[~mask_male])

    df_stats = pd.DataFrame({'Correlation coefficient / Population' : ['All', 'Male', 'Female'],
                             'Pearson' : [rho_p_all, rho_p_male, rho_p_female],
                             'Spearman' : [spe_s_all, spe_s_male, spe_s_female]}).set_index("Correlation coefficient / Population")

    df_stats_sp = pd.DataFrame({'Correlation coefficient (spectral curvature space) / Population' : ['All', 'Male', 'Female'],
                                'Pearson' : [rho_p_all_sp, rho_p_male_sp, rho_p_female_sp],
                                'Spearman' : [spe_s_all_sp, spe_s_male_sp, spe_s_female_sp]}).set_index("Correlation coefficient (spectral curvature space) / Population")

    print(f"Curvature-based error vs Age correlation @{face_region}")
    print(df_stats)
    print(df_stats_sp)

    x = df.age_lyhm.values[..., np.newaxis]
    y = h_mean
    reg = LinearRegression().fit(x, y)
    plt.plot(x, reg.predict(x), color='r', label=fr'$\hat{{y}}=\alpha_1 (age) + \alpha_0$')
    plt.scatter(df[(df.gender == 'female')].age_lyhm.values, h_mean[~mask_male], label='female')
    plt.scatter(df[(df.gender == 'male')].age_lyhm.values, h_mean[mask_male], label='male')
    plt.xlabel('Age')
    plt.ylabel(f'Error @{face_region}, difference between mean curvatures')
    plt.legend()
    plt.show()

    x = df.age_lyhm.values[mask_male][..., np.newaxis]
    y = h_mean[mask_male]
    reg = LinearRegression().fit(x, y)
    plt.plot(x, reg.predict(x), color='r', label=fr'$\hat{{y}}=\alpha_1 (age) + \alpha_0$')
    plt.scatter(df[(df.gender == 'male')].age_lyhm.values, h_mean[mask_male], label='male')
    plt.xlabel('Age')
    plt.ylabel(f'Error @{face_region}, difference between mean curvatures')
    plt.legend()
    plt.show()

    x = df.age_lyhm.values[~mask_male][..., np.newaxis]
    y = h_mean[~mask_male]
    reg = LinearRegression().fit(x, y)
    plt.plot(x, reg.predict(x), color='r', label=fr'$\hat{{y}}=\alpha_1 (age) + \alpha_0$')
    plt.scatter(df[(df.gender == 'female')].age_lyhm.values, h_mean[~mask_male], label='female')
    plt.xlabel('Age')
    plt.ylabel(f'Error @{face_region}, difference between mean curvatures')
    plt.legend()
    plt.show()


def calculate_optimal_n_clusters(spectral_coefs):
    """Plot elbow and silhouette curves for k-means model selection."""
    wcss = []
    k_range = range(1, 15)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        kmeans.fit(spectral_coefs)
        wcss.append(kmeans.inertia_)

    plt.plot(k_range, wcss, 'bx-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('WCSS')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()

    sil_scores = []

    for k in range(2, 15):
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = kmeans.fit_predict(spectral_coefs)
        score = silhouette_score(spectral_coefs, labels)
        sil_scores.append(score)

    plt.plot(range(2, 15), sil_scores, 'rx-')
    plt.xlabel('k')
    plt.ylabel('Silhouette Score')
    plt.show()


def cluster_curvature_maps(n_clusters, diff_curvature_maps, spectral_coefs, evecs, face_region, 
                           df_attributes_path='data/realy_attributes.csv', 
                           method_name='basel', n_eig_vis=50,
                           realy_images_path='data/REALY/REALY_benchmark/REALY_image/crop_image_frontal_512x512'):
    """Cluster spectral coefficients and visualize cluster-level mean maps."""
    reconstruction_data_path = f'data/{method_name}/reconstructions'
    df = pd.read_csv(df_attributes_path)
    template_region = trimesh.load(_template_region_path(method_name, face_region), process=False)

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(spectral_coefs)
    clusters_kmeans = kmeans.predict(spectral_coefs)

    print("Cluster assignments:")
    for i in range(n_clusters):
        idx = np.where(clusters_kmeans == i)[0]
        print(f"Cluster {i}:", idx + 1, Counter(df.ethnicity[idx]))

    h_mean, _ = calculate_mean_curvature_integral(diff_curvature_maps, evecs, spectral_coefs, face_region, reconstruction_data_path, method_name)
    cluster_means = {}
    for i in range(n_clusters):
        idx = np.where(clusters_kmeans == i)[0]
        mean_coeff = spectral_coefs[idx, :n_eig_vis].mean(axis=0)   # in spectral space
        mean_map = evecs[:, :n_eig_vis] @ mean_coeff
        cluster_means[i] = mean_map
        print(f"Cluster {i}: H_mean = {np.mean(h_mean[idx])}")

    for i, mean_map in cluster_means.items():
        # 1. Prepare Data
        V = np.array(template_region.vertices)  # Nx3
        F = np.array(template_region.faces)     # Mx3
        
        # Calculate face colors by averaging vertex values for each triangle
        # This is required for Poly3DCollection
        face_values = mean_map[F].mean(axis=1)
        
        # Robust symmetric scaling (optional but recommended for curvature/means)
        limit = np.percentile(np.abs(mean_map), 98)
        vmin, vmax = -limit, limit
        
        # 2. Setup 3D Plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 3. Create the 3D Collection
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap('jet')
        face_colors = cmap(norm(face_values))
        
        # Create triangle geometries: (M, 3, 3)
        poly_verts = V[F]
        
        collection = Poly3DCollection(poly_verts, facecolors=face_colors, edgecolors='none')
        ax.add_collection3d(collection)
        
        # 4. Camera and Framing
        center = V.mean(axis=0)
        max_range = np.max(np.abs(V - center))
        zoom_factor = 0.8  # Adjust to get closer/further
        
        ax.set_xlim(center[0] - max_range * zoom_factor, center[0] + max_range * zoom_factor)
        ax.set_ylim(center[1] - max_range * zoom_factor, center[1] + max_range * zoom_factor)
        ax.set_zlim(center[2] - max_range * zoom_factor, center[2] + max_range * zoom_factor)
        
        # Set Front View (Looking down Z-axis)
        ax.view_init(elev=90, azim=-90)
        ax.set_box_aspect([1, 1, 1])
        ax.axis('off')
        
        # 5. Title and Colorbar
        ax.set_title(f"Cluster {i} Mean Map", fontsize=15)
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        fig.colorbar(mappable, ax=ax, shrink=0.6, aspect=20)
        
        plt.show() # Or plt.savefig(...)

    plot_clusters_from_path(
        realy_images_path,
        clusters_kmeans,
        max_cols=4,
        base_figsize_per_image=2,
        extensions=(".png", ".jpg", ".jpeg", ".bmp")
    )

    res_path = f'data/results/{method_name}'
    os.makedirs(res_path, exist_ok=True)
    np.save(os.path.join(res_path, f'{face_region}_error_cluster_labels_kmeans_k={n_clusters}.npy'), clusters_kmeans)
    return clusters_kmeans


def get_ethnicity_error_correlation_statistics(cluster_lables, face_region, df_attributes_path='data/realy_attributes.csv'):
    """Visualize ethnicity distributions across discovered error clusters."""
    df = pd.read_csv(df_attributes_path)
    rows = []

    n_clusters = len(np.unique(cluster_lables))
    for i in range(n_clusters):
        idx = np.where(cluster_lables == i)[0]
        counts = Counter(df.ethnicity[idx])
        counts["cluster"] = i
        rows.append(counts)

    df_heatmap = pd.DataFrame(rows).fillna(0).set_index("cluster").astype(int)
    df_heatmap_sorted = df_heatmap.sort_values(by=['Asian', 'Indian', 'African'], ascending=False)
    plt.figure(figsize=(10, 6))
    sns.heatmap(df_heatmap_sorted, annot=True, cmap="coolwarm", fmt="g")
    plt.title("Raw Counts: Ethnicities per Error Cluster")
    plt.ylabel(f"Error Cluster @{face_region}")
    plt.show()

    df_col_norm = df_heatmap.div(df_heatmap.sum(axis=0), axis=1)
    df_col_sorted = df_col_norm.sort_values(by=['Asian', 'Indian', 'African'], ascending=False)
    plt.figure(figsize=(10, 6))
    sns.heatmap(df_col_sorted, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Column Normalized: Distribution of each Ethnicity across Clusters")
    plt.ylabel(f"Error Cluster @{face_region}")
    plt.show()

    df_row_norm = df_heatmap.div(df_heatmap.sum(axis=1), axis=0)
    df_row_sorted = df_row_norm.sort_values(by=['Asian', 'Indian', 'African'], ascending=False)
    plt.figure(figsize=(10, 6))
    sns.heatmap(df_row_sorted, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Row Normalized: Ethnic Composition of each Cluster")
    plt.ylabel(f"Error Cluster @{face_region}")
    plt.show()
