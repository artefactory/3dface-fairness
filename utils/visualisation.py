import os
from collections import defaultdict
import math

import matplotlib.pyplot as plt
from PIL import Image


def plot_clusters_from_path(
    images_path,
    labels,
    max_cols=4,
    base_figsize_per_image=2,
    extensions=(".png", ".jpg", ".jpeg", ".bmp"),
):
    """Visualize clustered face images as a set of per-cluster grids."""

    image_files = sorted([f for f in os.listdir(images_path) if f.lower().endswith(extensions)])

    assert len(image_files) == len(labels), "Number of labels must match number of images"

    clusters = defaultdict(list)
    for i, label in enumerate(labels):
        clusters[label].append(os.path.join(images_path, f"{i + 1}.jpg"))

    cluster_items = sorted(clusters.items())
    n_clusters = len(cluster_items)

    layouts = []
    height_ratios = []

    for _, imgs in cluster_items:
        n_imgs = len(imgs)
        n_cols = min(max_cols, n_imgs)
        n_rows = math.ceil(n_imgs / n_cols)

        layouts.append((n_rows, n_cols))
        height_ratios.append(n_rows)

    total_rows = sum(height_ratios)
    fig_height = 2 * base_figsize_per_image * total_rows
    fig_width = 2 * base_figsize_per_image * max_cols

    fig = plt.figure(figsize=(fig_width, fig_height))
    outer = fig.add_gridspec(n_clusters, 1, height_ratios=height_ratios, hspace=0.4)

    for i, ((label, img_paths), (n_rows, n_cols)) in enumerate(zip(cluster_items, layouts)):
        inner = outer[i].subgridspec(n_rows, n_cols, wspace=0.02, hspace=0.02)

        for j, img_path in enumerate(img_paths):
            ax = fig.add_subplot(inner[j])
            img = Image.open(img_path).convert("RGB")
            ax.imshow(img)
            ax.axis("off")

        for j in range(len(img_paths), n_rows * n_cols):
            ax = fig.add_subplot(inner[j])
            ax.axis("off")

        ax = fig.add_subplot(outer[i])
        ax.set_title(f"Cluster {label}", pad=8)
        ax.axis("off")

    plt.show()
