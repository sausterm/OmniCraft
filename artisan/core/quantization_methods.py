"""
Alternative color quantization methods for better color fidelity.
"""

import numpy as np
import cv2
from sklearn.cluster import KMeans


def median_cut_quantize(image_rgb: np.ndarray, n_colors: int) -> tuple:
    """
    Median cut algorithm - preserves color fidelity better than k-means.

    Works by recursively splitting the color space at the median,
    preserving dominant colors better than clustering.

    Args:
        image_rgb: RGB image (H, W, 3)
        n_colors: Target number of colors

    Returns:
        (quantized_image, palette)
    """
    from PIL import Image

    # Convert to PIL for median cut
    pil_img = Image.fromarray(image_rgb)

    # Quantize using median cut (ADAPTIVE method)
    quantized_pil = pil_img.quantize(colors=n_colors, method=Image.Quantize.MEDIANCUT)

    # Convert back to RGB
    quantized_pil_rgb = quantized_pil.convert('RGB')
    quantized_array = np.array(quantized_pil_rgb)

    # Extract palette
    palette_pil = quantized_pil.getpalette()
    palette = np.array(palette_pil[:n_colors*3]).reshape(n_colors, 3)

    return quantized_array, palette


def refined_kmeans_quantize(image_rgb: np.ndarray, n_colors: int) -> tuple:
    """
    K-means with post-processing refinement to better match original colors.

    After initial clustering, adjusts cluster centers to median color
    of pixels in each cluster for better color fidelity.

    Args:
        image_rgb: RGB image (H, W, 3)
        n_colors: Target number of colors

    Returns:
        (quantized_image, palette, label_map)
    """
    # Convert to LAB (perceptually uniform)
    lab_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    h, w = image_rgb.shape[:2]
    pixels_lab = lab_image.reshape(-1, 3)
    pixels_rgb = image_rgb.reshape(-1, 3)

    # Initial k-means clustering
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixels_lab)

    # Refine cluster centers using median of actual RGB values
    refined_centers_rgb = []
    for i in range(n_colors):
        cluster_pixels = pixels_rgb[labels == i]
        if len(cluster_pixels) > 0:
            # Use median instead of mean for better color preservation
            median_color = np.median(cluster_pixels, axis=0)
            refined_centers_rgb.append(median_color)
        else:
            # Fallback to k-means center if cluster is empty
            centers_lab_reshaped = kmeans.cluster_centers_[i:i+1].reshape(1, 1, 3).astype(np.uint8)
            center_rgb = cv2.cvtColor(centers_lab_reshaped, cv2.COLOR_LAB2RGB)
            refined_centers_rgb.append(center_rgb.flatten())

    palette = np.array(refined_centers_rgb).astype(np.uint8)

    # Create quantized image using refined palette
    quantized_pixels = palette[labels]
    quantized_image = quantized_pixels.reshape(h, w, 3)
    label_map = labels.reshape(h, w)

    return quantized_image, palette, label_map


def spatial_kmeans_quantize(image_rgb: np.ndarray, n_colors: int, spatial_weight: float = 0.3) -> tuple:
    """
    K-means with spatial information - keeps similar regions together.

    Clusters based on both color AND position, preventing random
    spatial distribution of colors (fixes the "weird sky" problem).

    Args:
        image_rgb: RGB image (H, W, 3)
        n_colors: Target number of colors
        spatial_weight: How much to weight position vs color (0-1)

    Returns:
        (quantized_image, palette, label_map)
    """
    # Convert to LAB
    lab_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    h, w = image_rgb.shape[:2]

    # Create position grid
    y_coords, x_coords = np.mgrid[0:h, 0:w]

    # Normalize positions to [0, 1]
    y_norm = y_coords.reshape(-1) / h
    x_norm = x_coords.reshape(-1) / w

    # Combine color and spatial features
    pixels_lab = lab_image.reshape(-1, 3)
    spatial_features = np.column_stack([
        pixels_lab[:, 0],  # L
        pixels_lab[:, 1],  # A
        pixels_lab[:, 2],  # B
        y_norm * spatial_weight * 100,  # Y position (scaled)
        x_norm * spatial_weight * 100   # X position (scaled)
    ])

    # Cluster with spatial information
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
    labels = kmeans.fit_predict(spatial_features)

    # Extract just the color part of cluster centers
    centers_lab = kmeans.cluster_centers_[:, :3]

    # Convert to RGB
    centers_lab_reshaped = centers_lab.reshape(1, -1, 3).astype(np.uint8)
    centers_rgb = cv2.cvtColor(centers_lab_reshaped, cv2.COLOR_LAB2RGB)
    palette = centers_rgb.reshape(n_colors, 3)

    # Refine with actual pixel median (like refined_kmeans)
    pixels_rgb = image_rgb.reshape(-1, 3)
    refined_palette = []
    for i in range(n_colors):
        cluster_pixels = pixels_rgb[labels == i]
        if len(cluster_pixels) > 0:
            median_color = np.median(cluster_pixels, axis=0)
            refined_palette.append(median_color)
        else:
            refined_palette.append(palette[i])

    palette = np.array(refined_palette).astype(np.uint8)

    # Create quantized image
    quantized_pixels = palette[labels]
    quantized_image = quantized_pixels.reshape(h, w, 3)
    label_map = labels.reshape(h, w)

    return quantized_image, palette, label_map
