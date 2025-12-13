"""
Organic image segmentation for natural layer detection.
Finds layers by following actual object boundaries instead of hard horizontal cuts.
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional


def segment_into_natural_layers(
    image_rgb: np.ndarray,
    target_layers: int = 8,
    min_coverage: float = 0.01
) -> List[Dict]:
    """
    Segment image into natural layers using color and luminosity segmentation.

    Instead of hard horizontal cuts, this:
    1. Segments image by color similarity (quantization + connected components)
    2. Groups regions into target number of layers by luminosity
    3. Creates organic layer masks that follow actual boundaries
    4. Orders layers back-to-front for painting

    Args:
        image_rgb: RGB image (H, W, 3)
        target_layers: Target number of layers to produce (5-12 recommended)
        min_coverage: Minimum coverage for a layer (as fraction of image)

    Returns:
        List of layer dicts with organic masks, ordered for painting
    """
    h, w = image_rgb.shape[:2]
    total_pixels = h * w

    # Convert to LAB for better perceptual segmentation
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    luminosity = lab[:, :, 0].astype(np.float32) / 255.0

    # Use color quantization + connected components for segmentation
    from ..core.paint_by_numbers import PaintByNumbers

    # Quantize to find color regions
    n_colors = 12

    import tempfile
    import os
    temp_path = tempfile.mktemp(suffix='.png')
    cv2.imwrite(temp_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))

    pbn = PaintByNumbers(temp_path, n_colors)
    pbn.quantize_colors()
    label_map = pbn.label_map

    os.unlink(temp_path)

    # Find connected components for each color - these are our organic segments
    all_segments = []
    segment_id = 0

    for color_id in range(n_colors):
        color_mask = (label_map == color_id).astype(np.uint8)
        num_labels, labels = cv2.connectedComponents(color_mask)

        for component_id in range(1, num_labels):
            mask = (labels == component_id)
            size = np.sum(mask)

            if size < 50:  # Skip tiny segments
                continue

            y_coords, x_coords = np.where(mask)
            seg_lum = luminosity[mask]
            seg_colors = image_rgb[mask]

            all_segments.append({
                'id': segment_id,
                'mask': mask,
                'size': size,
                'coverage': size / total_pixels,
                'avg_lum': np.mean(seg_lum),
                'avg_y': np.mean(y_coords) / h,
                'y_min': np.min(y_coords) / h,
                'y_max': np.max(y_coords) / h,
                'avg_color': np.mean(seg_colors, axis=0),
            })
            segment_id += 1

    if not all_segments:
        return []

    # Group segments into target_layers using luminosity bins
    layers = _group_segments_by_luminosity(all_segments, h, w, target_layers, min_coverage)

    # Ensure 100% coverage - assign any uncovered pixels to nearest layer by luminosity
    layers = _fill_coverage_gaps(layers, luminosity, h, w)

    # Sort by painting order: back-to-front (top of image first, bottom last)
    # Primary: average y position (lower y = higher in image = background)
    # Secondary: luminosity as tiebreaker for overlapping regions
    layers = sorted(layers, key=lambda l: (l['y_range'][0], -l['avg_luminosity']))

    # Assign names and priorities
    for i, layer in enumerate(layers):
        layer['priority'] = i + 1
        layer['name'] = _generate_layer_name(layer, i, len(layers))

    return layers


def _group_segments_by_luminosity(
    segments: List[Dict],
    h: int, w: int,
    target_layers: int,
    min_coverage: float
) -> List[Dict]:
    """
    Group segments into target number of layers based on luminosity.
    Preserves organic boundaries by combining segment masks.
    """
    if not segments:
        return []

    # Get luminosity range
    lum_values = [s['avg_lum'] for s in segments]
    lum_min, lum_max = min(lum_values), max(lum_values)
    lum_range = lum_max - lum_min + 0.001

    # Create bins based on luminosity
    bin_size = lum_range / target_layers
    bins = [[] for _ in range(target_layers)]

    for seg in segments:
        bin_idx = min(int((seg['avg_lum'] - lum_min) / bin_size), target_layers - 1)
        bins[bin_idx].append(seg)

    # Convert non-empty bins to layers
    layers = []
    for bin_idx, bin_segments in enumerate(bins):
        if not bin_segments:
            continue

        # Combine all segment masks in this bin (preserves organic boundaries!)
        combined_mask = np.zeros((h, w), dtype=bool)
        for seg in bin_segments:
            combined_mask |= seg['mask']

        coverage = np.sum(combined_mask) / (h * w)

        # Skip very tiny layers - they'll be orphaned pixels
        if coverage < min_coverage / 4:
            continue

        avg_lum = np.mean([s['avg_lum'] for s in bin_segments])
        all_y_mins = [s['y_min'] for s in bin_segments]
        all_y_maxs = [s['y_max'] for s in bin_segments]

        # Determine technique
        if avg_lum > 0.7:
            technique = 'highlight'
        elif avg_lum > 0.5:
            technique = 'blend'
        elif avg_lum < 0.25:
            technique = 'silhouette'
        else:
            technique = 'layer'

        layers.append({
            'mask': combined_mask,
            'coverage': coverage,
            'avg_luminosity': avg_lum,
            'y_range': (min(all_y_mins), max(all_y_maxs)),
            'technique': technique,
            'num_segments': len(bin_segments),
        })

    # Merge small layers into neighbors
    layers = _merge_small_layers(layers, h, w, min_coverage)

    return layers


def _fill_coverage_gaps(
    layers: List[Dict],
    luminosity: np.ndarray,
    h: int, w: int
) -> List[Dict]:
    """
    Fill any gaps in coverage by assigning uncovered pixels to nearest layer by luminosity.
    Ensures 100% of pixels are covered.
    """
    if not layers:
        return layers

    # Find uncovered pixels
    covered = np.zeros((h, w), dtype=bool)
    for layer in layers:
        covered |= layer['mask']

    uncovered = ~covered
    n_uncovered = np.sum(uncovered)

    if n_uncovered == 0:
        return layers  # Already 100% covered

    # Get luminosity of each layer as array
    layer_lums = np.array([l['avg_luminosity'] for l in layers])

    # Get luminosity values of uncovered pixels
    uncovered_lums = luminosity[uncovered]

    # Find closest layer for each uncovered pixel (vectorized)
    # Compute distance from each uncovered pixel to each layer luminosity
    diffs = np.abs(uncovered_lums[:, np.newaxis] - layer_lums[np.newaxis, :])
    best_layer_indices = np.argmin(diffs, axis=1)

    # Assign uncovered pixels to their best matching layers
    uncovered_coords = np.where(uncovered)
    for layer_idx in range(len(layers)):
        # Find which uncovered pixels go to this layer
        pixel_mask = (best_layer_indices == layer_idx)
        if np.any(pixel_mask):
            ys = uncovered_coords[0][pixel_mask]
            xs = uncovered_coords[1][pixel_mask]
            layers[layer_idx]['mask'][ys, xs] = True

    # Update coverage for all layers
    total_pixels = h * w
    for layer in layers:
        layer['coverage'] = np.sum(layer['mask']) / total_pixels

    return layers


def _merge_small_layers(
    layers: List[Dict],
    h: int, w: int,
    min_coverage: float
) -> List[Dict]:
    """Merge layers below min_coverage into their nearest neighbor by luminosity."""
    if len(layers) <= 1:
        return layers

    # Iterate until no small layers remain
    changed = True
    while changed:
        changed = False

        large = [l for l in layers if l['coverage'] >= min_coverage]
        small = [l for l in layers if l['coverage'] < min_coverage]

        if not small or not large:
            break

        # Merge smallest layer into nearest large layer
        smallest = min(small, key=lambda l: l['coverage'])
        nearest = min(large, key=lambda l: abs(l['avg_luminosity'] - smallest['avg_luminosity']))

        # Merge
        nearest['mask'] = nearest['mask'] | smallest['mask']
        nearest['coverage'] = np.sum(nearest['mask']) / (h * w)
        nearest['num_segments'] += smallest['num_segments']
        nearest['y_range'] = (
            min(nearest['y_range'][0], smallest['y_range'][0]),
            max(nearest['y_range'][1], smallest['y_range'][1])
        )

        # Remove the merged small layer
        layers = [l for l in layers if l is not smallest]
        changed = True

    return layers


def _generate_layer_name(layer: Dict, index: int, total: int) -> str:
    """Generate descriptive name for a layer based on its properties."""
    y_min, y_max = layer['y_range']
    lum = layer['avg_luminosity']

    # Position-based naming
    if y_max < 0.35:
        position = "sky"
    elif y_min > 0.65:
        position = "foreground"
    elif y_min > 0.4 and y_max < 0.7:
        position = "midground"
    else:
        position = "background"

    # Luminosity-based description
    if lum > 0.75:
        tone = "highlights"
    elif lum > 0.55:
        tone = "lights"
    elif lum > 0.35:
        tone = "midtones"
    elif lum > 0.2:
        tone = "shadows"
    else:
        tone = "darks"

    layer_num = index + 1
    return f"L{layer_num}_{position}_{tone}"


def watershed_segmentation(image_rgb: np.ndarray) -> np.ndarray:
    """
    Alternative: Watershed segmentation for more precise object boundaries.
    Use this for images with clear distinct objects.
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    markers = cv2.watershed(image_rgb, markers)

    return markers
