"""
Simplified organic segmentation - creates 6-10 natural layers instead of hundreds.
"""

import numpy as np
import cv2
from typing import List, Dict


def segment_into_painting_layers(image_rgb: np.ndarray, n_layers: int = 8) -> List[Dict]:
    """
    Create painting layers using simplified color+luminosity segmentation.

    Creates one layer per major color region, ordered back-to-front.

    Args:
        image_rgb: RGB image
        n_layers: Target number of painting layers (6-12)

    Returns:
        List of layer dicts with organic masks
    """
    h, w = image_rgb.shape[:2]

    # Convert to LAB
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    luminosity = lab[:, :, 0].astype(np.float32) / 255.0

    # Quantize to find major color regions
    from ..core.paint_by_numbers import PaintByNumbers
    import tempfile, os

    temp_path = tempfile.mktemp(suffix='.png')
    cv2.imwrite(temp_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))

    pbn = PaintByNumbers(temp_path, n_layers)
    pbn.quantize_colors()
    label_map = pbn.label_map

    os.unlink(temp_path)

    # Create one layer per color
    layers = []

    for color_id in range(n_layers):
        # Get all pixels of this color
        mask = (label_map == color_id)

        # Skip if too small
        coverage = np.sum(mask) / (h * w)
        if coverage < 0.01:  # Skip layers < 1%
            continue

        # Get properties
        y_coords, x_coords = np.where(mask)
        if len(y_coords) == 0:
            continue

        avg_y = np.mean(y_coords) / h
        avg_lum = np.mean(luminosity[mask])

        # Determine technique
        if avg_lum > 0.7:
            technique = 'highlight'
        elif avg_lum > 0.5:
            technique = 'blend'
        elif avg_lum < 0.25:
            technique = 'silhouette'
        else:
            technique = 'layer'

        # Generate name
        name = _generate_simple_layer_name(avg_y, avg_lum, coverage)

        # Generate description
        description = f"{name.replace('_', ' ')}"

        layers.append({
            'name': name,
            'description': description,
            'mask': mask,
            'coverage': coverage,
            'avg_luminosity': avg_lum,
            'y_range': (np.min(y_coords) / h, np.max(y_coords) / h),
            'technique': technique,
            'priority': 0  # Will be set after sorting
        })

    # Sort by painting order: brightest+highest first (sky), then darker+lower (foreground)
    layers.sort(key=lambda l: (-l['avg_luminosity'], l['y_range'][0]))

    # Assign priorities
    for i, layer in enumerate(layers):
        layer['priority'] = i + 1

    return layers


def _generate_simple_layer_name(avg_y: float, avg_lum: float, coverage: float) -> str:
    """Generate descriptive layer name."""

    # Position
    if avg_y < 0.35:
        position = "sky"
    elif avg_y > 0.7:
        position = "foreground"
    elif avg_y > 0.5:
        position = "midground"
    else:
        position = "background"

    # Tone
    if avg_lum > 0.7:
        tone = "highlights"
    elif avg_lum > 0.5:
        tone = "light"
    elif avg_lum > 0.3:
        tone = "mid"
    else:
        tone = "dark"

    # Special cases
    if position == "sky":
        if avg_lum > 0.6:
            return "sky_highlights"
        else:
            return "sky_base"
    elif position == "foreground":
        if coverage > 0.2:
            return f"foreground_main"
        else:
            return f"foreground_{tone}"
    else:
        return f"{position}_{tone}"
