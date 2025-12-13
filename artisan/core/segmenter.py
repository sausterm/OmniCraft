"""
Image Segmentation - Break images into layers and color regions.

Provides tools for:
1. Layer segmentation (luminosity, depth, semantic)
2. Color quantization and clustering
3. Layer boundary detection
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from enum import Enum


class LayerStrategy(Enum):
    """Strategy for determining layer boundaries."""
    LUMINOSITY = "luminosity"           # Dark to light
    DEPTH = "depth"                     # Far to near (vertical gradient)
    SEMANTIC = "semantic"               # Sky, water, foreground, etc.
    ADAPTIVE = "adaptive"               # Analyze image to pick best strategy


@dataclass
class LayerDefinition:
    """Definition of a logical painting/creation layer."""
    name: str
    mask: np.ndarray                    # Boolean mask of pixels in this layer
    technique: str                      # Suggested technique for this layer
    description: str                    # Human-readable overview
    order_priority: int = 0             # Lower = earlier in execution
    avg_luminosity: float = 0.5         # Average brightness
    avg_depth: float = 0.5              # Average depth (0=far, 1=near)


class ImageSegmenter:
    """Segments images into layers based on various strategies."""

    def __init__(self, image: np.ndarray):
        """
        Initialize segmenter with image.

        Args:
            image: RGB image as numpy array
        """
        self.image = image
        self.height, self.width = image.shape[:2]

        # Precompute useful maps
        self.lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        self.luminosity = self.lab[:, :, 0].astype(np.float32) / 255.0
        self.depth_map = self._estimate_depth()

    def _estimate_depth(self) -> np.ndarray:
        """
        Estimate depth from image (far to near).

        Uses heuristics:
        - Upper image = farther (sky, background)
        - Lower image = nearer (foreground, objects)
        - Darker areas in lower half = foreground

        Returns:
            Depth map (0 = far, 1 = near)
        """
        # Start with vertical gradient (top = 0, bottom = 1)
        y_coords = np.linspace(0, 1, self.height)
        depth = np.tile(y_coords[:, np.newaxis], (1, self.width))

        # Enhance foreground: dark objects in lower half are closer
        lower_half_mask = depth > 0.5
        dark_mask = self.luminosity < 0.25

        # Foreground objects (dark + low in frame)
        foreground_mask = lower_half_mask & dark_mask
        depth[foreground_mask] = 0.9  # Pull to foreground

        return depth

    def segment_by_luminosity(
        self,
        thresholds: Optional[List[float]] = None
    ) -> List[LayerDefinition]:
        """
        Segment image into layers based on luminosity (dark to light).

        Good for paintings where you work from dark backgrounds to bright highlights.

        Args:
            thresholds: Luminosity cutoff points (0-1). Default: [0.25, 0.55, 0.8]

        Returns:
            List of LayerDefinition objects, sorted by luminosity
        """
        if thresholds is None:
            thresholds = [0.25, 0.55, 0.8]

        layers = []
        prev_threshold = 0.0

        # Define layer names and techniques
        layer_info = [
            ("Dark Foundation", "base coat", "Establish the darkest values"),
            ("Mid-Tone Layer", "blend", "Build the middle values"),
            ("Bright Areas", "glazing", "Add luminous colors"),
            ("Highlights", "accent", "Brightest highlights and details"),
        ]

        for i, threshold in enumerate(thresholds + [1.1]):  # Add end threshold
            mask = (self.luminosity >= prev_threshold) & (self.luminosity < threshold)
            pixel_count = np.sum(mask)

            # Only create layer if it has enough pixels
            if pixel_count > 0.01 * self.height * self.width:
                name, technique, description = layer_info[min(i, len(layer_info) - 1)]

                layers.append(LayerDefinition(
                    name=name,
                    mask=mask,
                    technique=technique,
                    description=description,
                    order_priority=i,
                    avg_luminosity=float(np.mean(self.luminosity[mask])),
                    avg_depth=float(np.mean(self.depth_map[mask]))
                ))

            prev_threshold = threshold

        return layers

    def segment_by_depth(
        self,
        n_layers: int = 4
    ) -> List[LayerDefinition]:
        """
        Segment image into layers based on estimated depth (far to near).

        Good for landscapes and scenes with clear foreground/background.

        Args:
            n_layers: Number of depth layers to create

        Returns:
            List of LayerDefinition objects, sorted far to near
        """
        layers = []

        # Divide depth into n_layers
        for i in range(n_layers):
            depth_min = i / n_layers
            depth_max = (i + 1) / n_layers

            mask = (self.depth_map >= depth_min) & (self.depth_map < depth_max)
            pixel_count = np.sum(mask)

            if pixel_count > 0.01 * self.height * self.width:
                # Name based on depth
                if i == 0:
                    name = "Far Background"
                    description = "Distant elements (sky, mountains)"
                elif i == n_layers - 1:
                    name = "Foreground"
                    description = "Closest elements"
                else:
                    name = f"Middle Ground {i}"
                    description = f"Mid-distance layer {i}"

                layers.append(LayerDefinition(
                    name=name,
                    mask=mask,
                    technique="blend" if i < n_layers - 1 else "detail",
                    description=description,
                    order_priority=i,
                    avg_luminosity=float(np.mean(self.luminosity[mask])),
                    avg_depth=depth_min + (depth_max - depth_min) / 2
                ))

        return layers

    def segment_semantic(self) -> List[LayerDefinition]:
        """
        Segment image into semantic layers (sky, water, foreground, etc.).

        Uses heuristics and simple CV to identify meaningful regions.

        Returns:
            List of LayerDefinition objects with semantic meaning
        """
        layers = []
        order = 0

        # 1. Sky (upper, bright)
        sky_mask = (self.depth_map < 0.3) & (self.luminosity > 0.4)
        if np.sum(sky_mask) > 0.05 * self.height * self.width:
            layers.append(LayerDefinition(
                name="Sky",
                mask=sky_mask,
                technique="gradient",
                description="Sky and atmospheric background",
                order_priority=order,
                avg_luminosity=float(np.mean(self.luminosity[sky_mask])),
                avg_depth=0.1
            ))
            order += 1

        # 2. Water/Horizon (middle, moderate brightness)
        water_mask = (
            (self.depth_map >= 0.3) & (self.depth_map < 0.6) &
            (self.luminosity > 0.3)
        )
        water_mask = water_mask & ~sky_mask  # Exclude sky
        if np.sum(water_mask) > 0.05 * self.height * self.width:
            layers.append(LayerDefinition(
                name="Middle Ground",
                mask=water_mask,
                technique="blend",
                description="Horizon, water, or middle distance",
                order_priority=order,
                avg_luminosity=float(np.mean(self.luminosity[water_mask])),
                avg_depth=0.45
            ))
            order += 1

        # 3. Bright features (aurora, glow, reflections)
        bright_mask = (self.luminosity > 0.7) & ~sky_mask
        if np.sum(bright_mask) > 0.02 * self.height * self.width:
            layers.append(LayerDefinition(
                name="Luminous Features",
                mask=bright_mask,
                technique="glazing",
                description="Glowing, bright elements",
                order_priority=order,
                avg_luminosity=float(np.mean(self.luminosity[bright_mask])),
                avg_depth=float(np.mean(self.depth_map[bright_mask]))
            ))
            order += 1

        # 4. Foreground silhouettes (lower, dark)
        fg_mask = (self.depth_map > 0.6) & (self.luminosity < 0.25)
        if np.sum(fg_mask) > 0.02 * self.height * self.width:
            layers.append(LayerDefinition(
                name="Foreground Silhouettes",
                mask=fg_mask,
                technique="silhouette",
                description="Dark foreground shapes",
                order_priority=order,
                avg_luminosity=float(np.mean(self.luminosity[fg_mask])),
                avg_depth=0.9
            ))
            order += 1

        # 5. Catch remaining pixels
        used_mask = sum([layer.mask for layer in layers]) > 0
        remaining_mask = ~used_mask
        if np.sum(remaining_mask) > 0.03 * self.height * self.width:
            layers.append(LayerDefinition(
                name="Base Layer",
                mask=remaining_mask,
                technique="base coat",
                description="Foundation and remaining areas",
                order_priority=-1,  # Should be first
                avg_luminosity=float(np.mean(self.luminosity[remaining_mask])),
                avg_depth=float(np.mean(self.depth_map[remaining_mask]))
            ))

        # Re-sort by order priority
        layers.sort(key=lambda l: l.order_priority)

        return layers

    def segment(
        self,
        strategy: LayerStrategy = LayerStrategy.ADAPTIVE
    ) -> List[LayerDefinition]:
        """
        Segment image using specified strategy.

        Args:
            strategy: Which segmentation approach to use

        Returns:
            List of LayerDefinition objects
        """
        if strategy == LayerStrategy.LUMINOSITY:
            return self.segment_by_luminosity()
        elif strategy == LayerStrategy.DEPTH:
            return self.segment_by_depth()
        elif strategy == LayerStrategy.SEMANTIC:
            return self.segment_semantic()
        elif strategy == LayerStrategy.ADAPTIVE:
            # Analyze image to pick best strategy
            return self._adaptive_segment()
        else:
            # Default to luminosity
            return self.segment_by_luminosity()

    def _adaptive_segment(self) -> List[LayerDefinition]:
        """
        Choose best segmentation strategy based on image analysis.

        Returns:
            Layer definitions using optimal strategy
        """
        # Analyze image characteristics
        lum_variance = np.var(self.luminosity)
        depth_variance = np.var(self.depth_map)

        # High luminosity variance → use luminosity segmentation
        if lum_variance > 0.08:
            return self.segment_by_luminosity()

        # Check for clear foreground/background separation
        has_dark_foreground = np.sum(
            (self.depth_map > 0.6) & (self.luminosity < 0.3)
        ) > 0.1 * self.height * self.width

        if has_dark_foreground:
            return self.segment_semantic()

        # Default to depth-based
        return self.segment_by_depth()


def find_colors_in_layer(
    color_labels: np.ndarray,
    layer_mask: np.ndarray,
    min_pixels: int = 100
) -> List[Tuple[int, int, float]]:
    """
    Find which colors appear in a specific layer.

    Args:
        color_labels: Label map from color quantization (HxW array of color indices)
        layer_mask: Boolean mask of layer region
        min_pixels: Minimum pixel count to include a color

    Returns:
        List of (color_index, pixel_count, coverage_fraction) tuples,
        sorted by coverage (largest first)
    """
    colors_in_layer = []
    total_layer_pixels = np.sum(layer_mask)

    if total_layer_pixels == 0:
        return []

    # Find unique color indices in this layer
    colors_present = np.unique(color_labels[layer_mask])

    for color_idx in colors_present:
        # Count pixels of this color in this layer
        color_mask = color_labels == color_idx
        overlap = color_mask & layer_mask
        pixel_count = int(np.sum(overlap))

        if pixel_count >= min_pixels:
            coverage = pixel_count / total_layer_pixels
            colors_in_layer.append((int(color_idx), pixel_count, float(coverage)))

    # Sort by pixel count (descending)
    colors_in_layer.sort(key=lambda x: -x[1])

    return colors_in_layer
