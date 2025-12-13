"""
Depth-Based Layer Ordering - Orders painting layers by spatial depth.

The fundamental principle: paint back-to-front.
- Furthest elements first (sky, distant hills)
- Then middle ground (trees, grass)
- Then foreground subjects (dogs, people)
- Details and refinements last

This module analyzes an image to determine the depth/z-order of regions
and groups them into paintable layers ordered by perspective.
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum


class DepthZone(Enum):
    """Depth zones from back to front."""
    FAR_BACKGROUND = 0      # Sky, distant mountains
    BACKGROUND = 1          # Hills, treelines, far elements
    MIDGROUND = 2           # Grass fields, middle distance
    FOREGROUND = 3          # Main subjects, close elements
    NEAR_FOREGROUND = 4     # Closest elements, overlapping subjects


@dataclass
class SpatialLayer:
    """A layer defined by spatial depth, not color."""
    id: str
    name: str
    depth_zone: DepthZone
    depth_value: float              # 0.0 = furthest, 1.0 = nearest

    mask: np.ndarray
    coverage: float

    # Spatial bounds
    y_top: float                    # Top of layer (0-1)
    y_bottom: float                 # Bottom of layer (0-1)

    # Visual properties
    dominant_color: Tuple[int, int, int]
    avg_luminosity: float

    # What this layer represents
    description: str = ""
    semantic_type: str = "region"   # sky, hills, trees, grass, subject, etc.

    # Painting info
    paint_order: int = 0
    technique: str = "blocking"
    tips: List[str] = field(default_factory=list)


class DepthAnalyzer:
    """
    Analyzes image to determine depth layers for painting order.

    Uses multiple signals:
    1. Vertical position (top = far, bottom = near in most images)
    2. Color saturation (distant = less saturated, atmospheric perspective)
    3. Detail/texture (distant = smoother, less detail)
    4. Object size and position
    """

    def __init__(self):
        # Typical depth zones by vertical position (can be adjusted)
        self.zone_boundaries = {
            DepthZone.FAR_BACKGROUND: (0.0, 0.25),    # Top 25%
            DepthZone.BACKGROUND: (0.15, 0.45),       # Upper-mid
            DepthZone.MIDGROUND: (0.35, 0.75),        # Middle
            DepthZone.FOREGROUND: (0.5, 1.0),         # Lower half
            DepthZone.NEAR_FOREGROUND: (0.7, 1.0),    # Bottom 30%
        }

    def analyze_depth(
        self,
        image: np.ndarray,
        segments: List[Dict]
    ) -> List[SpatialLayer]:
        """
        Analyze segments and assign depth-based layers.

        Args:
            image: RGB image
            segments: List of segment dicts with 'mask' key

        Returns:
            List of SpatialLayers ordered by depth (back to front)
        """
        h, w = image.shape[:2]

        # Analyze each segment for depth cues
        analyzed_segments = []
        for i, seg in enumerate(segments):
            mask = seg.get('mask', seg.get('segmentation', None))
            if mask is None:
                continue

            depth_info = self._analyze_segment_depth(image, mask, h, w)
            depth_info['segment'] = seg
            depth_info['id'] = f"layer_{i:02d}"
            analyzed_segments.append(depth_info)

        # Group segments into spatial layers by depth zone
        layers = self._group_into_depth_layers(analyzed_segments, image, h, w)

        # Order layers back to front
        layers = sorted(layers, key=lambda l: l.depth_value)

        # Assign paint order
        for i, layer in enumerate(layers):
            layer.paint_order = i + 1

        return layers

    def _analyze_segment_depth(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        h: int, w: int
    ) -> Dict:
        """Analyze depth cues for a single segment."""
        y_coords, x_coords = np.where(mask)

        if len(y_coords) == 0:
            return {'depth': 0.5, 'zone': DepthZone.MIDGROUND}

        # Vertical position (strongest depth cue for landscapes)
        y_top = y_coords.min() / h
        y_bottom = y_coords.max() / h
        y_center = y_coords.mean() / h

        # Get colors in segment
        masked_pixels = image[mask]

        # Average color
        avg_color = tuple(np.median(masked_pixels, axis=0).astype(int))

        # Luminosity
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        avg_luminosity = np.mean(hsv[mask, 2]) / 255.0

        # Saturation (lower = more distant, atmospheric perspective)
        avg_saturation = np.mean(hsv[mask, 1]) / 255.0

        # Detail level (Laplacian variance - lower = smoother = more distant)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        masked_gray = gray.copy()
        masked_gray[~mask] = 0
        laplacian = cv2.Laplacian(masked_gray, cv2.CV_64F)
        detail_level = np.var(laplacian[mask]) if np.any(mask) else 0
        detail_normalized = min(1.0, detail_level / 1000.0)

        # Calculate depth value (0 = far, 1 = near)
        # Primary: vertical position (top = far)
        # Secondary: saturation (low = far)
        # Tertiary: detail level (low = far)
        depth = (
            y_center * 0.6 +           # Position is dominant
            avg_saturation * 0.25 +     # Saturation matters
            detail_normalized * 0.15    # Detail is minor factor
        )

        # Determine depth zone
        zone = self._classify_depth_zone(y_top, y_bottom, y_center, depth)

        return {
            'depth': depth,
            'zone': zone,
            'y_top': y_top,
            'y_bottom': y_bottom,
            'y_center': y_center,
            'avg_color': avg_color,
            'avg_luminosity': avg_luminosity,
            'avg_saturation': avg_saturation,
            'detail_level': detail_normalized,
            'coverage': np.sum(mask) / (h * w),
            'mask': mask,
        }

    def _classify_depth_zone(
        self,
        y_top: float,
        y_bottom: float,
        y_center: float,
        depth: float
    ) -> DepthZone:
        """Classify segment into a depth zone."""
        # Sky/far background - top of image, usually bright
        if y_bottom < 0.35 and y_top < 0.15:
            return DepthZone.FAR_BACKGROUND

        # Background - upper portion
        if y_center < 0.35:
            return DepthZone.BACKGROUND

        # Near foreground - bottom of image
        if y_top > 0.6:
            return DepthZone.NEAR_FOREGROUND

        # Foreground - lower half, larger coverage usually
        if y_center > 0.5:
            return DepthZone.FOREGROUND

        # Default to midground
        return DepthZone.MIDGROUND

    def _group_into_depth_layers(
        self,
        analyzed_segments: List[Dict],
        image: np.ndarray,
        h: int, w: int
    ) -> List[SpatialLayer]:
        """Group analyzed segments into depth-based layers."""

        # Group by depth zone first
        zone_groups = {}
        for seg in analyzed_segments:
            zone = seg['zone']
            if zone not in zone_groups:
                zone_groups[zone] = []
            zone_groups[zone].append(seg)

        layers = []

        for zone in DepthZone:
            if zone not in zone_groups:
                continue

            segments = zone_groups[zone]

            # For each zone, we might have multiple distinct objects
            # Group by spatial proximity and color similarity
            sub_groups = self._cluster_segments_in_zone(segments)

            for i, group in enumerate(sub_groups):
                # Combine masks
                combined_mask = np.zeros((h, w), dtype=bool)
                for seg in group:
                    combined_mask |= seg['mask']

                # Calculate combined properties
                avg_depth = np.mean([s['depth'] for s in group])
                avg_y_top = min(s['y_top'] for s in group)
                avg_y_bottom = max(s['y_bottom'] for s in group)

                # Get dominant color from combined region
                masked_pixels = image[combined_mask]
                if len(masked_pixels) > 0:
                    dominant_color = tuple(np.median(masked_pixels, axis=0).astype(int))
                    avg_lum = np.mean([s['avg_luminosity'] for s in group])
                else:
                    dominant_color = (128, 128, 128)
                    avg_lum = 0.5

                # Generate descriptive name
                name, description, semantic_type = self._generate_layer_info(
                    zone, avg_y_top, avg_y_bottom, avg_lum, len(sub_groups), i
                )

                # Get technique and tips
                technique, tips = self._get_painting_guidance(zone, semantic_type, avg_lum)

                layer = SpatialLayer(
                    id=f"{zone.name.lower()}_{i}",
                    name=name,
                    depth_zone=zone,
                    depth_value=avg_depth,
                    mask=combined_mask,
                    coverage=np.sum(combined_mask) / (h * w),
                    y_top=avg_y_top,
                    y_bottom=avg_y_bottom,
                    dominant_color=dominant_color,
                    avg_luminosity=avg_lum,
                    description=description,
                    semantic_type=semantic_type,
                    technique=technique,
                    tips=tips,
                )
                layers.append(layer)

        return layers

    def _cluster_segments_in_zone(self, segments: List[Dict]) -> List[List[Dict]]:
        """
        Cluster segments within a zone into distinct objects.
        Segments that are spatially connected or very similar get grouped.
        """
        if len(segments) <= 1:
            return [segments] if segments else []

        # Simple clustering: group by horizontal position for now
        # More sophisticated: use connected components or color similarity

        # Sort by horizontal center
        for seg in segments:
            y_coords, x_coords = np.where(seg['mask'])
            seg['x_center'] = np.mean(x_coords) if len(x_coords) > 0 else 0

        segments = sorted(segments, key=lambda s: s['x_center'])

        # Group spatially close segments
        groups = []
        current_group = [segments[0]]

        for seg in segments[1:]:
            # Check if this segment overlaps or is adjacent to current group
            overlaps = False
            for group_seg in current_group:
                # Check mask overlap
                overlap = np.sum(seg['mask'] & group_seg['mask'])
                # Check adjacency (dilate and check)
                if overlap > 0:
                    overlaps = True
                    break

                # Check color similarity
                color_diff = np.sqrt(sum(
                    (a - b) ** 2
                    for a, b in zip(seg['avg_color'], group_seg['avg_color'])
                ))
                if color_diff < 50:  # Similar color
                    overlaps = True
                    break

            if overlaps:
                current_group.append(seg)
            else:
                groups.append(current_group)
                current_group = [seg]

        groups.append(current_group)

        return groups

    def _generate_layer_info(
        self,
        zone: DepthZone,
        y_top: float,
        y_bottom: float,
        luminosity: float,
        num_in_zone: int,
        index: int
    ) -> Tuple[str, str, str]:
        """Generate human-readable layer name, description, and semantic type."""

        zone_names = {
            DepthZone.FAR_BACKGROUND: ("Sky/Distant Hills", "sky", "Paint the furthest elements - sky and distant horizon."),
            DepthZone.BACKGROUND: ("Background", "background", "Paint the background elements behind the main subjects."),
            DepthZone.MIDGROUND: ("Middle Ground", "midground", "Paint the middle distance - ground plane, environment."),
            DepthZone.FOREGROUND: ("Foreground Subject", "subject", "Paint the main foreground subjects."),
            DepthZone.NEAR_FOREGROUND: ("Near Foreground", "subject", "Paint the closest elements, overlapping the scene."),
        }

        base_name, semantic_type, base_desc = zone_names.get(
            zone,
            ("Region", "region", "Paint this region.")
        )

        # Add luminosity descriptor
        if luminosity > 0.7:
            tone = "Light"
        elif luminosity < 0.3:
            tone = "Dark"
        else:
            tone = ""

        # Build name
        if num_in_zone > 1:
            name = f"{tone} {base_name} {index + 1}".strip()
        else:
            name = f"{tone} {base_name}".strip()

        # Build description based on position
        if zone == DepthZone.FAR_BACKGROUND:
            description = "Start with the sky and any distant elements. Keep edges soft and colors muted."
        elif zone == DepthZone.BACKGROUND:
            if luminosity < 0.4:
                description = "Paint the dark background elements (trees, shadows). This creates depth behind subjects."
            else:
                description = "Paint the background. Keep it less detailed than foreground."
        elif zone == DepthZone.MIDGROUND:
            description = "Paint the middle ground. This bridges background and foreground."
        elif zone == DepthZone.FOREGROUND:
            description = "Paint the main subject. Take your time - this is the focus of the painting."
        else:
            description = "Paint the nearest elements. These can overlap other areas."

        return name, description, semantic_type

    def _get_painting_guidance(
        self,
        zone: DepthZone,
        semantic_type: str,
        luminosity: float
    ) -> Tuple[str, List[str]]:
        """Get technique and tips for a layer."""

        if zone == DepthZone.FAR_BACKGROUND:
            technique = "wet_blend"
            tips = [
                "Use large brush for broad coverage",
                "Keep colors soft and slightly muted",
                "Blend edges - nothing sharp in the distance",
                "Work quickly while paint is wet"
            ]
        elif zone == DepthZone.BACKGROUND:
            technique = "blocking"
            tips = [
                "Establish the background shapes",
                "Don't add too much detail - it competes with subjects",
                "Keep values slightly lighter than you think (atmospheric perspective)",
                "Soft edges where background meets midground"
            ]
        elif zone == DepthZone.MIDGROUND:
            technique = "layering"
            tips = [
                "Build up the ground plane",
                "Medium level of detail",
                "Watch your edges - some soft, some defined",
                "This area connects background to foreground"
            ]
        elif zone in [DepthZone.FOREGROUND, DepthZone.NEAR_FOREGROUND]:
            technique = "layering"
            tips = [
                "This is your focal area - take your time",
                "Build up form with careful value transitions",
                "More detail and sharper edges than background",
                "Strongest contrast and most saturated colors here"
            ]
        else:
            technique = "blocking"
            tips = ["Apply even coverage", "Follow the natural boundaries"]

        # Add luminosity-specific tips
        if luminosity > 0.7:
            tips.append("Bright area - mix with plenty of white, apply thick for opacity")
        elif luminosity < 0.3:
            tips.append("Dark area - may need multiple thin coats for rich darks")

        return technique, tips


def create_depth_ordered_layers(
    image: np.ndarray,
    segments: List[Dict]
) -> List[SpatialLayer]:
    """
    Convenience function to create depth-ordered painting layers.

    Args:
        image: RGB image
        segments: List of segment dicts (from any segmentation method)

    Returns:
        List of SpatialLayers ordered back-to-front for painting
    """
    analyzer = DepthAnalyzer()
    return analyzer.analyze_depth(image, segments)
