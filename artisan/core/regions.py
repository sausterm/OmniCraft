"""
Region Detection - Identify distinct canvas areas.

Provides multiple strategies for breaking down an image into regions:
1. Grid-based: Simple 3x3 or NxM grid subdivision
2. Contour-based: Find actual color blobs using connected components
3. Semantic: Identify meaningful regions (sky, water, foreground, etc.)
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class RegionDetectionConfig:
    """Configuration for region detection."""
    min_region_pixels: int = 100        # Minimum pixels for a region
    min_coverage_percent: float = 0.1   # Minimum % of canvas
    grid_size: Tuple[int, int] = (3, 3) # Grid dimensions (rows, cols)
    merge_small_regions: bool = True     # Merge tiny adjacent regions
    merge_threshold_percent: float = 2.0 # Merge if total < this %
    use_contours: bool = True            # Use contour detection
    min_contour_area: int = 500          # Minimum contour size


class RegionDetector:
    """Detects and describes regions within an image or mask."""

    def __init__(self, height: int, width: int, config: Optional[RegionDetectionConfig] = None):
        """
        Initialize region detector.

        Args:
            height: Image height in pixels
            width: Image width in pixels
            config: Detection configuration
        """
        self.height = height
        self.width = width
        self.total_pixels = height * width
        self.config = config or RegionDetectionConfig()

    def find_regions_grid(self, mask: np.ndarray) -> List[dict]:
        """
        Find regions using a simple grid subdivision.

        Fast and predictable, but doesn't follow image structure.

        Args:
            mask: Boolean mask of where to look for regions

        Returns:
            List of region dicts with keys: name, bounds, mask, coverage, centroid
        """
        regions = []
        rows, cols = self.config.grid_size

        grid_names = self._generate_grid_names(rows, cols)

        for row in range(rows):
            for col in range(cols):
                y1 = row / rows
                y2 = (row + 1) / rows
                x1 = col / cols
                x2 = (col + 1) / cols

                # Get pixel bounds
                py1, py2 = int(y1 * self.height), int(y2 * self.height)
                px1, px2 = int(x1 * self.width), int(x2 * self.width)

                # Extract region from mask
                region_mask = mask[py1:py2, px1:px2]
                pixel_count = np.sum(region_mask)

                if pixel_count >= self.config.min_region_pixels:
                    coverage = pixel_count / self.total_pixels * 100

                    if coverage >= self.config.min_coverage_percent:
                        # Calculate centroid
                        y_coords, x_coords = np.where(region_mask)
                        if len(y_coords) > 0:
                            cy = (np.mean(y_coords) + py1) / self.height
                            cx = (np.mean(x_coords) + px1) / self.width
                        else:
                            cy = (y1 + y2) / 2
                            cx = (x1 + x2) / 2

                        regions.append({
                            'name': grid_names[row][col],
                            'bounds': (y1, y2, x1, x2),
                            'mask': region_mask,
                            'coverage_percent': coverage,
                            'centroid': (cy, cx),
                            'pixel_count': pixel_count,
                        })

        # Optionally merge small regions
        if self.config.merge_small_regions:
            regions = self._merge_small_regions(regions)

        return regions

    def find_regions_contour(self, mask: np.ndarray) -> List[dict]:
        """
        Find regions using contour detection.

        Follows actual image structure - finds discrete color blobs.
        More accurate but slightly slower than grid-based.

        Args:
            mask: Boolean mask of where to look for regions

        Returns:
            List of region dicts with keys: name, bounds, mask, coverage, centroid
        """
        regions = []

        # Convert mask to uint8 for OpenCV
        mask_uint8 = (mask.astype(np.uint8) * 255)

        # Find contours
        contours, _ = cv2.findContours(
            mask_uint8,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # Process each contour
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)

            if area < self.config.min_contour_area:
                continue

            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)

            # Create mask for this specific contour
            contour_mask = np.zeros((self.height, self.width), dtype=np.uint8)
            cv2.drawContours(contour_mask, [contour], -1, 1, -1)

            # Extract region mask
            region_mask = contour_mask[y:y+h, x:x+w]
            pixel_count = int(area)
            coverage = pixel_count / self.total_pixels * 100

            if coverage < self.config.min_coverage_percent:
                continue

            # Calculate centroid
            M = cv2.moments(contour)
            if M['m00'] > 0:
                cx = M['m10'] / M['m00'] / self.width
                cy = M['m01'] / M['m00'] / self.height
            else:
                cy = (y + h/2) / self.height
                cx = (x + w/2) / self.width

            # Convert to fractional bounds
            bounds = (
                y / self.height,
                (y + h) / self.height,
                x / self.width,
                (x + w) / self.width
            )

            # Generate descriptive name
            name = self._describe_bounds(bounds)

            regions.append({
                'name': f"{name} (blob {i+1})" if len(contours) > 3 else name,
                'bounds': bounds,
                'mask': region_mask.astype(bool),
                'coverage_percent': coverage,
                'centroid': (cy, cx),
                'pixel_count': pixel_count,
                'contour': contour,
            })

        # Sort by coverage (largest first)
        regions.sort(key=lambda r: -r['coverage_percent'])

        # Remove blob numbering if we only have 1-3 regions
        if len(regions) <= 3:
            for region in regions:
                region['name'] = region['name'].split(' (blob')[0]

        return regions

    def find_regions(self, mask: np.ndarray) -> List[dict]:
        """
        Find regions using the configured detection method.

        Args:
            mask: Boolean mask of where to look for regions

        Returns:
            List of region dicts
        """
        if self.config.use_contours:
            return self.find_regions_contour(mask)
        else:
            return self.find_regions_grid(mask)

    def _generate_grid_names(self, rows: int, cols: int) -> List[List[str]]:
        """Generate human-readable names for grid cells."""
        if rows == 3 and cols == 3:
            # Standard 3x3 grid
            return [
                ['upper-left', 'upper-center', 'upper-right'],
                ['middle-left', 'center', 'middle-right'],
                ['lower-left', 'lower-center', 'lower-right']
            ]
        else:
            # Generic grid
            vert_names = ['top', 'upper', 'middle', 'lower', 'bottom'][:rows]
            horiz_names = ['left', 'center-left', 'center', 'center-right', 'right'][:cols]

            names = []
            for i, v in enumerate(vert_names):
                row = []
                for j, h in enumerate(horiz_names):
                    if i == rows // 2 and j == cols // 2:
                        row.append('center')
                    elif j == cols // 2:
                        row.append(f'{v}')
                    elif i == rows // 2:
                        row.append(f'{h}')
                    else:
                        row.append(f'{v}-{h}')
                names.append(row)

            return names

    def _describe_bounds(self, bounds: Tuple[float, float, float, float]) -> str:
        """
        Convert fractional bounds to human-readable description.

        Args:
            bounds: (y1, y2, x1, x2) as fractions 0-1

        Returns:
            Descriptive string like "upper-left", "center", etc.
        """
        y1, y2, x1, x2 = bounds

        # Determine vertical position
        y_center = (y1 + y2) / 2
        if y_center < 0.33:
            vert = "upper"
        elif y_center < 0.67:
            vert = "middle"
        else:
            vert = "lower"

        # Determine horizontal position
        x_center = (x1 + x2) / 2
        if x_center < 0.33:
            horiz = "left"
        elif x_center < 0.67:
            horiz = "center"
        else:
            horiz = "right"

        # Combine
        if vert == "middle" and horiz == "center":
            return "center"
        elif horiz == "center":
            return f"{vert} area"
        elif vert == "middle":
            return f"{horiz} side"
        else:
            return f"{vert}-{horiz}"

    def _merge_small_regions(self, regions: List[dict]) -> List[dict]:
        """
        Merge very small adjacent regions to avoid too many substeps.

        Args:
            regions: List of region dicts

        Returns:
            Merged region list
        """
        if len(regions) <= 1:
            return regions

        total_coverage = sum(r['coverage_percent'] for r in regions)

        # If total coverage is very small, merge all into one
        if total_coverage < self.config.merge_threshold_percent:
            merged_bounds = (
                min(r['bounds'][0] for r in regions),
                max(r['bounds'][1] for r in regions),
                min(r['bounds'][2] for r in regions),
                max(r['bounds'][3] for r in regions)
            )

            return [{
                'name': self._describe_bounds(merged_bounds),
                'bounds': merged_bounds,
                'mask': None,  # Don't merge masks
                'coverage_percent': total_coverage,
                'centroid': (
                    sum(r['centroid'][0] * r['coverage_percent'] for r in regions) / total_coverage,
                    sum(r['centroid'][1] * r['coverage_percent'] for r in regions) / total_coverage
                ),
                'pixel_count': sum(r['pixel_count'] for r in regions),
            }]

        return regions

    def add_semantic_labels(
        self,
        regions: List[dict],
        luminosity_map: np.ndarray,
        vertical_gradient: Optional[np.ndarray] = None
    ) -> List[dict]:
        """
        Add semantic labels to regions based on image analysis.

        Attempts to identify meaningful regions like "sky", "water", "foreground".

        Args:
            regions: List of region dicts
            luminosity_map: Luminosity/brightness map of image
            vertical_gradient: Optional vertical position heatmap

        Returns:
            Regions with added 'semantic_label' field
        """
        for region in regions:
            bounds = region['bounds']
            y1, y2, x1, x2 = bounds

            # Extract luminosity in this region
            py1, py2 = int(y1 * self.height), int(y2 * self.height)
            px1, px2 = int(x1 * self.width), int(x2 * self.width)

            region_lum = luminosity_map[py1:py2, px1:px2]
            avg_lum = np.mean(region_lum)

            # Vertical position
            y_center = (y1 + y2) / 2

            # Basic semantic labeling heuristics
            label = ""

            if y_center < 0.4 and avg_lum > 0.5:
                label = "sky"
            elif y_center > 0.6 and avg_lum < 0.3:
                label = "foreground"
            elif 0.3 <= y_center <= 0.6:
                if avg_lum > 0.6:
                    label = "bright area"
                else:
                    label = "middle ground"
            elif y_center > 0.5 and avg_lum > 0.4:
                label = "water/horizon"

            region['semantic_label'] = label

        return regions
