#!/usr/bin/env python3
"""
Multi-Step Hybrid Britto Style Transfer
Creates clean segmentation like real Britto art, then applies styling

Process:
1. Segment image into clean, distinct regions
2. Apply Britto color palette to each region (flat colors)
3. Add VERY thick black outlines between regions
4. Add decorative patterns (hearts, flowers, swirls, dots) to regions
5. Refine for clean, stained-glass-like appearance
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from sklearn.cluster import KMeans
import random

class BrittoHybridStyler:
    """Multi-step Britto style transfer with clean segmentation"""

    # Britto's signature color palette (bright, vibrant)
    BRITTO_PALETTE = np.array([
        [255, 0, 0],      # Pure Red
        [0, 100, 255],    # Bright Blue
        [255, 220, 0],    # Yellow
        [0, 255, 100],    # Bright Green
        [255, 140, 0],    # Orange
        [255, 20, 147],   # Hot Pink
        [0, 255, 255],    # Cyan/Turquoise
        [160, 0, 255],    # Purple
        [255, 255, 255],  # White
        [255, 100, 180],  # Pink
        [100, 255, 0],    # Lime Green
        [100, 200, 255],  # Light Blue
        [255, 200, 0],    # Gold
    ], dtype=np.uint8)

    def __init__(self, n_segments=15, outline_thickness=12, pattern_density=0.5):
        """
        Initialize Britto Hybrid Styler

        Args:
            n_segments: Number of color regions (8-20 recommended)
            outline_thickness: Thickness of black outlines (10-20 for Britto style)
            pattern_density: How much pattern decoration to add (0-1)
        """
        self.n_segments = n_segments
        self.outline_thickness = outline_thickness
        self.pattern_density = pattern_density

    def segment_image(self, image):
        """
        Step 1: Segment image into clean, distinct regions using watershed + k-means
        """
        print(f"Step 1: Segmenting image into {self.n_segments} regions...")

        img_array = np.array(image)
        h, w, c = img_array.shape

        # Apply bilateral filter for edge-preserving smoothing
        smoothed = cv2.bilateralFilter(img_array, 9, 75, 75)

        # K-means clustering on colors
        pixels = smoothed.reshape(-1, 3).astype(np.float64) / 255.0  # Normalize to [0, 1]
        kmeans = KMeans(n_clusters=self.n_segments, random_state=42, n_init=10)
        labels = kmeans.fit_predict(pixels)
        segments = labels.reshape(h, w)

        # Apply morphological operations to clean up segments
        kernel = np.ones((5, 5), np.uint8)
        segments_clean = cv2.morphologyEx(segments.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

        return segments_clean.astype(np.int32), img_array

    def apply_flat_colors(self, segments, img_array):
        """
        Step 2: Apply flat Britto colors to each segment
        """
        print("Step 2: Applying flat Britto color palette...")

        n_regions = segments.max() + 1
        result = np.zeros_like(img_array)

        # For each segment, choose a Britto color
        for i in range(n_regions):
            mask = segments == i

            # Get average color of this segment
            segment_pixels = img_array[mask]
            if len(segment_pixels) == 0:
                continue

            avg_color = segment_pixels.mean(axis=0)

            # Find closest Britto palette color
            distances = np.linalg.norm(self.BRITTO_PALETTE - avg_color, axis=1)
            closest_idx = np.argmin(distances)
            britto_color = self.BRITTO_PALETTE[closest_idx]

            # Apply flat color to entire segment
            result[mask] = britto_color

        return result

    def add_thick_outlines(self, segments, colored_image):
        """
        Step 3: Add VERY thick black outlines between segments (Britto signature)
        """
        print(f"Step 3: Adding thick black outlines ({self.outline_thickness}px)...")

        # Find boundaries between segments
        h, w = segments.shape
        boundaries = np.zeros((h, w), dtype=np.uint8)

        # Detect edges between different segments
        for i in range(1, h):
            for j in range(1, w):
                # Check if neighbors have different segment IDs
                if (segments[i, j] != segments[i-1, j] or
                    segments[i, j] != segments[i, j-1]):
                    boundaries[i, j] = 255

        # Dilate boundaries to make them THICK (Britto style)
        kernel = np.ones((self.outline_thickness, self.outline_thickness), np.uint8)
        thick_boundaries = cv2.dilate(boundaries, kernel, iterations=1)

        # Apply black outlines to image
        result = colored_image.copy()
        result[thick_boundaries > 0] = [0, 0, 0]  # Black

        return Image.fromarray(result)

    def add_decorative_patterns(self, image, segments):
        """
        Step 4: Add Britto decorative patterns (hearts, dots, swirls, flowers)
        """
        print("Step 4: Adding decorative patterns...")

        width, height = image.size
        overlay = Image.new('RGBA', (width, height), (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)

        # Pattern size based on image size
        pattern_size = max(4, min(width, height) // 60)

        n_regions = segments.max() + 1

        # Add patterns to random regions
        n_patterned_regions = int(n_regions * self.pattern_density)
        patterned_regions = random.sample(range(n_regions), min(n_patterned_regions, n_regions))

        for region_id in patterned_regions:
            mask = (segments == region_id)
            if not mask.any():
                continue

            # Get region bounds
            ys, xs = np.where(mask)
            if len(ys) < 100:  # Skip tiny regions
                continue

            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()
            region_w = x_max - x_min
            region_h = y_max - y_min

            # Choose random pattern type
            pattern_type = random.choice(['dots', 'hearts', 'flowers', 'swirls', 'stars'])

            # Pattern color (white or contrasting)
            pattern_color = (255, 255, 255, 200)  # Semi-transparent white

            if pattern_type == 'dots':
                self._draw_dots(draw, x_min, y_min, region_w, region_h, pattern_size, pattern_color)
            elif pattern_type == 'hearts':
                self._draw_hearts(draw, x_min, y_min, region_w, region_h, pattern_size, pattern_color)
            elif pattern_type == 'flowers':
                self._draw_flowers(draw, x_min, y_min, region_w, region_h, pattern_size, pattern_color)
            elif pattern_type == 'swirls':
                self._draw_swirls(draw, x_min, y_min, region_w, region_h, pattern_size, pattern_color)
            elif pattern_type == 'stars':
                self._draw_stars(draw, x_min, y_min, region_w, region_h, pattern_size, pattern_color)

        # Composite patterns onto image
        image_rgba = image.convert('RGBA')
        result = Image.alpha_composite(image_rgba, overlay)
        return result.convert('RGB')

    def _draw_dots(self, draw, x, y, w, h, size, color):
        """Draw dot pattern"""
        spacing = size * 8
        for dy in range(0, h, spacing):
            for dx in range(0, w, spacing):
                px, py = x + dx, y + dy
                draw.ellipse([px, py, px + size*3, py + size*3], fill=color)

    def _draw_hearts(self, draw, x, y, w, h, size, color):
        """Draw heart pattern"""
        spacing = size * 10
        for dy in range(0, h, spacing):
            for dx in range(0, w, spacing):
                px, py = x + dx, y + dy
                s = size * 2
                # Simple heart shape
                draw.ellipse([px, py, px + s, py + s], fill=color)
                draw.ellipse([px + s, py, px + s*2, py + s], fill=color)
                draw.polygon([(px, py + s), (px + s*2, py + s), (px + s, py + s*2)], fill=color)

    def _draw_flowers(self, draw, x, y, w, h, size, color):
        """Draw flower pattern"""
        spacing = size * 12
        for dy in range(0, h, spacing):
            for dx in range(0, w, spacing):
                px, py = x + dx, y + dy
                s = size * 2
                # Center
                draw.ellipse([px + s, py + s, px + s*2, py + s*2], fill=color)
                # Petals
                for angle in [0, 60, 120, 180, 240, 300]:
                    offset_x = int(s * 1.5 * np.cos(np.radians(angle)))
                    offset_y = int(s * 1.5 * np.sin(np.radians(angle)))
                    draw.ellipse([px + s + offset_x, py + s + offset_y,
                                px + s*2 + offset_x, py + s*2 + offset_y], fill=color)

    def _draw_swirls(self, draw, x, y, w, h, size, color):
        """Draw swirl/spiral pattern"""
        spacing = size * 12
        for dy in range(0, h, spacing):
            for dx in range(0, w, spacing):
                px, py = x + dx, y + dy
                # Draw spiral
                for r in range(size, size * 5, size):
                    draw.arc([px - r, py - r, px + r, py + r], 0, 270, fill=color, width=2)

    def _draw_stars(self, draw, x, y, w, h, size, color):
        """Draw star pattern"""
        spacing = size * 12
        for dy in range(0, h, spacing):
            for dx in range(0, w, spacing):
                px, py = x + dx + size*4, y + dy + size*4
                # 5-pointed star
                points = []
                for i in range(10):
                    angle = i * 36 - 90
                    r = size*4 if i % 2 == 0 else size*2
                    points.append((
                        px + r * np.cos(np.radians(angle)),
                        py + r * np.sin(np.radians(angle))
                    ))
                draw.polygon(points, fill=color)

    def apply_style(self, input_path, output_path):
        """
        Apply complete Britto hybrid style transformation
        """
        print(f"\nBritto Hybrid Style Transfer")
        print("="*60)

        # Load image
        print(f"Loading image: {input_path}")
        image = Image.open(input_path).convert('RGB')

        # Step 1: Segment into clean regions
        segments, img_array = self.segment_image(image)

        # Step 2: Apply flat Britto colors
        colored = self.apply_flat_colors(segments, img_array)

        # Step 3: Add thick black outlines
        outlined = self.add_thick_outlines(segments, colored)

        # Step 4: Add decorative patterns
        final = self.add_decorative_patterns(outlined, segments)

        # Save result
        print(f"\nSaving result to: {output_path}")
        final.save(output_path, quality=95)

        print("="*60)
        print("Complete!")

        return final


if __name__ == '__main__':
    # Create styler with Britto-appropriate settings
    styler = BrittoHybridStyler(
        n_segments=10,           # Good balance of regions
        outline_thickness=16,     # VERY thick outlines (Britto signature)
        pattern_density=0.25      # Only 25% of regions get patterns (mostly solid colors)
    )

    # Apply style
    styler.apply_style(
        'input/wilburderby1/input.png',
        'output/wilburderby1_britto_hybrid.png'
    )
