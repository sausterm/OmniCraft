"""
Technique Analyzer - Intelligent painting technique detection and instruction generation.

Analyzes images to determine optimal painting techniques beyond simple fill:
- Glazing (transparent layers over dark backgrounds)
- Wet-on-wet blending
- Dry brushing
- Layered buildup for glow effects
- Standard paint-by-numbers fill
"""

import numpy as np
import cv2
from scipy import ndimage
from sklearn.cluster import KMeans
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Tuple, Optional
import json


class Technique(Enum):
    """Painting techniques that can be recommended."""
    STANDARD_FILL = "standard_fill"
    GLAZING = "glazing"
    WET_ON_WET = "wet_on_wet"
    DRY_BRUSH = "dry_brush"
    LAYERED_BUILDUP = "layered_buildup"
    GRADIENT_BLEND = "gradient_blend"


@dataclass
class TechniqueZone:
    """A region of the image with a recommended technique."""
    technique: Technique
    region_mask: np.ndarray
    base_color: Tuple[int, int, int]
    overlay_colors: List[Tuple[int, int, int]]
    layer_order: int
    brush_direction: Optional[str]  # "horizontal", "vertical", "radial", "circular", None
    instructions: str
    difficulty: int  # 1-5


@dataclass
class ImageAnalysis:
    """Complete analysis of an image for technique recommendations."""
    has_dark_background: bool
    dark_background_percentage: float
    has_glow_effects: bool
    glow_centers: List[Tuple[int, int]]
    gradient_intensity: float  # 0-1, how much gradients dominate
    edge_softness: float  # 0-1, how soft edges are overall
    dominant_pattern: str  # "aurora", "sunset", "fire", "standard", etc.
    recommended_base: Tuple[int, int, int]
    technique_zones: List[TechniqueZone]
    layer_count: int
    overall_difficulty: int


class TechniqueAnalyzer:
    """Analyzes images to recommend painting techniques."""

    def __init__(self, image_path: str):
        """
        Initialize analyzer with an image.

        Args:
            image_path: Path to the image file
        """
        self.image_path = image_path
        self.original = cv2.imread(image_path)
        self.original_rgb = cv2.cvtColor(self.original, cv2.COLOR_BGR2RGB)
        self.height, self.width = self.original.shape[:2]

        # Analysis caches
        self._luminosity = None
        self._gradients = None
        self._edges_soft = None
        self._edges_hard = None

    @property
    def luminosity(self) -> np.ndarray:
        """Get luminosity channel (cached)."""
        if self._luminosity is None:
            lab = cv2.cvtColor(self.original, cv2.COLOR_BGR2LAB)
            self._luminosity = lab[:, :, 0].astype(np.float32) / 255.0
        return self._luminosity

    @property
    def gradients(self) -> np.ndarray:
        """Get gradient magnitude (cached)."""
        if self._gradients is None:
            gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            self._gradients = np.sqrt(sobelx**2 + sobely**2)
            self._gradients = self._gradients / self._gradients.max()  # Normalize
        return self._gradients

    def analyze_dark_background(self) -> Tuple[bool, float, np.ndarray]:
        """
        Detect if image has a dark background suitable for glazing.

        Returns:
            (has_dark_bg, percentage, dark_mask)
        """
        # Dark pixels: luminosity < 0.2
        dark_mask = self.luminosity < 0.2
        dark_percentage = np.mean(dark_mask)

        # Check if dark areas are connected and form background
        # Use morphological operations to find large dark regions
        kernel = np.ones((20, 20), np.uint8)
        dark_regions = cv2.morphologyEx(dark_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)

        # Check edges - backgrounds typically touch image borders
        border_mask = np.zeros_like(dark_mask)
        border_mask[0:10, :] = True
        border_mask[-10:, :] = True
        border_mask[:, 0:10] = True
        border_mask[:, -10:] = True

        dark_touches_border = np.mean(dark_regions[border_mask]) > 0.3

        has_dark_bg = dark_percentage > 0.25 and dark_touches_border

        return has_dark_bg, dark_percentage, dark_mask

    def analyze_glow_effects(self) -> Tuple[bool, List[Tuple[int, int]], np.ndarray]:
        """
        Detect glow effects (bright cores with falloff).

        Returns:
            (has_glow, glow_centers, glow_map)
        """
        # Find bright regions
        bright_mask = self.luminosity > 0.7

        # Find local maxima in luminosity
        local_max = ndimage.maximum_filter(self.luminosity, size=50)
        is_peak = (self.luminosity == local_max) & bright_mask

        # Get peak coordinates
        peak_coords = np.where(is_peak)

        # Cluster peaks to find glow centers
        if len(peak_coords[0]) > 0:
            points = np.column_stack((peak_coords[1], peak_coords[0]))  # x, y

            if len(points) > 5:
                # Cluster to reduce noise
                n_clusters = min(10, len(points) // 10 + 1)
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=3)
                kmeans.fit(points)
                centers = kmeans.cluster_centers_.astype(int)
            else:
                centers = points

            glow_centers = [(int(c[0]), int(c[1])) for c in centers]
        else:
            glow_centers = []

        # Create glow map - distance from bright regions with falloff
        dist_from_bright = ndimage.distance_transform_edt(~bright_mask)
        max_dist = max(self.height, self.width) * 0.2
        glow_map = np.clip(1.0 - dist_from_bright / max_dist, 0, 1)
        glow_map = glow_map * self.luminosity  # Weight by actual brightness

        has_glow = len(glow_centers) > 0 and np.mean(glow_map > 0.3) > 0.05

        return has_glow, glow_centers, glow_map

    def analyze_gradients(self) -> Tuple[float, np.ndarray]:
        """
        Analyze gradient presence and softness.

        Returns:
            (gradient_intensity, gradient_direction_map)
        """
        gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Compute gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)

        magnitude = np.sqrt(sobelx**2 + sobely**2)
        direction = np.arctan2(sobely, sobelx)

        # Gradient intensity: how much of image has gradients vs flat regions
        # Normalize magnitude
        mag_norm = magnitude / (magnitude.max() + 1e-6)

        # Medium gradients (not edges, not flat) indicate blendable regions
        soft_gradient_mask = (mag_norm > 0.05) & (mag_norm < 0.4)
        gradient_intensity = np.mean(soft_gradient_mask)

        return gradient_intensity, direction

    def analyze_edge_softness(self) -> float:
        """
        Measure overall edge softness (soft = good for blending).

        Returns:
            softness score 0-1 (1 = very soft edges)
        """
        gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)

        # Detect edges at different sensitivities
        edges_sensitive = cv2.Canny(gray, 30, 80)
        edges_hard = cv2.Canny(gray, 80, 200)

        # Soft edges = edges detected by sensitive but not hard threshold
        soft_edge_count = np.sum(edges_sensitive > 0) - np.sum(edges_hard > 0)
        hard_edge_count = np.sum(edges_hard > 0)

        total_edges = soft_edge_count + hard_edge_count
        if total_edges == 0:
            return 1.0

        softness = soft_edge_count / total_edges
        return softness

    def detect_pattern_type(self) -> str:
        """
        Classify the dominant pattern in the image.

        Returns:
            Pattern name: "aurora", "sunset", "fire", "water", "sky", "standard"
        """
        # Analyze color distribution
        hsv = cv2.cvtColor(self.original, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        has_dark_bg, dark_pct, _ = self.analyze_dark_background()
        has_glow, glow_centers, _ = self.analyze_glow_effects()
        gradient_intensity, _ = self.analyze_gradients()

        # Color analysis
        # Green/cyan hues (aurora)
        green_cyan_mask = ((h > 35) & (h < 100)) & (s > 50)
        green_cyan_pct = np.mean(green_cyan_mask)

        # Warm hues (sunset/fire)
        warm_mask = ((h < 30) | (h > 160)) & (s > 50)
        warm_pct = np.mean(warm_mask)

        # Blue hues (sky/water)
        blue_mask = ((h > 100) & (h < 130)) & (s > 30)
        blue_pct = np.mean(blue_mask)

        # Pattern detection logic
        if has_dark_bg and has_glow and green_cyan_pct > 0.1:
            return "aurora"
        elif warm_pct > 0.2 and gradient_intensity > 0.15:
            if has_dark_bg:
                return "fire"
            else:
                return "sunset"
        elif blue_pct > 0.3 and gradient_intensity > 0.1:
            # Check if horizontal gradients dominate (sky/water)
            return "sky"
        elif gradient_intensity > 0.2:
            return "gradient"
        else:
            return "standard"

    def generate_technique_zones(self, n_colors: int = 15) -> List[TechniqueZone]:
        """
        Generate technique zones based on analysis.

        Args:
            n_colors: Number of colors to use in the palette

        Returns:
            List of TechniqueZone objects
        """
        zones = []

        has_dark_bg, dark_pct, dark_mask = self.analyze_dark_background()
        has_glow, glow_centers, glow_map = self.analyze_glow_effects()
        gradient_intensity, grad_direction = self.analyze_gradients()
        edge_softness = self.analyze_edge_softness()
        pattern = self.detect_pattern_type()

        # Zone 1: Dark background (if present)
        if has_dark_bg and dark_pct > 0.2:
            # Find average dark color
            dark_pixels = self.original_rgb[dark_mask]
            if len(dark_pixels) > 0:
                base_color = tuple(np.mean(dark_pixels, axis=0).astype(int))
            else:
                base_color = (0, 0, 0)

            zones.append(TechniqueZone(
                technique=Technique.STANDARD_FILL,
                region_mask=dark_mask,
                base_color=base_color,
                overlay_colors=[],
                layer_order=1,
                brush_direction=None,
                instructions="Paint the entire canvas with the base dark color first. This creates the foundation for layering bright colors on top.",
                difficulty=1
            ))

        # Zone 2: Glow/bright regions (glazing candidates)
        if has_glow:
            bright_mask = self.luminosity > 0.5
            bright_pixels = self.original_rgb[bright_mask]

            if len(bright_pixels) > 0:
                # Cluster bright colors
                n_bright = min(5, max(2, n_colors // 4))
                if len(bright_pixels) > n_bright:
                    kmeans = KMeans(n_clusters=n_bright, random_state=42, n_init=3)
                    bright_pixels_sample = bright_pixels[::max(1, len(bright_pixels)//1000)]
                    kmeans.fit(bright_pixels_sample)
                    overlay_colors = [tuple(c.astype(int)) for c in kmeans.cluster_centers_]
                else:
                    overlay_colors = [tuple(np.mean(bright_pixels, axis=0).astype(int))]

                # Sort by luminosity (darkest first for layering)
                overlay_colors.sort(key=lambda c: sum(c))

                technique = Technique.GLAZING if has_dark_bg else Technique.LAYERED_BUILDUP

                zones.append(TechniqueZone(
                    technique=technique,
                    region_mask=bright_mask,
                    base_color=(0, 0, 0) if has_dark_bg else overlay_colors[0],
                    overlay_colors=overlay_colors,
                    layer_order=2,
                    brush_direction=self._detect_brush_direction(bright_mask, glow_centers),
                    instructions=self._generate_glow_instructions(technique, overlay_colors, pattern),
                    difficulty=3
                ))

        # Zone 3: Gradient/blend regions
        soft_gradient_mask = (self.gradients > 0.05) & (self.gradients < 0.4)
        if np.mean(soft_gradient_mask) > 0.1 and edge_softness > 0.5:
            gradient_pixels = self.original_rgb[soft_gradient_mask]
            if len(gradient_pixels) > 0:
                avg_color = tuple(np.mean(gradient_pixels, axis=0).astype(int))

                zones.append(TechniqueZone(
                    technique=Technique.WET_ON_WET,
                    region_mask=soft_gradient_mask,
                    base_color=avg_color,
                    overlay_colors=[],
                    layer_order=3,
                    brush_direction=self._dominant_direction(grad_direction, soft_gradient_mask),
                    instructions="Apply colors while previous layer is still wet. Blend edges by gently brushing where colors meet.",
                    difficulty=4
                ))

        return zones

    def _detect_brush_direction(self, mask: np.ndarray, centers: List[Tuple[int, int]]) -> str:
        """Detect recommended brush direction based on glow centers."""
        if not centers:
            return "vertical"

        # If centers are horizontally aligned -> vertical strokes
        # If centers are vertically aligned -> horizontal strokes
        if len(centers) == 1:
            # Single center -> radial
            cy = centers[0][1]
            if cy < self.height * 0.3:
                return "vertical_down"  # Light from top
            elif cy > self.height * 0.7:
                return "vertical_up"  # Light from bottom
            else:
                return "radial"

        xs = [c[0] for c in centers]
        ys = [c[1] for c in centers]

        x_spread = max(xs) - min(xs)
        y_spread = max(ys) - min(ys)

        if x_spread > y_spread * 1.5:
            return "vertical"  # Horizontal band of lights -> paint vertically
        elif y_spread > x_spread * 1.5:
            return "horizontal"
        else:
            return "radial"

    def _dominant_direction(self, direction_map: np.ndarray, mask: np.ndarray) -> str:
        """Find dominant gradient direction in masked region."""
        masked_dirs = direction_map[mask]
        if len(masked_dirs) == 0:
            return None

        # Convert to degrees and bin
        degrees = np.degrees(masked_dirs) % 180
        hist, _ = np.histogram(degrees, bins=4, range=(0, 180))

        dominant_bin = np.argmax(hist)
        if dominant_bin in [0, 3]:  # Near horizontal
            return "horizontal"
        else:
            return "vertical"

    def _generate_glow_instructions(self, technique: Technique, colors: List, pattern: str) -> str:
        """Generate specific instructions for glow effects."""
        if pattern == "aurora":
            return f"""AURORA TECHNIQUE (Glazing):
1. Ensure dark background is completely dry
2. Mix colors with glazing medium (or water for acrylics) to make transparent
3. Apply {colors[0]} in sweeping vertical strokes from top
4. While slightly wet, add {colors[1] if len(colors) > 1 else colors[0]} in flowing curves
5. Build up intensity with multiple thin layers, letting each dry
6. Add brightest highlights ({colors[-1]}) last with minimal medium
7. Use a soft dry brush to blend edges"""
        elif pattern == "sunset":
            return f"""SUNSET TECHNIQUE (Gradient Blend):
1. Work quickly with wet paint
2. Apply colors in horizontal bands
3. Blend where colors meet using a clean, dry brush
4. Add brightest colors near the light source
5. Build depth with thin glazes once dry"""
        else:
            return f"""LAYERED GLOW TECHNIQUE:
1. Start with darkest colors in the overlay palette
2. Apply in thin, transparent layers
3. Let each layer dry before adding the next
4. Build intensity gradually toward glow centers
5. Brightest colors ({colors[-1]}) applied last, sparingly"""

    def full_analysis(self, n_colors: int = 15) -> ImageAnalysis:
        """
        Perform complete image analysis.

        Args:
            n_colors: Number of colors for technique zone generation

        Returns:
            Complete ImageAnalysis object
        """
        has_dark_bg, dark_pct, dark_mask = self.analyze_dark_background()
        has_glow, glow_centers, glow_map = self.analyze_glow_effects()
        gradient_intensity, _ = self.analyze_gradients()
        edge_softness = self.analyze_edge_softness()
        pattern = self.detect_pattern_type()
        technique_zones = self.generate_technique_zones(n_colors)

        # Determine recommended base color
        if has_dark_bg:
            dark_pixels = self.original_rgb[dark_mask]
            if len(dark_pixels) > 0:
                base_color = tuple(np.median(dark_pixels, axis=0).astype(int))
            else:
                base_color = (0, 0, 0)
        else:
            base_color = tuple(np.median(self.original_rgb.reshape(-1, 3), axis=0).astype(int))

        # Calculate difficulty
        difficulty = 1
        if has_glow:
            difficulty += 1
        if gradient_intensity > 0.15:
            difficulty += 1
        if edge_softness > 0.6:
            difficulty += 1
        if pattern in ["aurora", "fire"]:
            difficulty += 1
        difficulty = min(5, difficulty)

        return ImageAnalysis(
            has_dark_background=has_dark_bg,
            dark_background_percentage=dark_pct,
            has_glow_effects=has_glow,
            glow_centers=glow_centers,
            gradient_intensity=gradient_intensity,
            edge_softness=edge_softness,
            dominant_pattern=pattern,
            recommended_base=base_color,
            technique_zones=technique_zones,
            layer_count=len(technique_zones) + 1,
            overall_difficulty=difficulty
        )

    def generate_layer_instructions(self, analysis: ImageAnalysis = None) -> List[Dict]:
        """
        Generate step-by-step layer instructions.

        Returns:
            List of instruction dictionaries with layer info
        """
        if analysis is None:
            analysis = self.full_analysis()

        instructions = []

        # Layer 0: Preparation
        instructions.append({
            "layer": 0,
            "name": "Preparation",
            "technique": "setup",
            "instructions": f"""MATERIALS NEEDED:
- Canvas or heavy paper
- Paints (see color guide)
- Brushes: flat brush for backgrounds, round brush for details
- {'Glazing medium or water' if analysis.has_dark_background else 'Palette for mixing'}
- Clean water and paper towels

BEFORE YOU START:
- Pattern detected: {analysis.dominant_pattern.upper()}
- Difficulty: {'*' * analysis.overall_difficulty} ({analysis.overall_difficulty}/5)
- This painting has {analysis.layer_count} main layers
{'- Dark background detected - will use GLAZING technique' if analysis.has_dark_background else ''}
{'- Glow effects detected - will build up brightness in layers' if analysis.has_glow_effects else ''}
""",
            "colors": [],
            "dry_time": "N/A"
        })

        # Generate layer instructions from technique zones
        for zone in sorted(analysis.technique_zones, key=lambda z: z.layer_order):
            layer_instruction = {
                "layer": zone.layer_order,
                "name": f"Layer {zone.layer_order}: {zone.technique.value.replace('_', ' ').title()}",
                "technique": zone.technique.value,
                "instructions": zone.instructions,
                "colors": [zone.base_color] + zone.overlay_colors,
                "brush_direction": zone.brush_direction,
                "dry_time": "30 min - 1 hour" if zone.technique == Technique.GLAZING else "Until tacky" if zone.technique == Technique.WET_ON_WET else "N/A",
                "difficulty": zone.difficulty
            }
            instructions.append(layer_instruction)

        # Final layer: Details
        instructions.append({
            "layer": len(instructions),
            "name": "Final Details",
            "technique": "finishing",
            "instructions": """FINISHING TOUCHES:
1. Step back and assess the overall composition
2. Add any final highlights with pure white or brightest colors
3. Deepen shadows if needed
4. Sign your work!

TIPS:
- Less is more with final details
- Let the layers underneath show through
- Don't overwork - know when to stop""",
            "colors": [],
            "dry_time": "Complete dry: 24 hours"
        })

        return instructions

    def export_analysis(self, output_path: str, n_colors: int = 15):
        """Export analysis to JSON file."""
        analysis = self.full_analysis(n_colors)
        instructions = self.generate_layer_instructions(analysis)

        export_data = {
            "image": self.image_path,
            "analysis": {
                "pattern": analysis.dominant_pattern,
                "has_dark_background": analysis.has_dark_background,
                "dark_percentage": round(analysis.dark_background_percentage, 2),
                "has_glow_effects": analysis.has_glow_effects,
                "glow_centers": analysis.glow_centers,
                "gradient_intensity": round(analysis.gradient_intensity, 2),
                "edge_softness": round(analysis.edge_softness, 2),
                "recommended_base_color": analysis.recommended_base,
                "difficulty": analysis.overall_difficulty,
                "layer_count": analysis.layer_count
            },
            "layer_instructions": instructions
        }

        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)

        print(f"Analysis exported to {output_path}")
        return export_data


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python technique_analyzer.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    analyzer = TechniqueAnalyzer(image_path)
    analysis = analyzer.full_analysis()

    print("=" * 60)
    print("TECHNIQUE ANALYSIS RESULTS")
    print("=" * 60)
    print(f"Pattern detected: {analysis.dominant_pattern.upper()}")
    print(f"Difficulty: {'*' * analysis.overall_difficulty} ({analysis.overall_difficulty}/5)")
    print(f"Dark background: {analysis.has_dark_background} ({analysis.dark_background_percentage:.1%})")
    print(f"Glow effects: {analysis.has_glow_effects}")
    print(f"Gradient intensity: {analysis.gradient_intensity:.2f}")
    print(f"Edge softness: {analysis.edge_softness:.2f}")
    print(f"Recommended layers: {analysis.layer_count}")
    print(f"Base color: RGB{analysis.recommended_base}")
    print()

    print("TECHNIQUE ZONES:")
    for zone in analysis.technique_zones:
        print(f"  Layer {zone.layer_order}: {zone.technique.value}")
        print(f"    Base: RGB{zone.base_color}")
        if zone.overlay_colors:
            print(f"    Overlays: {[f'RGB{c}' for c in zone.overlay_colors]}")
        if zone.brush_direction:
            print(f"    Brush direction: {zone.brush_direction}")

    print()
    instructions = analyzer.generate_layer_instructions(analysis)
    print("LAYER-BY-LAYER INSTRUCTIONS:")
    for inst in instructions:
        print(f"\n--- {inst['name']} ---")
        print(inst['instructions'][:500])
