"""
Acrylic Medium - Generate acrylic painting instructions.

Implements the MediumBase interface for acrylic painting with Bob Ross style.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import os

from ..base import (
    MediumBase,
    Material,
    MaterialType,
    CanvasArea,
    Substep,
    Layer,
)
from ...core import (
    PaintByNumbers,
    ImageSegmenter,
    LayerStrategy,
    RegionDetector,
    RegionDetectionConfig,
    find_colors_in_layer,
)
from ...analysis.technique_analyzer import TechniqueAnalyzer
from .constants import BrushType, StrokeMotion, PAINT_NAMES, ENCOURAGEMENTS
from .visualizer import AcrylicVisualizer


class AcrylicMedium(MediumBase):
    """
    Acrylic painting medium with Bob Ross style instructions.

    Generates warm, encouraging instructions for acrylic painting
    with detailed brush techniques and color mixing guidance.
    """

    def __init__(
        self,
        image_path: str,
        n_colors: int = 15,
        detail_level: str = "granular",
        layer_strategy: LayerStrategy = LayerStrategy.ADAPTIVE,
        **kwargs
    ):
        """
        Initialize acrylic medium.

        Args:
            image_path: Path to source image
            n_colors: Number of colors to quantize to
            detail_level: "granular" or "standard"
            layer_strategy: How to segment layers
            **kwargs: Additional configuration
        """
        super().__init__(image_path, **kwargs)

        self.n_colors = n_colors
        self.detail_level = detail_level
        self.layer_strategy = layer_strategy

        # Load and analyze image
        self.original = cv2.imread(image_path)
        self.original_rgb = cv2.cvtColor(self.original, cv2.COLOR_BGR2RGB)
        self.height, self.width = self.original.shape[:2]

        # Run analysis
        self.analyzer = TechniqueAnalyzer(image_path)
        self.analysis = self.analyzer.full_analysis(n_colors)

        # Color quantization
        self.pbn = PaintByNumbers(image_path, n_colors)
        self.pbn.quantize_colors()
        self.pbn.create_regions()

        # Create segmenter and region detector
        self.segmenter = ImageSegmenter(self.original_rgb)
        self.region_detector = RegionDetector(
            self.height,
            self.width,
            RegionDetectionConfig(use_contours=True)
        )

        # Extract palette
        self.palette = self._extract_palette()

    def _extract_palette(self) -> List[Dict]:
        """Extract color palette with Bob Ross-style names."""
        palette = []
        for i, rgb in enumerate(self.pbn.palette):
            bob_name = self._get_bob_ross_color_name(rgb)
            palette.append({
                'index': i,
                'rgb': tuple(int(c) for c in rgb),
                'bob_name': bob_name,
                'hex': '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))
            })
        return palette

    def _get_bob_ross_color_name(self, rgb: np.ndarray) -> str:
        """Convert RGB to Bob Ross-style color name."""
        r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
        h, s, v = self._rgb_to_hsv(r, g, b)

        # Very dark colors
        if v < 30:
            return PAINT_NAMES['black']

        # Very light colors
        if v > 220 and s < 30:
            return PAINT_NAMES['white']

        # Grayscale
        if s < 20:
            if v < 80:
                return "Midnight Black + Titanium White (gray mix)"
            else:
                return "Titanium White + touch of black"

        # Color by hue
        if h > 330 or h < 15:
            return PAINT_NAMES['pink'] if (v > 150 and s < 60) else PAINT_NAMES['red']
        elif 15 <= h < 45:
            return PAINT_NAMES['orange']
        elif 45 <= h < 70:
            return PAINT_NAMES['yellow']
        elif 70 <= h < 150:
            return PAINT_NAMES['light_green'] if v > 150 else PAINT_NAMES['green']
        elif 150 <= h < 190:
            return "Pthalo Blue + Sap Green mix"
        elif 190 <= h < 260:
            return PAINT_NAMES['light_blue'] if v > 150 else PAINT_NAMES['blue']
        elif 260 <= h < 290:
            return PAINT_NAMES['purple']
        elif 290 <= h < 330:
            return PAINT_NAMES['magenta']

        return "custom mix"

    def _rgb_to_hsv(self, r: int, g: int, b: int) -> Tuple[int, int, int]:
        """Convert RGB to HSV."""
        r, g, b = r/255.0, g/255.0, b/255.0
        mx = max(r, g, b)
        mn = min(r, g, b)
        df = mx - mn

        if mx == mn:
            h = 0
        elif mx == r:
            h = (60 * ((g-b)/df) + 360) % 360
        elif mx == g:
            h = (60 * ((b-r)/df) + 120) % 360
        else:
            h = (60 * ((r-g)/df) + 240) % 360

        s = 0 if mx == 0 else (df/mx) * 100
        v = mx * 100

        return int(h), int(s), int(v)

    # =========================================================================
    # MediumBase Interface Implementation
    # =========================================================================

    def get_materials(self) -> List[Material]:
        """Get list of paint materials needed."""
        materials = []

        for color_info in self.palette:
            material = Material(
                name=color_info['bob_name'],
                material_type=MaterialType.PAINT,
                color_rgb=color_info['rgb'],
                color_hex=color_info['hex'],
                identifier=f"Color {color_info['index'] + 1}",
                brand="Bob Ross",
                quantity="2.5 oz tube",
                price_usd=8.99,
                metadata={'palette_index': color_info['index']}
            )
            materials.append(material)

        return materials

    def plan_layers(self) -> List[Layer]:
        """Plan painting layers using configured strategy."""
        # Use ImageSegmenter to determine layers
        layer_defs = self.segmenter.segment(self.layer_strategy)

        layers = []
        for i, layer_def in enumerate(layer_defs):
            layer = Layer(
                layer_number=i + 1,
                name=layer_def.name,
                description=layer_def.description,
                wait_time=self._get_dry_time(layer_def.technique),
                technique_tip=self._get_technique_tip(layer_def.technique),
                encouragement=ENCOURAGEMENTS[i % len(ENCOURAGEMENTS)]
            )

            # Store layer_def for use in generate_substeps
            layer.metadata = {'layer_def': layer_def}

            layers.append(layer)

        return layers

    def generate_substeps(self, layer: Layer) -> List[Substep]:
        """Break layer into granular substeps by color and region."""
        if self.detail_level == "standard":
            return self._generate_standard_substeps(layer)
        else:  # granular
            return self._generate_granular_substeps(layer)

    def _generate_granular_substeps(self, layer: Layer) -> List[Substep]:
        """Generate detailed substeps broken down by color and canvas region."""
        substeps = []
        layer_def = layer.metadata.get('layer_def')

        if layer_def is None:
            return []

        # Find colors in this layer
        colors_in_layer = find_colors_in_layer(
            self.pbn.label_map,
            layer_def.mask,
            min_pixels=100
        )

        substep_count = 0

        for color_idx, pixel_count, coverage_fraction in colors_in_layer:
            # Get color info
            color_info = self.palette[color_idx]

            # Find regions where this color appears
            color_mask = (self.pbn.label_map == color_idx) & layer_def.mask
            regions = self.region_detector.find_regions(color_mask)

            for region in regions:
                substep_count += 1

                # Create material
                material = Material(
                    name=color_info['bob_name'],
                    material_type=MaterialType.PAINT,
                    color_rgb=color_info['rgb'],
                    color_hex=color_info['hex'],
                )

                # Create canvas area
                area = CanvasArea(
                    name=region['name'],
                    bounds=region['bounds'],
                    mask=region['mask'],
                    coverage_percent=region['coverage_percent'],
                    centroid=region['centroid'],
                )

                # Determine brush and motion
                brush = self._get_brush_for_coverage(region['coverage_percent'])
                motion = self._get_motion_for_technique(layer_def.technique)

                # Generate instruction
                instruction = self._generate_substep_instruction(
                    color_info['bob_name'],
                    area,
                    layer_def.technique,
                    brush,
                    motion
                )

                substep = Substep(
                    substep_id=f"{layer.layer_number}.{substep_count}",
                    material=material,
                    area=area,
                    technique=layer_def.technique,
                    instruction=instruction,
                    tool=brush.value,
                    motion=motion.value,
                    duration_hint=self._estimate_duration(region['coverage_percent']),
                    order_priority=substep_count
                )

                substeps.append(substep)

        return substeps

    def _generate_standard_substeps(self, layer: Layer) -> List[Substep]:
        """Generate simplified substeps for standard detail level."""
        # For standard level, create one substep per color in the layer
        layer_def = layer.metadata.get('layer_def')
        if layer_def is None:
            return []

        substeps = []
        colors_in_layer = find_colors_in_layer(
            self.pbn.label_map,
            layer_def.mask,
            min_pixels=500  # Higher threshold for standard
        )

        for i, (color_idx, pixel_count, coverage_fraction) in enumerate(colors_in_layer):
            color_info = self.palette[color_idx]

            material = Material(
                name=color_info['bob_name'],
                material_type=MaterialType.PAINT,
                color_rgb=color_info['rgb'],
                color_hex=color_info['hex'],
            )

            # Whole layer as area
            area = CanvasArea(
                name=layer.name.lower(),
                bounds=(0, 1, 0, 1),
                coverage_percent=coverage_fraction * 100,
            )

            instruction = f"Apply {color_info['bob_name']} to {layer.name.lower()} areas."

            substep = Substep(
                substep_id=f"{layer.layer_number}.{i+1}",
                material=material,
                area=area,
                technique=layer_def.technique,
                instruction=instruction,
                duration_hint="2-5 minutes",
                order_priority=i
            )

            substeps.append(substep)

        return substeps

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _get_brush_for_coverage(self, coverage: float) -> BrushType:
        """Determine appropriate brush based on area coverage."""
        if coverage > 10:
            return BrushType.TWO_INCH
        elif coverage > 3:
            return BrushType.ONE_INCH
        elif coverage > 1:
            return BrushType.FAN_BRUSH
        else:
            return BrushType.ROUND

    def _get_motion_for_technique(self, technique: str) -> StrokeMotion:
        """Get appropriate stroke motion for technique."""
        motion_map = {
            'base coat': StrokeMotion.HORIZONTAL_PULL,
            'blend': StrokeMotion.BLEND,
            'glazing': StrokeMotion.VERTICAL_PULL,
            'gradient': StrokeMotion.VERTICAL_PULL,
            'accent': StrokeMotion.TAP,
            'silhouette': StrokeMotion.LOAD_AND_PULL,
            'detail': StrokeMotion.CIRCULAR,
        }
        return motion_map.get(technique, StrokeMotion.BLEND)

    def _get_dry_time(self, technique: str) -> str:
        """Get recommended drying time."""
        if technique in ['base coat', 'silhouette']:
            return "Let dry 5-10 minutes before continuing"
        elif technique == 'glazing':
            return "Work while previous layer is slightly tacky"
        else:
            return ""

    def _get_technique_tip(self, technique: str) -> str:
        """Get technique-specific tip."""
        tips = {
            'base coat': "Use long, even strokes to ensure complete coverage.",
            'blend': "Keep your brush clean and use soft, back-and-forth motions.",
            'glazing': "Use very thin paint - we want to see through to the layer below.",
            'gradient': "Start with more paint at the top and gradually lighten as you move down.",
            'accent': "Less is more! Small touches make big impacts.",
            'silhouette': "Silhouettes should be solid - no texture needed here.",
            'detail': "Take your time with details - they bring the painting to life.",
        }
        return tips.get(technique, "")

    def _generate_substep_instruction(
        self,
        color_name: str,
        area: CanvasArea,
        technique: str,
        brush: BrushType,
        motion: StrokeMotion
    ) -> str:
        """Generate a concise, Bob Ross-style instruction."""
        instructions = {
            'base coat': f"Load your {brush.value} with {color_name} and cover the {area.name} using {motion.value}.",
            'blend': f"Apply {color_name} to the {area.name} with your {brush.value}, using {motion.value}.",
            'glazing': f"Lightly glaze {color_name} over the {area.name} - just a whisper of color.",
            'gradient': f"Pull {color_name} across the {area.name} with your {brush.value}, letting it fade naturally.",
            'accent': f"Tap a small amount of {color_name} onto the {area.name} for highlights.",
            'silhouette': f"Paint {color_name} firmly in the {area.name} for solid coverage.",
            'detail': f"Carefully add {color_name} to the {area.name} using your {brush.value}.",
        }
        return instructions.get(technique, f"Apply {color_name} to the {area.name}.")

    def _estimate_duration(self, coverage: float) -> str:
        """Estimate time for a substep based on coverage."""
        if coverage > 15:
            return "2-3 minutes"
        elif coverage > 5:
            return "1-2 minutes"
        elif coverage > 1:
            return "30-60 seconds"
        else:
            return "15-30 seconds"

    # =========================================================================
    # Visual Guide Generation
    # =========================================================================

    def create_visual_guide(
        self,
        output_dir: str,
        create_filmstrips: bool = True
    ):
        """
        Generate complete visual guide with substep images.

        This creates a directory structure with visual outputs for each
        layer and substep.

        Args:
            output_dir: Directory to save visual outputs
            create_filmstrips: Whether to create filmstrip overview images

        Returns:
            Path to output directory
        """
        # Ensure we have layers with substeps
        if not self.layers or not all(layer.substeps for layer in self.layers):
            print("Generating guide first...")
            self.generate_full_guide()

        # Create visualizer
        visualizer = AcrylicVisualizer(
            self.original_rgb,
            self.height,
            self.width
        )

        # Generate visual guide
        visualizer.create_granular_guide(
            self.layers,
            output_dir,
            create_filmstrips=create_filmstrips
        )

        return output_dir
