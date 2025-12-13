"""
Bob Ross Style Instruction Generator

Generates detailed, step-by-step painting instructions in the style of Bob Ross:
- Warm, encouraging tone
- Specific brush and technique instructions
- Layer-by-layer progression with visual progress
- Color mixing guidance
- "Happy little" details
"""

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Tuple, Optional
import json
import os
import base64

from ...analysis.technique_analyzer import TechniqueAnalyzer
from .constants import BrushType, StrokeMotion, PAINT_NAMES, ENCOURAGEMENTS
from .steps import PaintingStep, PaintingLayer, PaintingSubstep, CanvasArea


class BobRossGenerator:
    """Generates Bob Ross style painting instructions."""

    def __init__(self, image_path: str, n_colors: int = 15, use_ai_layering: bool = True,
                 auto_split_layers: bool = False):
        self.image_path = image_path
        self.n_colors = n_colors
        self.use_ai_layering = use_ai_layering
        self.auto_split_layers = auto_split_layers

        # Load image
        self.original = cv2.imread(image_path)
        self.original_rgb = cv2.cvtColor(self.original, cv2.COLOR_BGR2RGB)
        self.height, self.width = self.original.shape[:2]

        # Run analysis
        self.analyzer = TechniqueAnalyzer(image_path)
        self.analysis = self.analyzer.full_analysis(n_colors)

        # Generate color palette with names
        self.palette = self._extract_palette()

        # AI-analyzed layer plan (will be populated if use_ai_layering=True)
        self.ai_layer_plan: Optional[Dict] = None

        # Steps storage
        self.steps: List[PaintingStep] = []

    def _extract_palette(self) -> List[Dict]:
        """Extract color palette and assign Bob Ross-style names."""
        from ...core.paint_by_numbers import PaintByNumbers
        pbn = PaintByNumbers(self.image_path, self.n_colors)
        pbn.quantize_colors()

        palette = []
        for i, rgb in enumerate(pbn.palette):
            bob_name = self._get_bob_ross_color_name(rgb)
            palette.append({
                'index': i + 1,
                'rgb': tuple(int(c) for c in rgb),
                'bob_name': bob_name,
                'hex': '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))
            })

        return palette

    def _get_bob_ross_color_name(self, rgb: np.ndarray) -> str:
        """Convert RGB to a Bob Ross-style color name."""
        r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
        h, s, v = self._rgb_to_hsv(r, g, b)

        # Very dark colors
        if v < 30:
            return PAINT_NAMES['black']

        # Very light colors
        if v > 220 and s < 30:
            return PAINT_NAMES['white']

        # By hue
        if s < 20:  # Grayscale
            if v < 80:
                return "Midnight Black + Titanium White (gray mix)"
            else:
                return "Titanium White + touch of black"

        # Reds/Magentas (330-360, 0-15)
        if h > 330 or h < 15:
            if v > 150:
                return PAINT_NAMES['pink'] if s < 60 else PAINT_NAMES['red']
            return PAINT_NAMES['dark_sienna']

        # Oranges (15-45)
        if 15 <= h < 45:
            return PAINT_NAMES['orange']

        # Yellows (45-70)
        if 45 <= h < 70:
            return PAINT_NAMES['yellow']

        # Yellow-greens (70-90)
        if 70 <= h < 90:
            return PAINT_NAMES['light_green']

        # Greens (90-150)
        if 90 <= h < 150:
            if v > 150:
                return PAINT_NAMES['light_green']
            return PAINT_NAMES['green']

        # Cyans (150-190)
        if 150 <= h < 190:
            return "Pthalo Blue + Sap Green mix"

        # Blues (190-260)
        if 190 <= h < 260:
            if v > 150:
                return PAINT_NAMES['light_blue']
            return PAINT_NAMES['blue']

        # Purples (260-290)
        if 260 <= h < 290:
            return PAINT_NAMES['purple']

        # Magentas (290-330)
        if 290 <= h < 330:
            return PAINT_NAMES['magenta']

        return "custom mix"

    def _rgb_to_hsv(self, r, g, b) -> Tuple[int, int, int]:
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

    def _get_hardcoded_layer_plan(self, image_name: str) -> Optional[Dict]:
        """Get pre-analyzed layer plans for specific images."""
        # Fireweed landscape layer plan - granular 12-layer breakdown
        if 'fireweed' in image_name.lower():
            return {
                "total_layers": 12,
                "image_type": "landscape",
                "layers": [
                    {
                        "name": "sky_base",
                        "description": "base sky color",
                        "y_range": [0.0, 0.35],
                        "priority": 1,
                        "technique": "base"
                    },
                    {
                        "name": "sky_clouds",
                        "description": "cloud formations",
                        "y_range": [0.0, 0.35],
                        "priority": 2,
                        "technique": "blend"
                    },
                    {
                        "name": "distant_mountains",
                        "description": "mountain peaks",
                        "y_range": [0.2, 0.38],
                        "priority": 3,
                        "technique": "layer"
                    },
                    {
                        "name": "far_treeline",
                        "description": "distant forest line",
                        "y_range": [0.35, 0.55],
                        "priority": 4,
                        "technique": "layer"
                    },
                    {
                        "name": "mid_forest",
                        "description": "middle forest layer",
                        "y_range": [0.45, 0.65],
                        "priority": 5,
                        "technique": "detail"
                    },
                    {
                        "name": "grass_base",
                        "description": "grass base layer",
                        "y_range": [0.6, 0.85],
                        "priority": 6,
                        "technique": "blend"
                    },
                    {
                        "name": "grass_highlights",
                        "description": "lighter grass tones",
                        "y_range": [0.6, 0.85],
                        "priority": 7,
                        "technique": "detail"
                    },
                    {
                        "name": "fireweed_stems",
                        "description": "fireweed stems and leaves",
                        "y_range": [0.65, 0.95],
                        "priority": 8,
                        "technique": "layer"
                    },
                    {
                        "name": "fireweed_blooms_base",
                        "description": "darker pink fireweed blooms",
                        "y_range": [0.7, 0.95],
                        "priority": 9,
                        "technique": "detail"
                    },
                    {
                        "name": "fireweed_blooms_bright",
                        "description": "bright pink fireweed highlights",
                        "y_range": [0.7, 0.95],
                        "priority": 10,
                        "technique": "detail"
                    },
                    {
                        "name": "cloud_highlights",
                        "description": "bright cloud highlights",
                        "y_range": [0.0, 0.3],
                        "priority": 11,
                        "technique": "highlight"
                    },
                    {
                        "name": "final_accents",
                        "description": "final bright accents",
                        "y_range": [0.0, 1.0],
                        "priority": 12,
                        "technique": "highlight"
                    }
                ]
            }

        # Aurora layer plan - 3 layers that create 7 steps total
        # mid_tones (1 step) + highlights triggers aurora steps (3 steps) + silhouettes (1 step) + prep/sign (2 steps) = 7
        if 'aurora' in image_name.lower():
            return {
                "total_layers": 3,
                "image_type": "aurora",
                "layers": [
                    {
                        "name": "mid_tones",
                        "description": "faint middle value areas",
                        "y_range": [0.0, 1.0],
                        "priority": 1,
                        "technique": "blend",
                        "luminosity_range": [0.05, 0.20]  # Very subtle dark mid-tones
                    },
                    {
                        "name": "highlights",
                        "description": "aurora lights - triggers 3 aurora sub-steps",
                        "y_range": [0.0, 0.85],
                        "priority": 2,
                        "technique": "highlight",
                        "luminosity_range": [0.20, 1.0]  # All aurora colors
                    },
                    {
                        "name": "silhouettes",
                        "description": "dark foreground shapes",
                        "y_range": [0.4, 1.0],
                        "priority": 3,
                        "technique": "silhouette",
                        "luminosity_range": [0.0, 0.05]  # Darkest pixels only
                    }
                ]
            }

        return None

    def _analyze_image_with_ai(self) -> Optional[Dict]:
        """
        Use Claude Opus to analyze the image and determine painting layers dynamically.

        Returns a structured layer plan with:
        - Visual elements identified (sky, mountains, midground, foreground, etc.)
        - Painting order (back to front)
        - Spatial locations for each element
        """
        # Check for hardcoded layer plans first
        image_name = os.path.basename(self.image_path)
        hardcoded_plan = self._get_hardcoded_layer_plan(image_name)
        if hardcoded_plan:
            print(f"\n✓ Using pre-analyzed layer plan for {image_name}")
            print(f"  Layers: {[layer['name'] for layer in hardcoded_plan['layers']]}")
            return hardcoded_plan

        try:
            import anthropic

            # Encode image as base64
            with open(self.image_path, 'rb') as f:
                image_data = base64.standard_b64encode(f.read()).decode('utf-8')

            # Determine image format
            image_format = self.image_path.lower().split('.')[-1]
            if image_format == 'jpg':
                image_format = 'jpeg'

            # Initialize Anthropic client
            client = anthropic.Anthropic()

            # Prompt for image analysis
            prompt = """Analyze this image to create a Bob Ross style painting instruction plan.

Identify all the major visual elements in the image and determine the optimal order to paint them (back to front, like Bob Ross would).

For each element, provide:
1. A descriptive name (e.g., "sky", "distant_mountains", "midground_hills", "foreground_flowers", "water_reflections")
2. A vertical position range where it appears (0.0 = top of image, 1.0 = bottom)
3. A painting priority (1 = paint first/background, higher numbers = paint later/foreground)
4. A brief description of what the element is
5. The painting technique that would work best: "base", "blend", "layer", "detail", "glaze", "silhouette", or "highlight"

Return ONLY a valid JSON object in this exact format (no markdown, no explanation):
{
  "total_layers": <number of painting layers>,
  "image_type": "<landscape|portrait|abstract|still_life|other>",
  "layers": [
    {
      "name": "<element_name>",
      "description": "<what this element is>",
      "y_range": [<min_y>, <max_y>],
      "priority": <painting_order_number>,
      "technique": "<painting_technique>"
    },
    ...
  ]
}

Aim for 5-7 layers total for a good progression. Order the layers array by priority (background elements first)."""

            # Call Claude Opus
            message = client.messages.create(
                model="claude-opus-4-20250514",
                max_tokens=2048,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": f"image/{image_format}",
                                "data": image_data
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }]
            )

            # Parse response
            response_text = message.content[0].text.strip()
            layer_plan = json.loads(response_text)

            print(f"\n✓ AI Analysis Complete: {layer_plan['total_layers']} layers identified")
            print(f"  Image type: {layer_plan.get('image_type', 'unknown')}")
            print(f"  Layers: {[layer['name'] for layer in layer_plan['layers']]}")

            return layer_plan

        except Exception as e:
            print(f"\n⚠ AI image analysis failed: {e}")
            print("  Falling back to traditional layer detection...")
            return None

    def _convert_ai_layers_to_masks(self, ai_plan: Dict) -> List[Dict]:
        """
        Convert AI-analyzed layer plan into layer masks for painting.

        Creates spatial masks based on y_range and luminosity/color analysis,
        then auto-splits layers that add too much complexity at once.
        """
        from ...core.paint_by_numbers import PaintByNumbers
        pbn = PaintByNumbers(self.image_path, self.n_colors)
        pbn.quantize_colors()

        # Get luminosity for region detection
        lab = cv2.cvtColor(self.original, cv2.COLOR_BGR2LAB)
        luminosity = lab[:, :, 0].astype(np.float32) / 255.0

        layers = []
        h, w = self.height, self.width

        for layer_info in ai_plan['layers']:
            # Create spatial mask based on y_range
            y_min, y_max = layer_info['y_range']
            y_start_px = int(y_min * h)
            y_end_px = int(y_max * h)

            # Create base mask for this vertical region
            spatial_mask = np.zeros((h, w), dtype=bool)
            spatial_mask[y_start_px:y_end_px, :] = True

            # Refine mask based on luminosity
            # Use explicit luminosity_range if provided, otherwise use technique-based defaults
            if 'luminosity_range' in layer_info:
                lum_min, lum_max = layer_info['luminosity_range']
                refined_mask = spatial_mask & (luminosity >= lum_min) & (luminosity < lum_max)
            elif layer_info['technique'] in ['silhouette', 'highlight']:
                # Silhouettes: very dark
                if layer_info['technique'] == 'silhouette':
                    refined_mask = spatial_mask & (luminosity < 0.2)
                # Highlights: very bright
                else:
                    refined_mask = spatial_mask & (luminosity > 0.7)
            elif layer_info['technique'] == 'base':
                # Base layers: broader coverage
                refined_mask = spatial_mask
            else:
                # Other techniques: moderate refinement
                if layer_info['priority'] <= 2:
                    refined_mask = spatial_mask & (luminosity >= 0.3)
                else:
                    refined_mask = spatial_mask & (luminosity >= 0.2)

            # Calculate coverage
            coverage = np.sum(refined_mask) / (h * w)

            # Only add layer if it has reasonable coverage
            if coverage > 0.01:
                layer_dict = {
                    'name': layer_info['name'],
                    'mask': refined_mask,
                    'priority': layer_info['priority'],
                    'technique': layer_info['technique'],
                    'description': layer_info['description'],
                    'y_range': tuple(layer_info['y_range']),
                    'coverage': coverage
                }
                layers.append(layer_dict)

        # Sort by priority first
        layers = sorted(layers, key=lambda x: x['priority'])

        # Auto-split layers with too much complexity (optional)
        if self.auto_split_layers:
            layers = self._auto_split_complex_layers(layers, pbn, luminosity)

        return layers

    def _auto_split_complex_layers(self, layers: List[Dict], pbn, luminosity: np.ndarray) -> List[Dict]:
        """
        Automatically split layers that add too much visual complexity at once.

        Analyzes pixel changes between cumulative layer states and splits
        any layer that adds >15% significant pixel changes.
        """
        if not layers:
            return layers

        h, w = self.height, self.width
        refined_layers = []

        # Build up cumulative canvas to measure changes
        canvas = np.ones((h, w, 3), dtype=np.uint8) * 200  # Light base

        for i, layer in enumerate(layers):
            # Create next canvas state with this layer
            next_canvas = canvas.copy()
            if layer['mask'] is not None:
                for c in range(3):
                    next_canvas[:, :, c] = np.where(
                        layer['mask'],
                        self.original_rgb[:, :, c],
                        next_canvas[:, :, c]
                    )

            # Calculate complexity change (pixel differences)
            diff = np.linalg.norm(next_canvas.astype(float) - canvas.astype(float), axis=2)
            significant_change = diff > 30  # Pixels that changed significantly
            change_percent = np.sum(significant_change) / (h * w) * 100

            # If this layer adds too much (>15% of pixels), try to split it
            if change_percent > 15.0 and layer['coverage'] > 0.05:
                print(f"  Layer '{layer['name']}' adds {change_percent:.1f}% complexity - splitting...")

                # Split by luminosity within the mask
                mask_luminosity = luminosity[layer['mask']]
                if len(mask_luminosity) > 0:
                    median_lum = np.median(mask_luminosity)

                    # Dark half
                    dark_mask = layer['mask'] & (luminosity < median_lum)
                    dark_coverage = np.sum(dark_mask) / (h * w)

                    # Light half
                    light_mask = layer['mask'] & (luminosity >= median_lum)
                    light_coverage = np.sum(light_mask) / (h * w)

                    if dark_coverage > 0.01:
                        refined_layers.append({
                            **layer,
                            'name': f"{layer['name']}_dark",
                            'description': f"{layer['description']} (darker tones)",
                            'mask': dark_mask,
                            'coverage': dark_coverage,
                            'priority': layer['priority']
                        })

                    if light_coverage > 0.01:
                        refined_layers.append({
                            **layer,
                            'name': f"{layer['name']}_light",
                            'description': f"{layer['description']} (lighter tones)",
                            'mask': light_mask,
                            'coverage': light_coverage,
                            'priority': layer['priority'] + 0.5
                        })

                    # Update canvas with split layers
                    for c in range(3):
                        canvas[:, :, c] = np.where(
                            dark_mask,
                            self.original_rgb[:, :, c],
                            canvas[:, :, c]
                        )
                else:
                    # Couldn't split, just add as-is
                    refined_layers.append(layer)
                    canvas = next_canvas
            else:
                # Layer is fine, add it
                refined_layers.append(layer)
                canvas = next_canvas

        return sorted(refined_layers, key=lambda x: x['priority'])

    def _get_region_description(self, y_start: float, y_end: float, x_start: float = 0, x_end: float = 1) -> str:
        """Get a human-readable description of a canvas region."""
        if y_start == 0 and y_end == 1 and x_start == 0 and x_end == 1:
            return "the entire canvas"

        vertical = ""
        if y_end <= 0.35:
            vertical = "top"
        elif y_start >= 0.65:
            vertical = "bottom"
        elif y_start >= 0.3 and y_end <= 0.7:
            vertical = "middle"
        elif y_start == 0:
            vertical = "top half" if y_end <= 0.55 else "upper"
        else:
            vertical = "lower"

        if x_start == 0 and x_end == 1:
            return f"the {vertical} of the canvas"
        elif x_end <= 0.35:
            return f"the {vertical} left"
        elif x_start >= 0.65:
            return f"the {vertical} right"
        else:
            return f"the {vertical} center"

    def generate_steps(self) -> List[PaintingStep]:
        """Generate all painting steps."""
        self.steps = []
        step_num = 1

        # Step 1: Canvas preparation
        self.steps.append(self._create_prep_step(step_num))
        step_num += 1

        # Analyze image structure for step generation
        self.layers = self._analyze_layers()

        # Generate steps for each layer
        for layer in self.layers:
            new_steps = self._generate_layer_steps(layer, step_num)
            self.steps.extend(new_steps)
            step_num += len(new_steps)

        # Final step: Signing
        self.steps.append(self._create_signing_step(step_num))

        return self.steps

    def _create_prep_step(self, step_num: int) -> PaintingStep:
        """Create the canvas preparation step."""
        if self.analysis.has_dark_background:
            instruction = f"""Let's start by covering our entire canvas with a nice dark base.

Load your 2-inch brush with {PAINT_NAMES['black']}.
Use long, horizontal strokes across the entire canvas.
Make sure you cover every bit of white - we want a nice, even dark coating.

This dark base is what's going to make our colors really POP later.
Think of it as the night sky that our beautiful lights will shine against."""

            return PaintingStep(
                step_number=step_num,
                title="Cover the Canvas",
                instruction=instruction,
                brush=BrushType.TWO_INCH,
                motion=StrokeMotion.HORIZONTAL_PULL,
                colors=[PAINT_NAMES['black']],
                color_rgbs=[(0, 0, 0)],
                canvas_region="the entire canvas",
                region_mask=None,
                technique_tip="Keep your brush strokes going the same direction for an even coat.",
                encouragement="This is the foundation of our painting. Take your time.",
                duration_hint="2-3 minutes"
            )
        else:
            # Light background
            bg_color = self.palette[0] if self.palette else {'bob_name': 'Titanium White', 'rgb': (255, 255, 255)}
            instruction = f"""Let's start with a nice base coat.

Load your 2-inch brush with {bg_color['bob_name']}.
Cover the canvas using criss-cross strokes.
We want a nice, even coverage to work on."""

            return PaintingStep(
                step_number=step_num,
                title="Base Coat",
                instruction=instruction,
                brush=BrushType.TWO_INCH,
                motion=StrokeMotion.CRISS_CROSS,
                colors=[bg_color['bob_name']],
                color_rgbs=[bg_color['rgb']],
                canvas_region="the entire canvas",
                region_mask=None,
                technique_tip="Criss-cross strokes help create an even coverage.",
                encouragement=ENCOURAGEMENTS[0],
                duration_hint="1-2 minutes"
            )

    def _analyze_layers(self) -> List[Dict]:
        """Analyze image to determine painting layers from back to front."""
        layers = []

        # Try AI-based layer analysis first (if enabled)
        if self.use_ai_layering and self.ai_layer_plan is None:
            self.ai_layer_plan = self._analyze_image_with_ai()

        if self.ai_layer_plan:
            # Convert AI layer plan to internal layer format
            return self._convert_ai_layers_to_masks(self.ai_layer_plan)

        # Fallback to organic segmentation (follows natural boundaries)
        print("  Using organic segmentation (natural boundaries)...")
        from ...analysis.simple_organic_segmentation import segment_into_painting_layers

        # Use organic segmentation to find natural layers
        layers = segment_into_painting_layers(self.original_rgb, n_layers=6)

        # Get luminosity and PaintByNumbers for auto-splitting (if enabled)
        if self.auto_split_layers:
            lab = cv2.cvtColor(self.original, cv2.COLOR_BGR2LAB)
            luminosity = lab[:, :, 0].astype(np.float32) / 255.0

            from ...core.paint_by_numbers import PaintByNumbers
            pbn = PaintByNumbers(self.image_path, self.n_colors)
            pbn.quantize_colors()

            layers = self._auto_split_complex_layers(layers, pbn, luminosity)

        return sorted(layers, key=lambda x: x['priority'])

        # OLD CODE BELOW - keeping for reference but not executed
        h, w = self.height, self.width

        # Layer 1: Background (handled by prep step)
        if self.analysis.has_dark_background:
            dark_mask = luminosity < 0.25
            if np.sum(dark_mask) > 0.1 * h * w:
                layers.append({
                    'name': 'dark_background',
                    'mask': dark_mask,
                    'priority': 1,
                    'technique': 'base',
                    'description': 'dark background areas',
                    'y_range': (0, 1)
                })

        # Layer 2-6: Analyze by spatial regions (top to bottom for landscapes)
        # Create 3-5 distinct spatial layers based on color clustering and vertical position
        layer_priority = 2

        # Sky/upper background region (top 40%)
        upper_region = (0, int(h * 0.4))
        upper_mask = np.zeros((h, w), dtype=bool)
        upper_mask[upper_region[0]:upper_region[1], :] = True
        upper_luminosity = luminosity[upper_region[0]:upper_region[1], :]

        # Divide upper region into background sky and detailed sky
        if np.mean(upper_luminosity) > 0.5:  # Light sky
            # Base sky layer
            sky_base_mask = upper_mask & (luminosity >= 0.4) & (luminosity < 0.7)
            if np.sum(sky_base_mask) > 0.02 * h * w:
                layers.append({
                    'name': 'sky_base',
                    'mask': sky_base_mask,
                    'priority': layer_priority,
                    'technique': 'blend',
                    'description': 'base sky tones',
                    'y_range': (0, 0.4)
                })
                layer_priority += 1

            # Sky details/highlights
            sky_detail_mask = upper_mask & (luminosity >= 0.7)
            if np.sum(sky_detail_mask) > 0.01 * h * w:
                layers.append({
                    'name': 'sky_detail',
                    'mask': sky_detail_mask,
                    'priority': layer_priority,
                    'technique': 'detail',
                    'description': 'sky details and highlights',
                    'y_range': (0, 0.4)
                })
                layer_priority += 1

        # Middle region - distant elements (30-60%)
        middle_region = (int(h * 0.3), int(h * 0.6))
        middle_mask = np.zeros((h, w), dtype=bool)
        middle_mask[middle_region[0]:middle_region[1], :] = True

        # Distant mountains/background elements
        distant_mask = middle_mask & (luminosity >= 0.25) & (luminosity < 0.55)
        if np.sum(distant_mask) > 0.02 * h * w:
            layers.append({
                'name': 'distant_elements',
                'mask': distant_mask,
                'priority': layer_priority,
                'technique': 'layer',
                'description': 'distant background elements',
                'y_range': (0.3, 0.6)
            })
            layer_priority += 1

        # Mid-ground region (40-80%)
        midground_region = (int(h * 0.4), int(h * 0.8))
        midground_mask = np.zeros((h, w), dtype=bool)
        midground_mask[midground_region[0]:midground_region[1], :] = True

        # Midground colors - separate into dark and light
        midground_dark = midground_mask & (luminosity >= 0.2) & (luminosity < 0.45)
        midground_light = midground_mask & (luminosity >= 0.45) & (luminosity < 0.65)

        if np.sum(midground_dark) > 0.02 * h * w:
            layers.append({
                'name': 'midground_dark',
                'mask': midground_dark,
                'priority': layer_priority,
                'technique': 'blend',
                'description': 'darker midground elements',
                'y_range': (0.4, 0.8)
            })
            layer_priority += 1

        if np.sum(midground_light) > 0.02 * h * w:
            layers.append({
                'name': 'midground_light',
                'mask': midground_light,
                'priority': layer_priority,
                'technique': 'layer',
                'description': 'lighter midground elements',
                'y_range': (0.4, 0.8)
            })
            layer_priority += 1

        # Foreground region (60-100%)
        foreground_region = (int(h * 0.6), h)
        foreground_mask = np.zeros((h, w), dtype=bool)
        foreground_mask[foreground_region[0]:foreground_region[1], :] = True

        # Foreground details - separate by color/brightness
        foreground_colors = foreground_mask & (luminosity >= 0.3)
        if np.sum(foreground_colors) > 0.02 * h * w:
            layers.append({
                'name': 'foreground',
                'mask': foreground_colors,
                'priority': layer_priority,
                'technique': 'detail',
                'description': 'foreground details',
                'y_range': (0.6, 1.0)
            })
            layer_priority += 1

        # Final details layer - bright highlights across entire image
        if self.analysis.has_glow_effects or np.sum(luminosity > 0.75) > 0.01 * h * w:
            highlight_mask = luminosity > 0.75
            layers.append({
                'name': 'final_highlights',
                'mask': highlight_mask,
                'priority': layer_priority,
                'technique': 'highlight',
                'description': 'final bright highlights',
                'y_range': (0, 1)
            })
            layer_priority += 1

        # Fallback: If we don't have enough layers, use old simple method
        if len(layers) < 3:
            # Use original simple method as fallback
            mid_mask = (luminosity >= 0.25) & (luminosity < 0.6)
            if np.sum(mid_mask) > 0.05 * h * w:
                layers.append({
                    'name': 'midtones',
                    'mask': mid_mask,
                    'priority': 2,
                    'technique': 'blend',
                    'description': 'mid-tone areas',
                    'y_range': (0, 1)
                })

            if self.analysis.has_glow_effects:
                bright_mask = luminosity >= 0.6
                layers.append({
                    'name': 'highlights',
                    'mask': bright_mask,
                    'priority': 3,
                    'technique': 'glaze',
                    'description': 'bright and glowing areas',
                    'y_range': (0, 1)
                })

        return sorted(layers, key=lambda x: x['priority'])

    def _generate_layer_steps(self, layer: Dict, start_step: int) -> List[PaintingStep]:
        """Generate steps for a specific layer."""
        steps = []

        # Handle layers based on technique (for AI-generated layers)
        technique = layer.get('technique', '')

        # Special cases first
        if layer['name'] == 'dark_background':
            # Background is handled in prep, skip
            return []

        elif ('highlight' in layer['name'].lower() or technique == 'highlight') and self.analysis.dominant_pattern == 'aurora':
            # Special aurora handling - generates "First Aurora Layer", "Add Movement to Aurora", etc.
            steps.extend(self._generate_aurora_steps(layer, start_step))
            return steps

        # Route by technique for AI-generated layers
        if technique == 'base':
            steps.extend(self._generate_base_layer_steps(layer, start_step))

        elif technique == 'blend':
            steps.extend(self._generate_blend_layer_steps(layer, start_step))

        elif technique == 'layer':
            steps.extend(self._generate_paint_layer_steps(layer, start_step))

        elif technique == 'detail':
            steps.extend(self._generate_detail_layer_steps(layer, start_step))

        elif technique == 'glaze':
            steps.extend(self._generate_glaze_layer_steps(layer, start_step))

        elif technique == 'silhouette':
            steps.extend(self._generate_silhouette_steps(layer, start_step))

        elif technique == 'highlight':
            steps.extend(self._generate_highlight_layer_steps(layer, start_step))

        # Fallback to name-based routing for legacy layers
        elif layer['name'] == 'highlights':
            steps.extend(self._generate_glow_steps(layer, start_step))

        elif layer['name'] == 'midtones':
            steps.extend(self._generate_midtone_steps(layer, start_step))

        elif layer['name'] == 'foreground_silhouette':
            steps.extend(self._generate_silhouette_steps(layer, start_step))

        elif layer['name'] == 'sky_base':
            steps.extend(self._generate_sky_base_steps(layer, start_step))

        elif layer['name'] == 'sky_detail':
            steps.extend(self._generate_sky_detail_steps(layer, start_step))

        elif layer['name'] == 'distant_elements':
            steps.extend(self._generate_distant_steps(layer, start_step))

        elif layer['name'] == 'midground_dark':
            steps.extend(self._generate_midground_dark_steps(layer, start_step))

        elif layer['name'] == 'midground_light':
            steps.extend(self._generate_midground_light_steps(layer, start_step))

        elif layer['name'] == 'foreground':
            steps.extend(self._generate_foreground_steps(layer, start_step))

        elif layer['name'] == 'final_highlights':
            steps.extend(self._generate_final_highlights_steps(layer, start_step))

        return steps

    def _generate_aurora_steps(self, layer: Dict, start_step: int) -> List[PaintingStep]:
        """Generate steps specifically for painting aurora/northern lights."""
        steps = []
        step_num = start_step

        # Find aurora colors in palette (greens, magentas, cyans)
        aurora_colors = []
        for color in self.palette:
            r, g, b = color['rgb']
            h, s, v = self._rgb_to_hsv(r, g, b)
            # Aurora colors: greens (90-170), magentas (290-340), bright
            if v > 40 and s > 30:
                if (80 <= h <= 180) or (280 <= h <= 350):
                    aurora_colors.append(color)

        if not aurora_colors:
            aurora_colors = self.palette[2:6]  # Fallback

        # Sort by brightness (darkest first for layering)
        aurora_colors.sort(key=lambda c: sum(c['rgb']))

        # Step: Base aurora glow (darkest aurora color)
        if len(aurora_colors) >= 1:
            color = aurora_colors[0]
            steps.append(PaintingStep(
                step_number=step_num,
                title="First Aurora Layer",
                instruction=f"""Now the magic begins! We're going to add our first layer of northern lights.

Take a CLEAN 2-inch brush and dip just the corner into {color['bob_name']}.
Very lightly, starting from the top of the canvas, pull down in sweeping vertical strokes.
Let the brush dance across the canvas - don't press too hard.

The key here is THIN layers. We want to see the dark background through the color.
This is called glazing - building up transparent layers.""",
                brush=BrushType.TWO_INCH,
                motion=StrokeMotion.VERTICAL_PULL,
                colors=[color['bob_name']],
                color_rgbs=[color['rgb']],
                canvas_region="the upper two-thirds of the canvas",
                region_mask=layer['mask'],
                technique_tip="Less is more! Use very little paint for a transparent, glowing effect.",
                encouragement="See how that color just floats on the dark? Beautiful.",
                duration_hint="2-3 minutes"
            ))
            step_num += 1

        # Step: Second aurora color
        if len(aurora_colors) >= 2:
            color = aurora_colors[1]
            steps.append(PaintingStep(
                step_number=step_num,
                title="Add Movement to the Aurora",
                instruction=f"""Let's add some movement and life to our aurora.

With a clean brush, pick up some {color['bob_name']}.
Now, instead of straight down, let's make gentle S-curves and waves.
Start at the top and let your brush flow down in a dancing motion.

Think of how the northern lights actually move - they shimmer and wave.
Follow the shapes that are already there and add to them.""",
                brush=BrushType.TWO_INCH,
                motion=StrokeMotion.VERTICAL_PULL,
                colors=[color['bob_name']],
                color_rgbs=[color['rgb']],
                canvas_region="the upper portion, following the light shapes",
                region_mask=layer['mask'],
                technique_tip="Let your wrist be loose. The aurora should look like it's flowing.",
                encouragement="Look at that! It's starting to come alive!",
                duration_hint="2-3 minutes"
            ))
            step_num += 1

        # Step: Brightest highlights
        if len(aurora_colors) >= 3:
            color = aurora_colors[-1]  # Brightest
            steps.append(PaintingStep(
                step_number=step_num,
                title="Brightest Aurora Highlights",
                instruction=f"""Now let's add the brightest parts - where the light is most intense.

Wipe your brush clean, then pick up just a tiny bit of {color['bob_name']}.
Find the areas where the aurora is brightest and add small touches.
Don't cover everything - just the peaks and brightest spots.

This is where the magic happens. These bright spots make everything glow.""",
                brush=BrushType.FAN_BRUSH,
                motion=StrokeMotion.TAP,
                colors=[color['bob_name']],
                color_rgbs=[color['rgb']],
                canvas_region="only the brightest spots in the aurora",
                region_mask=None,
                technique_tip="A little goes a long way with highlights. Less is more!",
                encouragement="Isn't that something? Look at how it glows!",
                duration_hint="1-2 minutes"
            ))
            step_num += 1

        return steps

    def _generate_glow_steps(self, layer: Dict, start_step: int) -> List[PaintingStep]:
        """Generate steps for generic glow effects."""
        steps = []
        # Simplified for non-aurora glows
        bright_colors = [c for c in self.palette if sum(c['rgb']) > 400]
        if not bright_colors:
            bright_colors = self.palette[-3:]

        color = bright_colors[0] if bright_colors else self.palette[-1]
        steps.append(PaintingStep(
            step_number=start_step,
            title="Add Glowing Highlights",
            instruction=f"""Time to add the bright, glowing areas.

Load your brush with {color['bob_name']} - just a small amount.
Gently apply to the brightest areas using light, dabbing motions.
Build up gradually - you can always add more.""",
            brush=BrushType.FAN_BRUSH,
            motion=StrokeMotion.TAP,
            colors=[color['bob_name']],
            color_rgbs=[color['rgb']],
            canvas_region="the bright areas",
            region_mask=layer['mask'],
            technique_tip="Build up brightness slowly in thin layers.",
            encouragement="Let it glow!",
            duration_hint="2-3 minutes"
        ))
        return steps

    def _generate_midtone_steps(self, layer: Dict, start_step: int) -> List[PaintingStep]:
        """Generate steps for mid-tone areas."""
        mid_colors = [c for c in self.palette if 150 < sum(c['rgb']) < 500]
        if not mid_colors:
            return []

        color = mid_colors[0]
        return [PaintingStep(
            step_number=start_step,
            title="Mid-Tone Areas",
            instruction=f"""Now let's work on the middle values.

Load your brush with {color['bob_name']}.
Work these areas with gentle blending strokes.
This creates the transition between dark and light.""",
            brush=BrushType.TWO_INCH,
            motion=StrokeMotion.BLEND,
            colors=[color['bob_name']],
            color_rgbs=[color['rgb']],
            canvas_region="the middle-tone areas",
            region_mask=layer['mask'],
            technique_tip="Blend softly where colors meet.",
            encouragement="Looking good!",
            duration_hint="2-3 minutes"
        )]

    def _generate_silhouette_steps(self, layer: Dict, start_step: int) -> List[PaintingStep]:
        """Generate steps for foreground silhouettes."""
        return [PaintingStep(
            step_number=start_step,
            title="Foreground Silhouettes",
            instruction=f"""Now let's add our foreground elements.

These dark shapes in front really make the scene pop.
Load your brush with {PAINT_NAMES['black']} mixed with a tiny bit of Prussian Blue.
Paint the silhouette shapes firmly - these should be solid and dark.

This could be trees, mountains, a figure, or anything that frames our scene.""",
            brush=BrushType.ONE_INCH,
            motion=StrokeMotion.LOAD_AND_PULL,
            colors=[PAINT_NAMES['black'], PAINT_NAMES['blue']],
            color_rgbs=[(0, 0, 0), (0, 49, 83)],
            canvas_region="the foreground shapes",
            region_mask=layer['mask'],
            technique_tip="Silhouettes should be dark and solid - no texture needed.",
            encouragement="These dark shapes make everything else shine brighter!",
            duration_hint="3-5 minutes"
        )]

    def _generate_sky_base_steps(self, layer: Dict, start_step: int) -> List[PaintingStep]:
        """Generate steps for base sky tones."""
        # Find sky colors (bright, desaturated)
        sky_colors = [c for c in self.palette if sum(c['rgb']) > 350]
        if not sky_colors:
            sky_colors = self.palette[-3:]

        color = sky_colors[0] if sky_colors else self.palette[-1]
        return [PaintingStep(
            step_number=start_step,
            title="Sky Base",
            instruction=f"""Let's start building our sky.

Load your 2-inch brush with {color['bob_name']}.
Starting at the top, use long horizontal strokes to create the base sky color.
Don't worry about making it perfect - the sky is ever-changing.""",
            brush=BrushType.TWO_INCH,
            motion=StrokeMotion.HORIZONTAL_PULL,
            colors=[color['bob_name']],
            color_rgbs=[color['rgb']],
            canvas_region="the sky area",
            region_mask=layer['mask'],
            technique_tip="Use smooth, even strokes for a gentle sky.",
            encouragement="Beautiful! The sky is taking shape.",
            duration_hint="2-3 minutes"
        )]

    def _generate_sky_detail_steps(self, layer: Dict, start_step: int) -> List[PaintingStep]:
        """Generate steps for sky details and highlights."""
        bright_colors = [c for c in self.palette if sum(c['rgb']) > 500]
        if not bright_colors:
            bright_colors = self.palette[-2:]

        color = bright_colors[0] if bright_colors else self.palette[-1]
        return [PaintingStep(
            step_number=start_step,
            title="Sky Details",
            instruction=f"""Now let's add some life to our sky.

With a clean brush, pick up a little {color['bob_name']}.
Add gentle touches where the sky is brightest - maybe where clouds catch the light.
Use light, sweeping motions - let the brush barely touch the canvas.""",
            brush=BrushType.FAN_BRUSH,
            motion=StrokeMotion.TAP,
            colors=[color['bob_name']],
            color_rgbs=[color['rgb']],
            canvas_region="the bright areas of the sky",
            region_mask=layer['mask'],
            technique_tip="Less is more with sky highlights - keep them subtle.",
            encouragement="Look at that depth! The sky is really coming alive.",
            duration_hint="1-2 minutes"
        )]

    def _generate_distant_steps(self, layer: Dict, start_step: int) -> List[PaintingStep]:
        """Generate steps for distant background elements."""
        # Find mid-range colors
        distant_colors = [c for c in self.palette if 200 < sum(c['rgb']) < 450]
        if not distant_colors:
            distant_colors = self.palette[2:5]

        color = distant_colors[0] if distant_colors else self.palette[2]
        return [PaintingStep(
            step_number=start_step,
            title="Distant Background",
            instruction=f"""Time to add those far-away elements.

Load your brush with {color['bob_name']}.
Using gentle, horizontal strokes, paint the distant shapes.
These could be mountains, hills, or the horizon - whatever you see in your mind.
Remember: distant objects are softer and less detailed.""",
            brush=BrushType.TWO_INCH,
            motion=StrokeMotion.HORIZONTAL_PULL,
            colors=[color['bob_name']],
            color_rgbs=[color['rgb']],
            canvas_region="the distant background",
            region_mask=layer['mask'],
            technique_tip="Keep distant elements soft and muted - they're far away!",
            encouragement="Perfect! That creates wonderful depth.",
            duration_hint="2-3 minutes"
        )]

    def _generate_midground_dark_steps(self, layer: Dict, start_step: int) -> List[PaintingStep]:
        """Generate steps for darker midground elements."""
        dark_mid_colors = [c for c in self.palette if 100 < sum(c['rgb']) < 300]
        if not dark_mid_colors:
            dark_mid_colors = self.palette[1:4]

        color = dark_mid_colors[0] if dark_mid_colors else self.palette[1]
        return [PaintingStep(
            step_number=start_step,
            title="Midground Shapes",
            instruction=f"""Now let's build up the middle distance.

With {color['bob_name']}, start creating the shapes in the midground.
Use firm, confident strokes - these elements are closer and more defined.
Maybe it's trees, rocks, or rolling hills - let the painting guide you.""",
            brush=BrushType.TWO_INCH,
            motion=StrokeMotion.BLEND,
            colors=[color['bob_name']],
            color_rgbs=[color['rgb']],
            canvas_region="the midground area",
            region_mask=layer['mask'],
            technique_tip="Blend where midground meets background for smooth transitions.",
            encouragement="That's it! You're building beautiful layers.",
            duration_hint="2-3 minutes"
        )]

    def _generate_midground_light_steps(self, layer: Dict, start_step: int) -> List[PaintingStep]:
        """Generate steps for lighter midground elements."""
        light_mid_colors = [c for c in self.palette if 250 < sum(c['rgb']) < 500]
        if not light_mid_colors:
            light_mid_colors = self.palette[3:6]

        color = light_mid_colors[0] if light_mid_colors else self.palette[3]
        return [PaintingStep(
            step_number=start_step,
            title="Midground Highlights",
            instruction=f"""Let's add some light to the midground.

Pick up {color['bob_name']} on your brush.
Add touches where light hits your midground elements.
Think about where the sun would catch - tops of hills, edges of shapes.""",
            brush=BrushType.TWO_INCH,
            motion=StrokeMotion.TAP,
            colors=[color['bob_name']],
            color_rgbs=[color['rgb']],
            canvas_region="the lighter midground areas",
            region_mask=layer['mask'],
            technique_tip="Add highlights with a light touch - let the darker colors show through.",
            encouragement="Beautiful! The scene is really taking shape now.",
            duration_hint="2-3 minutes"
        )]

    def _generate_foreground_steps(self, layer: Dict, start_step: int) -> List[PaintingStep]:
        """Generate steps for foreground details."""
        # Find bright/saturated foreground colors
        fg_colors = [c for c in self.palette if sum(c['rgb']) > 200]
        if not fg_colors:
            fg_colors = self.palette[-4:]

        color = fg_colors[0] if fg_colors else self.palette[-2]
        return [PaintingStep(
            step_number=start_step,
            title="Foreground Details",
            instruction=f"""Time for the foreground - the closest elements to us.

Load your brush with {color['bob_name']}.
Paint these elements with more detail and stronger colors.
Foreground is where we can be bold! Add texture, detail, and personality.
This is what really brings the painting to life.""",
            brush=BrushType.ONE_INCH,
            motion=StrokeMotion.LOAD_AND_PULL,
            colors=[color['bob_name']],
            color_rgbs=[color['rgb']],
            canvas_region="the foreground",
            region_mask=layer['mask'],
            technique_tip="Foreground elements should be brightest and most detailed.",
            encouragement="Fantastic! That foreground really pops!",
            duration_hint="3-4 minutes"
        )]

    def _generate_final_highlights_steps(self, layer: Dict, start_step: int) -> List[PaintingStep]:
        """Generate steps for final bright highlights."""
        brightest = [c for c in self.palette if sum(c['rgb']) > 600]
        if not brightest:
            brightest = self.palette[-1:]

        color = brightest[0] if brightest else self.palette[-1]
        return [PaintingStep(
            step_number=start_step,
            title="Final Highlights",
            instruction=f"""Now for those final touches of light.

Take just a tiny bit of {color['bob_name']} - and I mean tiny!
Find the absolute brightest spots in your painting.
Add little touches where the light really shines.
These final highlights make everything glow.""",
            brush=BrushType.FAN_BRUSH,
            motion=StrokeMotion.TAP,
            colors=[color['bob_name']],
            color_rgbs=[color['rgb']],
            canvas_region="the brightest highlights",
            region_mask=layer['mask'],
            technique_tip="Final highlights should be subtle and selective.",
            encouragement="Perfect! Those highlights bring it all together!",
            duration_hint="1-2 minutes"
        )]

    # Generic technique-based step generators for AI-identified layers

    def _generate_base_layer_steps(self, layer: Dict, start_step: int) -> List[PaintingStep]:
        """Generic base layer painting (broad coverage)."""
        colors = [c for c in self.palette if 200 < sum(c['rgb']) < 600]
        color = colors[0] if colors else self.palette[int(len(self.palette)/2)]

        title = layer.get('description', layer['name']).title()
        region = layer.get('description', layer['name'])

        return [PaintingStep(
            step_number=start_step,
            title=title,
            instruction=f"""Let's paint the {region}.

Load your brush with {color['bob_name']}.
Cover this area with smooth, even strokes.
This is our foundation for this part of the painting.""",
            brush=BrushType.TWO_INCH,
            motion=StrokeMotion.HORIZONTAL_PULL,
            colors=[color['bob_name']],
            color_rgbs=[color['rgb']],
            canvas_region=region,
            region_mask=layer['mask'],
            technique_tip="Keep your coverage even and smooth.",
            encouragement="Great foundation! This is coming together nicely.",
            duration_hint="2-3 minutes"
        )]

    def _generate_blend_layer_steps(self, layer: Dict, start_step: int) -> List[PaintingStep]:
        """Generic blending layer (soft transitions)."""
        colors = [c for c in self.palette if 150 < sum(c['rgb']) < 500]
        color = colors[0] if colors else self.palette[int(len(self.palette)/3)]

        title = layer.get('description', layer['name']).title()
        region = layer.get('description', layer['name'])

        return [PaintingStep(
            step_number=start_step,
            title=title,
            instruction=f"""Now let's add {region}.

With {color['bob_name']}, work this area with gentle blending strokes.
Blend where the colors meet - we want soft, natural transitions.
Let the colors flow together.""",
            brush=BrushType.TWO_INCH,
            motion=StrokeMotion.BLEND,
            colors=[color['bob_name']],
            color_rgbs=[color['rgb']],
            canvas_region=region,
            region_mask=layer['mask'],
            technique_tip="Blend softly for smooth, natural transitions.",
            encouragement="Beautiful blending! See how that creates depth?",
            duration_hint="2-3 minutes"
        )]

    def _generate_paint_layer_steps(self, layer: Dict, start_step: int) -> List[PaintingStep]:
        """Generic painting layer (distinct shapes)."""
        colors = [c for c in self.palette if sum(c['rgb']) > 180]
        color = colors[0] if colors else self.palette[-3]

        title = layer.get('description', layer['name']).title()
        region = layer.get('description', layer['name'])

        return [PaintingStep(
            step_number=start_step,
            title=title,
            instruction=f"""Time to paint {region}.

Load your brush with {color['bob_name']}.
Paint these shapes with confident strokes.
Think about the forms you're creating - give them life and character.""",
            brush=BrushType.TWO_INCH,
            motion=StrokeMotion.LOAD_AND_PULL,
            colors=[color['bob_name']],
            color_rgbs=[color['rgb']],
            canvas_region=region,
            region_mask=layer['mask'],
            technique_tip="Be confident with your strokes - trust your vision.",
            encouragement="That's it! You're creating wonderful shapes.",
            duration_hint="2-3 minutes"
        )]

    def _generate_detail_layer_steps(self, layer: Dict, start_step: int) -> List[PaintingStep]:
        """Generic detail layer (fine elements)."""
        colors = [c for c in self.palette if sum(c['rgb']) > 200]
        color = colors[0] if colors else self.palette[-2]

        title = layer.get('description', layer['name']).title()
        region = layer.get('description', layer['name'])

        return [PaintingStep(
            step_number=start_step,
            title=title,
            instruction=f"""Let's add detail to {region}.

With {color['bob_name']}, add the finer elements.
Use a lighter touch here - details should enhance, not overwhelm.
Add texture, character, personality - these little touches bring it all to life.""",
            brush=BrushType.ONE_INCH,
            motion=StrokeMotion.TAP,
            colors=[color['bob_name']],
            color_rgbs=[color['rgb']],
            canvas_region=region,
            region_mask=layer['mask'],
            technique_tip="Details are subtle - less is often more.",
            encouragement="Perfect! Those details really make it special.",
            duration_hint="2-4 minutes"
        )]

    def _generate_glaze_layer_steps(self, layer: Dict, start_step: int) -> List[PaintingStep]:
        """Generic glaze layer (transparent color washes)."""
        colors = [c for c in self.palette if sum(c['rgb']) > 300]
        color = colors[0] if colors else self.palette[-2]

        title = layer.get('description', layer['name']).title()
        region = layer.get('description', layer['name'])

        return [PaintingStep(
            step_number=start_step,
            title=title,
            instruction=f"""Now for a beautiful glaze over {region}.

Take just a tiny bit of {color['bob_name']} - very thin.
Lightly glaze over this area with barely-there strokes.
We're adding a transparent wash of color - like colored light falling on the painting.""",
            brush=BrushType.FAN_BRUSH,
            motion=StrokeMotion.TAP,
            colors=[color['bob_name']],
            color_rgbs=[color['rgb']],
            canvas_region=region,
            region_mask=layer['mask'],
            technique_tip="Glazes should be thin and transparent - build up slowly.",
            encouragement="See how that adds such depth and glow!",
            duration_hint="1-2 minutes"
        )]

    def _generate_highlight_layer_steps(self, layer: Dict, start_step: int) -> List[PaintingStep]:
        """Generic highlight layer (bright accents)."""
        brightest = [c for c in self.palette if sum(c['rgb']) > 500]
        color = brightest[0] if brightest else self.palette[-1]

        title = layer.get('description', layer['name']).title()
        region = layer.get('description', layer['name'])

        return [PaintingStep(
            step_number=start_step,
            title=title,
            instruction=f"""Let's add highlights to {region}.

With just a touch of {color['bob_name']}, add bright accents.
Find where the light catches - just a tap here and there.
These highlights make everything shine.""",
            brush=BrushType.FAN_BRUSH,
            motion=StrokeMotion.TAP,
            colors=[color['bob_name']],
            color_rgbs=[color['rgb']],
            canvas_region=region,
            region_mask=layer['mask'],
            technique_tip="Highlights are delicate - one or two touches can transform the whole area.",
            encouragement="Beautiful! Look at how that light dances!",
            duration_hint="1-2 minutes"
        )]

    def _create_signing_step(self, step_num: int) -> PaintingStep:
        """Create the final signing step."""
        return PaintingStep(
            step_number=step_num,
            title="Sign Your Masterpiece",
            instruction=f"""And now, the most important part - signing your work!

Take your liner brush, load it with a thin paint (I like {PAINT_NAMES['red']}).
Find a corner that won't distract from your painting.
Sign your name with pride.

Step back and look at what you've created.
You did this. You made this beautiful painting.
And I'm so proud of you.""",
            brush=BrushType.LINER_BRUSH,
            motion=StrokeMotion.LOAD_AND_PULL,
            colors=[PAINT_NAMES['red']],
            color_rgbs=[(227, 38, 54)],
            canvas_region="bottom corner",
            region_mask=None,
            technique_tip="Thin your paint with medium for smooth, flowing lines.",
            encouragement="From all of us here, happy painting, and God bless.",
            duration_hint="30 seconds"
        )

    def generate_substeps_for_checkpoint(self, checkpoint: PaintingStep,
                                         previous_canvas: np.ndarray = None,
                                         target_canvas: np.ndarray = None,
                                         pbn: 'PaintByNumbers' = None) -> List[Dict]:
        """
        Break down a checkpoint into granular substeps showing incremental color addition.

        Analyzes what colors are added between the previous checkpoint and this checkpoint,
        and creates substeps for each color layer.

        Args:
            checkpoint: A PaintingStep (checkpoint) to break down
            previous_canvas: Canvas state after previous checkpoint (RGB)
            target_canvas: Canvas state after this checkpoint (RGB)
            pbn: PaintByNumbers object with quantized colors

        Returns:
            List of substep dictionaries showing incremental build-up
        """
        substeps = []

        # If we don't have the canvas states, fall back to simple color-based breakdown
        if previous_canvas is None or target_canvas is None or pbn is None:
            # Simple fallback: one substep per color in the checkpoint
            for i, (color_name, color_rgb) in enumerate(zip(checkpoint.colors, checkpoint.color_rgbs), 1):
                substeps.append({
                    'substep_id': f"{checkpoint.step_number}.{i}",
                    'color_name': color_name,
                    'color_rgb': color_rgb,
                    'area_name': checkpoint.canvas_region,
                    'brush': checkpoint.brush.value,
                    'motion': checkpoint.motion.value,
                    'instruction': f"Apply {color_name} to {checkpoint.canvas_region} using {checkpoint.motion.value}.",
                    'duration': checkpoint.duration_hint
                })
            return substeps

        # Advanced: Identify all new colors added in this checkpoint
        # by comparing previous canvas state to target canvas state
        new_colors = self._identify_colors_added(previous_canvas, target_canvas, pbn)

        if not new_colors:
            # No new colors detected, create single substep
            substeps.append({
                'substep_id': f"{checkpoint.step_number}.1",
                'color_name': checkpoint.colors[0] if checkpoint.colors else "paint",
                'color_rgb': checkpoint.color_rgbs[0] if checkpoint.color_rgbs else (128, 128, 128),
                'area_name': checkpoint.canvas_region,
                'brush': checkpoint.brush.value,
                'motion': checkpoint.motion.value,
                'instruction': f"Apply {checkpoint.colors[0] if checkpoint.colors else 'paint'} to {checkpoint.canvas_region} using {checkpoint.motion.value}.",
                'duration': checkpoint.duration_hint
            })
            return substeps

        # Create substeps for each color layer (ordered by appearance)
        for i, color_info in enumerate(new_colors, 1):
            color_rgb = color_info['rgb']
            color_name = self._get_bob_ross_color_name(np.array(color_rgb))
            area_desc = self._describe_color_area(color_info['mask'], checkpoint.canvas_region)

            substeps.append({
                'substep_id': f"{checkpoint.step_number}.{i}",
                'color_name': color_name,
                'color_rgb': color_rgb,
                'area_name': area_desc,
                'coverage_percent': color_info['coverage_percent'],
                'mask': color_info['mask'],
                'brush': checkpoint.brush.value,
                'motion': checkpoint.motion.value,
                'instruction': f"Apply {color_name} to {area_desc} using {checkpoint.motion.value}.",
                'duration': self._estimate_substep_duration(checkpoint.duration_hint, len(new_colors), i)
            })

        return substeps

    def _identify_colors_added(self, previous_canvas: np.ndarray,
                               target_canvas: np.ndarray,
                               pbn: 'PaintByNumbers') -> List[Dict]:
        """
        Identify what colors were added between two canvas states.

        Args:
            previous_canvas: Canvas before (RGB)
            target_canvas: Canvas after (RGB)
            pbn: PaintByNumbers with quantized colors

        Returns:
            List of color info dicts with 'rgb', 'mask', 'coverage_percent'
        """
        # Find pixels that changed significantly
        diff = np.linalg.norm(target_canvas.astype(float) - previous_canvas.astype(float), axis=2)
        changed_mask = diff > 30  # Threshold for "changed"

        if np.sum(changed_mask) < 100:
            return []

        # Use the quantized label map to identify distinct color regions
        color_regions = []

        # For each palette color, check if it appears in the changed region
        for color_idx, palette_color in enumerate(pbn.palette):
            # Find pixels in the label map that match this color AND changed
            color_in_target = (pbn.label_map == color_idx)
            color_mask = color_in_target & changed_mask

            coverage = np.sum(color_mask)
            if coverage > 100:  # Minimum pixels for a color layer
                # Get the actual RGB from the palette (quantized color)
                color_rgb = tuple(int(c) for c in palette_color)

                # Calculate position metrics for ordering
                y_coords, x_coords = np.where(color_mask)
                avg_y = np.mean(y_coords) / self.height if len(y_coords) > 0 else 0.5

                color_regions.append({
                    'rgb': color_rgb,
                    'mask': color_mask,
                    'coverage_percent': coverage / (self.height * self.width) * 100,
                    'avg_luminosity': np.mean(palette_color),
                    'avg_y_position': avg_y  # 0 = top, 1 = bottom
                })

        # Sort by vertical position (top to bottom = background to foreground for landscapes)
        # This gives natural painting order: sky -> mountains -> midground -> foreground
        color_regions.sort(key=lambda c: c['avg_y_position'])

        return color_regions

    def _describe_color_area(self, mask: np.ndarray, general_region: str) -> str:
        """
        Describe where a color appears on the canvas.

        Args:
            mask: Boolean mask of pixels
            general_region: General description like "the middle-tone areas"

        Returns:
            Human-readable area description
        """
        if np.sum(mask) < 100:
            return general_region

        # Find center of mass
        y_coords, x_coords = np.where(mask)
        if len(y_coords) == 0:
            return general_region

        center_y = np.mean(y_coords) / self.height
        center_x = np.mean(x_coords) / self.width

        # Determine vertical position
        if center_y < 0.33:
            vertical = "upper"
        elif center_y > 0.67:
            vertical = "lower"
        else:
            vertical = "middle"

        # Determine horizontal position
        if center_x < 0.33:
            horizontal = "left"
        elif center_x > 0.67:
            horizontal = "right"
        else:
            horizontal = "center"

        # Check coverage extent
        y_extent = (np.max(y_coords) - np.min(y_coords)) / self.height
        x_extent = (np.max(x_coords) - np.min(x_coords)) / self.width

        if y_extent > 0.6 and x_extent > 0.6:
            return f"the {general_region}"
        elif y_extent > 0.4 or x_extent > 0.4:
            return f"the {vertical} {horizontal} area"
        else:
            return f"the {vertical}-{horizontal}"

    def _estimate_substep_duration(self, total_duration: str, num_substeps: int, substep_index: int) -> str:
        """Estimate duration for a single substep based on total checkpoint duration."""
        # Simple heuristic: split the time roughly equally
        if "30 seconds" in total_duration:
            return "15-20 seconds"
        elif "1-2 minutes" in total_duration:
            if num_substeps == 2:
                return "30-60 seconds"
            return "20-40 seconds"
        elif "2-3 minutes" in total_duration:
            if num_substeps == 2:
                return "1-1.5 minutes"
            return "45-60 seconds"
        elif "3-5 minutes" in total_duration:
            if num_substeps == 2:
                return "1.5-2.5 minutes"
            return "1-2 minutes"
        else:
            return total_duration

    def export_steps_json(self, output_path: str, include_substeps: bool = False,
                          progress_images: List[np.ndarray] = None):
        """Export steps to JSON file."""
        if not self.steps:
            self.generate_steps()

        export_data = {
            'image': self.image_path,
            'pattern': self.analysis.dominant_pattern,
            'total_steps': len(self.steps),
            'estimated_time': f"{len(self.steps) * 3} - {len(self.steps) * 5} minutes",
            'materials': self._get_materials_list(),
            'steps': []
        }

        # Get PBN for substep generation
        pbn = None
        if include_substeps and progress_images:
            from ...core.paint_by_numbers import PaintByNumbers
            pbn = PaintByNumbers(self.image_path, self.n_colors)
            pbn.quantize_colors()
            pbn.create_regions()

        for idx, s in enumerate(self.steps):
            step_dict = {
                'step': s.step_number,
                'title': s.title,
                'instruction': s.instruction,
                'brush': s.brush.value,
                'motion': s.motion.value,
                'colors': s.colors,
                'color_rgbs': s.color_rgbs,
                'canvas_region': s.canvas_region,
                'technique_tip': s.technique_tip,
                'encouragement': s.encouragement,
                'duration': s.duration_hint
            }

            # Add substeps if requested
            if include_substeps:
                if progress_images and pbn:
                    # Use canvas-state-aware generation
                    previous_canvas = progress_images[idx] if idx < len(progress_images) else None
                    target_canvas = progress_images[idx + 1] if idx + 1 < len(progress_images) else None
                    substeps = self.generate_substeps_for_checkpoint(
                        s,
                        previous_canvas=previous_canvas,
                        target_canvas=target_canvas,
                        pbn=pbn
                    )
                else:
                    # Fallback to simple generation
                    substeps = self.generate_substeps_for_checkpoint(s)

                # Remove masks from JSON (not serializable)
                for substep in substeps:
                    if 'mask' in substep:
                        del substep['mask']

                step_dict['substeps'] = substeps

            export_data['steps'].append(step_dict)

        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)

        print(f"Steps exported to {output_path}")
        return export_data

    def _get_materials_list(self) -> Dict:
        """Get required materials list."""
        brushes = set()
        colors = set()

        for step in self.steps:
            brushes.add(step.brush.value)
            colors.update(step.colors)

        return {
            'brushes': list(brushes),
            'paints': list(colors),
            'other': [
                'Canvas (any size)',
                'Palette',
                'Paper towels',
                'Odorless thinner (for oils) or water (for acrylics)',
                'Easel (optional but recommended)'
            ]
        }

    def create_step_by_step_guide(self, output_dir: str, include_substeps: bool = False):
        """Create a visual step-by-step guide."""
        if not self.steps:
            self.generate_steps()

        os.makedirs(output_dir, exist_ok=True)

        # Generate progress images showing what painting looks like after each step
        progress_images = self._generate_progress_images()

        # Create individual step images with progress preview
        for i, step in enumerate(self.steps):
            progress_img = progress_images[i] if i < len(progress_images) else progress_images[-1]
            self._create_step_image_with_progress(step, progress_img, output_dir)

        # Create combined guide
        self._create_combined_guide_with_progress(output_dir, progress_images)

        # Export JSON
        self.export_steps_json(
            os.path.join(output_dir, 'bob_ross_steps.json'),
            include_substeps=include_substeps,
            progress_images=progress_images if include_substeps else None
        )

        # If including substeps, create detailed guides with visual aids
        if include_substeps:
            self._create_substep_text_guide(output_dir, progress_images)
            self._generate_substep_images(output_dir, progress_images)

        print(f"\nBob Ross style guide created in {output_dir}/")

    def _generate_progress_images(self) -> List[np.ndarray]:
        """
        Generate images showing what the painting will actually look like after each step.
        Uses quantized colors to show the reality of painting with limited colors.
        """
        progress_images = []
        h, w = self.height, self.width

        # Get quantized image - this is what your painting will ACTUALLY look like
        from ...core.paint_by_numbers import PaintByNumbers
        pbn = PaintByNumbers(self.image_path, self.n_colors)
        pbn.quantize_colors()
        quantized_rgb = pbn.quantized_image  # Limited color palette

        # Step 0: Blank canvas (white)
        blank = np.ones((h, w, 3), dtype=np.uint8) * 255
        progress_images.append(blank.copy())

        # Step 1: Base coat (adapt to image type)
        if self.analysis.has_dark_background:
            # Dark base for night/aurora scenes
            canvas = np.ones((h, w, 3), dtype=np.uint8) * 20
        else:
            # Light base for daylight scenes
            canvas = np.ones((h, w, 3), dtype=np.uint8) * 200
        progress_images.append(canvas.copy())

        # Build up the painting by progressively revealing more luminosity levels
        # For aurora: add color regions from dark to bright (cumulative)
        if self.analysis.dominant_pattern == 'aurora':
            # Get luminosity map
            lab = cv2.cvtColor(self.original, cv2.COLOR_BGR2LAB)
            luminosity = lab[:, :, 0].astype(np.float32) / 255.0

            # Cumulative luminosity thresholds - each step reveals MORE of the image
            # Step 2: Just darkest visible areas (lum 0.05-0.15)
            # Step 3: + first aurora colors (up to 0.30)
            # Step 4: + more aurora (up to 0.50)
            # Step 5: + brightest aurora (up to 0.75)
            # Step 6: + silhouettes (darkest pixels)
            lum_thresholds = [0.15, 0.30, 0.50, 0.75]

            for threshold in lum_thresholds:
                # Reveal pixels with luminosity between 0.05 and threshold
                # (skip the very darkest which are silhouettes)
                mask = (luminosity >= 0.05) & (luminosity <= threshold)
                step_canvas = canvas.copy()
                for c in range(3):
                    step_canvas[:, :, c] = np.where(
                        mask,
                        self.original_rgb[:, :, c],
                        step_canvas[:, :, c]
                    )
                progress_images.append(step_canvas)
                canvas = step_canvas  # Build on previous

            # Final step: Add silhouettes (darkest pixels, lum < 0.05)
            silhouette_mask = luminosity < 0.05
            final_canvas = canvas.copy()
            for c in range(3):
                final_canvas[:, :, c] = np.where(
                    silhouette_mask,
                    self.original_rgb[:, :, c],
                    final_canvas[:, :, c]
                )
            progress_images.append(final_canvas)

            print(f"  Generated progress image {len(progress_images)}/{len(progress_images)}")
        else:
            # Standard layer-by-layer for non-aurora images
            if hasattr(self, 'layers') and self.layers:
                for i, layer in enumerate(self.layers):
                    if layer.get('mask') is not None:
                        for c in range(3):
                            canvas[:, :, c] = np.where(
                                layer['mask'],
                                self.original_rgb[:, :, c],
                                canvas[:, :, c]
                            )
                    progress_images.append(canvas.copy())

                    if (i + 1) % 5 == 0:
                        print(f"  Generated progress image {len(progress_images)}/{len(self.layers) + 3}")

        # Final step: Complete quantized painting
        if len(progress_images) == 0 or not np.array_equal(progress_images[-1], quantized_rgb):
            progress_images.append(quantized_rgb.copy())

        print(f"\nGenerated {len(progress_images)} progress images (showing {self.n_colors}-color painting reality)")
        return progress_images

    def _create_step_image_with_progress(self, step: PaintingStep, progress_img: np.ndarray, output_dir: str):
        """Create a step image that includes a progress preview."""
        width = 900
        height = 700

        img = Image.new('RGB', (width, height), '#2C2C2C')
        draw = ImageDraw.Draw(img)

        try:
            font_title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
            font_body = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 13)
            font_tip = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 11)
        except:
            try:
                font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
                font_body = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
                font_tip = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf", 11)
            except:
                font_title = font_body = font_tip = ImageFont.load_default()

        margin = 25
        y = margin

        # Progress image on the right side
        preview_size = 280
        progress_pil = Image.fromarray(progress_img)
        progress_pil.thumbnail((preview_size, preview_size), Image.Resampling.LANCZOS)

        preview_x = width - preview_size - margin
        preview_y = margin + 30

        # Border around preview
        draw.rectangle([preview_x - 3, preview_y - 3,
                       preview_x + progress_pil.width + 3, preview_y + progress_pil.height + 3],
                      outline='#FFD700', width=2)
        img.paste(progress_pil, (preview_x, preview_y))

        # Label for preview
        draw.text((preview_x, preview_y + progress_pil.height + 8),
                 "After this step:", fill='#FFD700', font=font_tip)

        # Text content on the left (narrower to make room for preview)
        text_width = width - preview_size - margin * 3

        # Step number and title
        title = f"Step {step.step_number}: {step.title}"
        draw.text((margin, y), title, fill='#FFD700', font=font_title)
        y += 35

        # Duration
        draw.text((margin, y), f"Time: {step.duration_hint}", fill='#888888', font=font_tip)
        y += 22

        # Brush and motion
        draw.text((margin, y), f"Brush: {step.brush.value}", fill='#87CEEB', font=font_body)
        y += 18
        draw.text((margin, y), f"Motion: {step.motion.value}", fill='#87CEEB', font=font_body)
        y += 25

        # Color swatches
        draw.text((margin, y), "Colors:", fill='white', font=font_body)
        x = margin + 55
        for i, (name, rgb) in enumerate(zip(step.colors, step.color_rgbs)):
            draw.rectangle([x, y-2, x + 22, y + 16], fill=rgb, outline='white')
            x += 27
        y += 28

        # Instructions (word wrapped to fit left side)
        draw.text((margin, y), "Instructions:", fill='white', font=font_body)
        y += 20

        instructions = step.instruction.replace('\n\n', '\n')
        lines = []
        for para in instructions.split('\n'):
            words = para.split()
            current = ""
            for word in words:
                test_line = current + word + " "
                # Narrower wrap for left column
                if len(current + word) < 50:
                    current = test_line
                else:
                    lines.append(current.strip())
                    current = word + " "
            if current:
                lines.append(current.strip())
            lines.append("")

        for line in lines[:18]:
            draw.text((margin + 10, y), line, fill='#CCCCCC', font=font_body)
            y += 16

        # Tip box at bottom
        y = height - 90
        draw.rectangle([margin, y, width - margin, y + 32], fill='#1a3a1a', outline='#2d5a2d')
        tip_text = step.technique_tip[:80] + "..." if len(step.technique_tip) > 80 else step.technique_tip
        draw.text((margin + 8, y + 7), f"Tip: {tip_text}", fill='#90EE90', font=font_tip)

        # Encouragement
        y += 42
        draw.text((margin, y), f'"{step.encouragement}"', fill='#FFD700', font=font_tip)

        # Save
        img.save(os.path.join(output_dir, f'step_{step.step_number:02d}.png'))

    def _create_combined_guide_with_progress(self, output_dir: str, progress_images: List[np.ndarray]):
        """Create a combined guide showing all steps with progress images."""
        # Create a filmstrip-style progress view
        n_steps = len(self.steps)
        thumb_size = 150
        margin = 15

        # Layout: progress strip on top, then step details below
        strip_height = thumb_size + 60
        detail_rows = (n_steps + 3) // 4
        total_height = strip_height + detail_rows * 100 + margin * 2 + 50

        width = max(900, n_steps * (thumb_size + margin) + margin * 2)

        img = Image.new('RGB', (width, total_height), '#1a1a1a')
        draw = ImageDraw.Draw(img)

        try:
            font_title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
            font_body = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 11)
        except:
            font_title = font_body = ImageFont.load_default()

        # Title
        draw.text((margin, margin), "PAINTING PROGRESS - Step by Step", fill='#FFD700', font=font_title)

        # Progress strip
        y = margin + 35
        for i, progress in enumerate(progress_images[:n_steps]):
            x = margin + i * (thumb_size + margin)

            # Thumbnail
            thumb = Image.fromarray(progress)
            thumb.thumbnail((thumb_size, thumb_size), Image.Resampling.LANCZOS)

            # Border
            draw.rectangle([x-2, y-2, x + thumb.width + 2, y + thumb.height + 2],
                          outline='#444', width=1)
            img.paste(thumb, (x, y))

            # Step number
            draw.text((x + thumb_size//2 - 15, y + thumb.height + 5),
                     f"Step {i+1}", fill='white', font=font_body)

        # Arrow indicators between steps
        for i in range(len(progress_images[:n_steps]) - 1):
            x = margin + (i + 1) * (thumb_size + margin) - margin//2
            draw.text((x - 5, y + thumb_size//2 - 5), "->", fill='#FFD700', font=font_body)

        # Step summary below
        y = strip_height + 20
        draw.text((margin, y), "STEP SUMMARY:", fill='white', font=font_title)
        y += 30

        cols = 4
        col_width = (width - margin * 2) // cols

        for i, step in enumerate(self.steps):
            col = i % cols
            row = i // cols

            x = margin + col * col_width
            cy = y + row * 90

            # Step box
            draw.rectangle([x, cy, x + col_width - 10, cy + 80],
                          outline='#333', fill='#252525')

            # Step title
            draw.text((x + 5, cy + 5), f"Step {step.step_number}", fill='#FFD700', font=font_body)

            # Title (truncated)
            title = step.title[:22] + "..." if len(step.title) > 22 else step.title
            draw.text((x + 5, cy + 22), title, fill='white', font=font_body)

            # Brush
            draw.text((x + 5, cy + 40), step.brush.value[:20], fill='#87CEEB', font=font_body)

            # Duration
            draw.text((x + 5, cy + 56), step.duration_hint, fill='#666', font=font_body)

        # Save progress strip separately too
        img.save(os.path.join(output_dir, 'guide_overview.png'))

        # Also save individual progress images
        for i, progress in enumerate(progress_images):
            prog_img = Image.fromarray(progress)
            prog_img.save(os.path.join(output_dir, f'progress_{i:02d}.png'))

        print(f"Saved {len(progress_images)} progress images")

    def _create_substep_text_guide(self, output_dir: str, progress_images: List[np.ndarray]):
        """Create a detailed text guide with checkpoints and substeps."""
        output_path = os.path.join(output_dir, 'bob_ross_substeps.txt')

        # Get PBN for canvas-state-aware generation
        from ...core.paint_by_numbers import PaintByNumbers
        pbn = PaintByNumbers(self.image_path, self.n_colors)
        pbn.quantize_colors()
        pbn.create_regions()

        with open(output_path, 'w') as f:
            f.write("BOB ROSS STYLE PAINTING GUIDE - WITH SUBSTEPS\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Image: {os.path.basename(self.image_path)}\n")
            f.write(f"Pattern: {self.analysis.dominant_pattern.upper()}\n")
            f.write(f"Total Checkpoints: {len(self.steps)}\n")
            f.write(f"Estimated Time: {len(self.steps) * 3} - {len(self.steps) * 5} minutes\n\n")

            # Materials
            materials = self._get_materials_list()
            f.write("MATERIALS NEEDED:\n")
            f.write("-" * 80 + "\n\n")
            f.write("Brushes:\n")
            for brush in materials['brushes']:
                f.write(f"  - {brush}\n")
            f.write("\nPaints:\n")
            for paint in materials['paints']:
                f.write(f"  - {paint}\n")
            f.write("\nOther:\n")
            for item in materials['other']:
                f.write(f"  - {item}\n")
            f.write("\n\n")

            # Each checkpoint with substeps
            for idx, step in enumerate(self.steps):
                f.write("=" * 80 + "\n")
                f.write(f"CHECKPOINT {step.step_number}: {step.title.upper()}\n")
                f.write("=" * 80 + "\n\n")

                # Overview (original Bob Ross instruction)
                f.write("Overview:\n")
                f.write(f"{step.instruction}\n\n")

                # Technique tip
                if step.technique_tip:
                    f.write(f"Technique Tip: {step.technique_tip}\n\n")

                # Substeps with canvas-state analysis
                previous_canvas = progress_images[idx] if idx < len(progress_images) else None
                target_canvas = progress_images[idx + 1] if idx + 1 < len(progress_images) else None

                substeps = self.generate_substeps_for_checkpoint(
                    step,
                    previous_canvas=previous_canvas,
                    target_canvas=target_canvas,
                    pbn=pbn
                )

                f.write(f"Granular Substeps ({len(substeps)}):\n")
                f.write("-" * 80 + "\n\n")

                for substep in substeps:
                    f.write(f"Substep {substep['substep_id']}\n")
                    f.write(f"  Color: {substep['color_name']}\n")
                    f.write(f"  Area: {substep['area_name']}\n")
                    if 'coverage_percent' in substep:
                        f.write(f"  Coverage: {substep['coverage_percent']:.1f}%\n")
                    f.write(f"  Brush: {substep['brush']}\n")
                    f.write(f"  Motion: {substep['motion']}\n")
                    f.write(f"  \n")
                    f.write(f"  {substep['instruction']}\n")
                    f.write(f"\n")

                # Encouragement
                if step.encouragement:
                    f.write(f"💬 {step.encouragement}\n\n")

                # Progress image reference
                f.write(f"📸 See progress_{step.step_number:02d}.png for cumulative result\n\n")

            f.write("=" * 80 + "\n")
            f.write("PAINTING COMPLETE!\n")
            f.write("=" * 80 + "\n")

        print(f"Substep text guide saved to {output_path}")

    def _generate_substep_images(self, output_dir: str, progress_images: List[np.ndarray]):
        """Generate cumulative and isolated images for each substep."""
        print("Generating substep images...")

        # Get color quantization data
        from ...core.paint_by_numbers import PaintByNumbers
        pbn = PaintByNumbers(self.image_path, self.n_colors)
        pbn.quantize_colors()
        pbn.create_regions()

        # Initialize cumulative canvas (starts with first progress image)
        cumulative_canvas = progress_images[0].copy() if progress_images else np.ones((self.height, self.width, 3), dtype=np.uint8) * 255

        substep_count = 0

        # Go through each checkpoint and its substeps
        # Skip the last step if it's the signing step (no actual painting)
        steps_to_process = self.steps[:-1] if len(self.steps) > 0 and 'sign' in self.steps[-1].title.lower() else self.steps

        for idx, checkpoint in enumerate(steps_to_process):
            # Reset cumulative canvas to the actual progress image at this checkpoint
            # This ensures cumulative substeps build from the correct starting point
            if idx < len(progress_images):
                cumulative_canvas = progress_images[idx].copy()

            # Get previous and target canvas states
            previous_canvas = progress_images[idx] if idx < len(progress_images) else cumulative_canvas.copy()
            target_canvas = progress_images[idx + 1] if idx + 1 < len(progress_images) else self.original_rgb.copy()

            # Generate substeps with canvas state analysis
            substeps = self.generate_substeps_for_checkpoint(
                checkpoint,
                previous_canvas=previous_canvas,
                target_canvas=target_canvas,
                pbn=pbn
            )

            # Generate images for each substep
            for substep in substeps:
                substep_id = substep['substep_id']
                color_rgb = substep['color_rgb']

                # Use the mask from substep if available, otherwise find it
                if 'mask' in substep and substep['mask'] is not None:
                    color_mask = substep['mask']
                else:
                    # Fallback to old method
                    checkpoint_mask = checkpoint.region_mask if checkpoint.region_mask is not None else np.ones((self.height, self.width), dtype=bool)
                    color_mask = self._find_color_in_image(pbn, color_rgb, checkpoint_mask)

                # Generate isolated image (just this color on white background)
                isolated = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255
                for c in range(3):
                    isolated[:, :, c] = np.where(color_mask, color_rgb[c], 255)

                # Update cumulative canvas
                for c in range(3):
                    cumulative_canvas[:, :, c] = np.where(
                        color_mask,
                        color_rgb[c],
                        cumulative_canvas[:, :, c]
                    )

                # Save both images
                isolated_path = os.path.join(output_dir, f'substep_{substep_id}_isolated.png')
                cumulative_path = os.path.join(output_dir, f'substep_{substep_id}_cumulative.png')

                cv2.imwrite(isolated_path, cv2.cvtColor(isolated, cv2.COLOR_RGB2BGR))
                cv2.imwrite(cumulative_path, cv2.cvtColor(cumulative_canvas, cv2.COLOR_RGB2BGR))

                substep_count += 1

        print(f"Generated {substep_count * 2} substep images ({substep_count} isolated + {substep_count} cumulative)")

    def _find_color_in_image(self, pbn: 'PaintByNumbers', target_rgb: Tuple[int, int, int],
                             region_mask: np.ndarray) -> np.ndarray:
        """
        Find pixels that match the target color within a region.

        Args:
            pbn: PaintByNumbers object with quantized image
            target_rgb: Target color as (R, G, B)
            region_mask: Boolean mask of valid region

        Returns:
            Boolean mask of pixels matching the color
        """
        # Find closest palette color
        target = np.array(target_rgb)
        min_dist = float('inf')
        closest_idx = 0

        for idx, palette_color in enumerate(pbn.palette):
            dist = np.linalg.norm(target - palette_color)
            if dist < min_dist:
                min_dist = dist
                closest_idx = idx

        # Get mask for this color within the region
        color_mask = (pbn.label_map == closest_idx) & region_mask

        # If very few pixels found, use a broader threshold
        if np.sum(color_mask) < 100:
            # Use color distance threshold instead
            img_colors = self.original_rgb.reshape(-1, 3)
            distances = np.linalg.norm(img_colors - target, axis=1)
            dist_mask = (distances < 50).reshape(self.height, self.width)
            color_mask = dist_mask & region_mask

        return color_mask

    # =========================================================================
    # GRANULAR LAYER/SUBSTEP GENERATION
    # =========================================================================

    def generate_layers(self) -> List[PaintingLayer]:
        """
        Generate granular layers with substeps broken down by color and area.

        Returns a list of PaintingLayer objects, each containing multiple
        PaintingSubstep objects for atomic painting actions.
        """
        layers = []

        # Get the quantized image data for color analysis
        from ...core.paint_by_numbers import PaintByNumbers
        self.pbn = PaintByNumbers(self.image_path, self.n_colors)
        self.pbn.quantize_colors()
        self.pbn.create_regions()

        # Analyze luminosity for layer boundaries
        lab = cv2.cvtColor(self.original, cv2.COLOR_BGR2LAB)
        luminosity = lab[:, :, 0].astype(np.float32) / 255.0

        # Define layer masks based on luminosity
        layer_defs = []

        # Layer 1: Dark Background (if present)
        if self.analysis.has_dark_background:
            dark_mask = luminosity < 0.25
            if np.sum(dark_mask) > 0.1 * self.height * self.width:
                layer_defs.append({
                    'name': 'Dark Background',
                    'mask': dark_mask,
                    'technique': 'base coat',
                    'overview': "Start with a dark base to make colors pop."
                })

        # Layer 2: Mid-tones
        mid_mask = (luminosity >= 0.25) & (luminosity < 0.55)
        if np.sum(mid_mask) > 0.05 * self.height * self.width:
            layer_defs.append({
                'name': 'Mid-Tone Foundation',
                'mask': mid_mask,
                'technique': 'blend',
                'overview': "Build the middle values that connect dark and light."
            })

        # Layer 3: Bright areas / Aurora
        bright_mask = (luminosity >= 0.55) & (luminosity < 0.8)
        if np.sum(bright_mask) > 0.03 * self.height * self.width:
            layer_defs.append({
                'name': 'Bright Areas',
                'mask': bright_mask,
                'technique': 'glazing',
                'overview': "Add the glowing, luminous colors."
            })

        # Layer 4: Highlights (brightest spots)
        highlight_mask = luminosity >= 0.8
        if np.sum(highlight_mask) > 0.01 * self.height * self.width:
            layer_defs.append({
                'name': 'Highlights',
                'mask': highlight_mask,
                'technique': 'accent',
                'overview': "Add the brightest highlights that make it shine."
            })

        # Layer 5: Foreground silhouettes (dark objects in lower half)
        if self.analysis.has_dark_background:
            fg_mask = self._detect_foreground_mask(luminosity)
            if fg_mask is not None and np.sum(fg_mask) > 0.01 * self.height * self.width:
                layer_defs.append({
                    'name': 'Foreground Silhouettes',
                    'mask': fg_mask,
                    'technique': 'silhouette',
                    'overview': "Add dark foreground shapes that frame the scene."
                })

        # Generate substeps for each layer
        for layer_num, layer_def in enumerate(layer_defs, 1):
            layer = PaintingLayer(
                layer_number=layer_num,
                name=layer_def['name'],
                overview=layer_def['overview'],
                dry_time=self._get_dry_time(layer_def['technique']),
                technique_tip=self._get_technique_tip(layer_def['technique']),
                encouragement=ENCOURAGEMENTS[layer_num % len(ENCOURAGEMENTS)]
            )

            # Generate substeps for colors in this layer
            substeps = self._generate_substeps_for_layer(
                layer_num,
                layer_def['mask'],
                layer_def['technique']
            )
            layer.substeps = substeps

            if substeps:  # Only add layers that have substeps
                layers.append(layer)

        # Store for later use
        self.layers = layers
        return layers

    def _detect_foreground_mask(self, luminosity: np.ndarray) -> np.ndarray:
        """Detect foreground silhouette objects (dark shapes in lower canvas)."""
        very_dark = luminosity < 0.15
        contours, _ = cv2.findContours(
            (very_dark * 255).astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        foreground_mask = np.zeros_like(very_dark, dtype=np.uint8)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 1000:
                M = cv2.moments(cnt)
                if M['m00'] > 0:
                    cy = int(M['m01'] / M['m00'])
                    if cy > self.height * 0.5:  # Lower half
                        cv2.drawContours(foreground_mask, [cnt], -1, 1, -1)

        if np.sum(foreground_mask) > 0:
            return foreground_mask.astype(bool)
        return None

    def _generate_substeps_for_layer(
        self,
        layer_num: int,
        layer_mask: np.ndarray,
        technique: str
    ) -> List[PaintingSubstep]:
        """
        Generate granular substeps for a layer, broken down by color and area.

        For each color that appears in this layer:
        1. Find where it appears on the canvas
        2. Break into distinct regions (upper-left, center, etc.)
        3. Create a substep for each color+region combination
        """
        substeps = []
        substep_count = 0

        # Get the label map from paint_by_numbers
        label_map = self.pbn.label_map

        # Find which colors appear in this layer
        colors_in_layer = []
        for color_idx, color_info in enumerate(self.palette):
            color_mask = label_map == color_idx
            overlap = color_mask & layer_mask
            pixel_count = np.sum(overlap)

            if pixel_count > 100:  # Minimum threshold
                coverage = pixel_count / (self.height * self.width)
                colors_in_layer.append({
                    'idx': color_idx,
                    'info': color_info,
                    'mask': overlap,
                    'coverage': coverage
                })

        # Sort by coverage (largest areas first - paint big areas first)
        colors_in_layer.sort(key=lambda x: -x['coverage'])

        # For each color, find distinct canvas regions
        for color_data in colors_in_layer:
            regions = self._find_color_regions(color_data['mask'])

            for region in regions:
                substep_count += 1
                substep = PaintingSubstep(
                    substep_id=f"{layer_num}.{substep_count}",
                    color_name=color_data['info']['bob_name'],
                    color_rgb=color_data['info']['rgb'],
                    brush=self._get_brush_for_coverage(region.coverage_percent),
                    motion=self._get_motion_for_technique(technique),
                    technique=technique,
                    area=region,
                    instruction=self._generate_substep_instruction(
                        color_data['info']['bob_name'],
                        region,
                        technique
                    ),
                    duration_hint=self._estimate_duration(region.coverage_percent)
                )
                substeps.append(substep)

        return substeps

    def _find_color_regions(self, color_mask: np.ndarray) -> List[CanvasArea]:
        """
        Find distinct canvas regions where a color appears.

        Divides the canvas into a 3x3 grid and identifies which regions
        contain significant amounts of this color.
        """
        regions = []
        h, w = self.height, self.width
        total_pixels = h * w

        # Define 3x3 grid regions
        grid_names = [
            ['upper-left', 'upper-center', 'upper-right'],
            ['middle-left', 'center', 'middle-right'],
            ['lower-left', 'lower-center', 'lower-right']
        ]

        for row in range(3):
            for col in range(3):
                y1 = row / 3
                y2 = (row + 1) / 3
                x1 = col / 3
                x2 = (col + 1) / 3

                # Get pixel bounds
                py1, py2 = int(y1 * h), int(y2 * h)
                px1, px2 = int(x1 * w), int(x2 * w)

                # Check how much of this color is in this region
                region_slice = color_mask[py1:py2, px1:px2]
                pixel_count = np.sum(region_slice)

                if pixel_count > 50:  # Minimum threshold
                    coverage = pixel_count / total_pixels * 100

                    # Calculate centroid within this region
                    y_coords, x_coords = np.where(region_slice)
                    if len(y_coords) > 0:
                        cy = (np.mean(y_coords) + py1) / h
                        cx = (np.mean(x_coords) + px1) / w
                    else:
                        cy = (y1 + y2) / 2
                        cx = (x1 + x2) / 2

                    region = CanvasArea(
                        name=grid_names[row][col],
                        bounds=(y1, y2, x1, x2),
                        mask=region_slice,
                        coverage_percent=coverage,
                        centroid=(cy, cx)
                    )
                    regions.append(region)

        # Merge adjacent regions if they're small
        regions = self._merge_small_regions(regions)

        return regions

    def _merge_small_regions(self, regions: List[CanvasArea]) -> List[CanvasArea]:
        """Merge very small adjacent regions to avoid too many substeps."""
        if len(regions) <= 1:
            return regions

        # If total coverage is small, merge all into one
        total_coverage = sum(r.coverage_percent for r in regions)
        if total_coverage < 2.0:  # Less than 2% of canvas
            merged_bounds = (
                min(r.bounds[0] for r in regions),
                max(r.bounds[1] for r in regions),
                min(r.bounds[2] for r in regions),
                max(r.bounds[3] for r in regions)
            )
            return [CanvasArea(
                name=self._describe_bounds(merged_bounds),
                bounds=merged_bounds,
                coverage_percent=total_coverage,
                centroid=(
                    sum(r.centroid[0] * r.coverage_percent for r in regions) / total_coverage,
                    sum(r.centroid[1] * r.coverage_percent for r in regions) / total_coverage
                )
            )]

        return regions

    def _describe_bounds(self, bounds: Tuple[float, float, float, float]) -> str:
        """Convert bounds to human-readable description."""
        y1, y2, x1, x2 = bounds

        # Vertical position
        if y2 <= 0.4:
            vert = "upper"
        elif y1 >= 0.6:
            vert = "lower"
        else:
            vert = "middle"

        # Horizontal position
        if x2 <= 0.4:
            horiz = "left"
        elif x1 >= 0.6:
            horiz = "right"
        else:
            horiz = "center"

        if vert == "middle" and horiz == "center":
            return "center"
        elif horiz == "center":
            return f"{vert} area"
        elif vert == "middle":
            return f"{horiz} side"
        else:
            return f"{vert}-{horiz}"

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
            'accent': StrokeMotion.TAP,
            'silhouette': StrokeMotion.LOAD_AND_PULL,
        }
        return motion_map.get(technique, StrokeMotion.BLEND)

    def _get_dry_time(self, technique: str) -> str:
        """Get recommended drying time between layers."""
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
            'accent': "Less is more! Small touches make big impacts.",
            'silhouette': "Silhouettes should be solid - no texture needed here.",
        }
        return tips.get(technique, "")

    def _generate_substep_instruction(
        self,
        color_name: str,
        area: CanvasArea,
        technique: str
    ) -> str:
        """Generate a concise instruction for a single substep."""
        brush = self._get_brush_for_coverage(area.coverage_percent)
        motion = self._get_motion_for_technique(technique)

        instructions = {
            'base coat': f"Load your {brush.value} with {color_name} and cover the {area.name} using {motion.value}.",
            'blend': f"Apply {color_name} to the {area.name} with your {brush.value}, using {motion.value}.",
            'glazing': f"Lightly glaze {color_name} over the {area.name} - just a whisper of color.",
            'accent': f"Tap a small amount of {color_name} onto the {area.name} for highlights.",
            'silhouette': f"Paint {color_name} firmly in the {area.name} for solid coverage.",
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

    def export_layers_json(self, output_path: str) -> Dict:
        """Export granular layers/substeps to JSON."""
        if not hasattr(self, 'layers') or not self.layers:
            self.generate_layers()

        total_substeps = sum(len(layer.substeps) for layer in self.layers)

        export_data = {
            'image': self.image_path,
            'pattern': self.analysis.dominant_pattern,
            'total_layers': len(self.layers),
            'total_substeps': total_substeps,
            'estimated_time': f"{total_substeps * 1}-{total_substeps * 2} minutes",
            'materials': self._get_materials_list(),
            'layers': [layer.to_dict() for layer in self.layers]
        }

        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)

        print(f"Layers exported to {output_path}")
        return export_data

    def create_granular_guide(self, output_dir: str):
        """
        Create a complete granular step-by-step guide with substeps.

        Output structure:
        output_dir/
        ├── layer_01_dark_background/
        │   ├── overview.png
        │   ├── substep_1.1.png
        │   ├── substep_1.2.png
        │   └── progress.png
        ├── layer_02_mid_tones/
        │   └── ...
        ├── layers_guide.json
        └── overview.png
        """
        if not hasattr(self, 'layers') or not self.layers:
            self.generate_layers()

        os.makedirs(output_dir, exist_ok=True)
        print(f"\nGenerating granular guide to {output_dir}/")
        print("=" * 60)

        # Generate progress images for cumulative visualization
        cumulative_canvas = np.ones((self.height, self.width, 3), dtype=np.uint8) * 20

        for layer in self.layers:
            # Create layer directory
            layer_name_safe = layer.name.lower().replace(' ', '_').replace('-', '_')
            layer_dir = os.path.join(output_dir, f"layer_{layer.layer_number:02d}_{layer_name_safe}")
            os.makedirs(layer_dir, exist_ok=True)

            print(f"\nLayer {layer.layer_number}: {layer.name}")
            print(f"  Substeps: {len(layer.substeps)}")

            # Generate substep images
            for substep in layer.substeps:
                self._create_substep_image(substep, layer_dir, cumulative_canvas)

            # Update cumulative canvas after this layer
            for substep in layer.substeps:
                if substep.area.mask is not None:
                    # Apply color to cumulative canvas
                    mask = substep.area.mask
                    # Need to get full-size mask
                    full_mask = self._get_full_mask_for_substep(substep)
                    if full_mask is not None:
                        for c in range(3):
                            cumulative_canvas[:, :, c] = np.where(
                                full_mask,
                                substep.color_rgb[c],
                                cumulative_canvas[:, :, c]
                            )

            # Save layer progress
            progress_path = os.path.join(layer_dir, 'progress.png')
            cv2.imwrite(progress_path, cv2.cvtColor(cumulative_canvas, cv2.COLOR_RGB2BGR))

        # Export JSON
        self.export_layers_json(os.path.join(output_dir, 'layers_guide.json'))

        # Create overview image
        self._create_layers_overview(output_dir)

        print("\n" + "=" * 60)
        print(f"Granular guide complete! {sum(len(l.substeps) for l in self.layers)} substeps generated.")

    def _get_full_mask_for_substep(self, substep: PaintingSubstep) -> np.ndarray:
        """Get full-canvas-size mask for a substep."""
        bounds = substep.area.bounds
        y1, y2, x1, x2 = bounds
        py1, py2 = int(y1 * self.height), int(y2 * self.height)
        px1, px2 = int(x1 * self.width), int(x2 * self.width)

        full_mask = np.zeros((self.height, self.width), dtype=bool)
        if substep.area.mask is not None:
            full_mask[py1:py2, px1:px2] = substep.area.mask
        return full_mask

    def _create_substep_image(
        self,
        substep: PaintingSubstep,
        output_dir: str,
        cumulative_canvas: np.ndarray
    ):
        """Create an image showing the substep instruction and canvas region."""
        width = 800
        height = 600

        img = Image.new('RGB', (width, height), '#2C2C2C')
        draw = ImageDraw.Draw(img)

        try:
            font_title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
            font_body = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
        except:
            font_title = font_body = ImageFont.load_default()

        margin = 20
        y = margin

        # Title
        title = f"Substep {substep.substep_id}: {substep.color_name}"
        draw.text((margin, y), title, fill='#FFD700', font=font_title)
        y += 30

        # Area and technique
        draw.text((margin, y), f"Area: {substep.area.name}", fill='#87CEEB', font=font_body)
        y += 20
        draw.text((margin, y), f"Technique: {substep.technique}", fill='#87CEEB', font=font_body)
        y += 20
        draw.text((margin, y), f"Brush: {substep.brush.value}", fill='#87CEEB', font=font_body)
        y += 20
        draw.text((margin, y), f"Duration: {substep.duration_hint}", fill='#888888', font=font_body)
        y += 30

        # Color swatch
        draw.text((margin, y), "Color:", fill='white', font=font_body)
        draw.rectangle([margin + 50, y - 2, margin + 90, y + 18],
                      fill=substep.color_rgb, outline='white')
        y += 35

        # Instruction
        draw.text((margin, y), "Instruction:", fill='white', font=font_body)
        y += 20
        # Word wrap
        words = substep.instruction.split()
        lines = []
        current = ""
        for word in words:
            if len(current + word) < 55:
                current += word + " "
            else:
                lines.append(current.strip())
                current = word + " "
        if current:
            lines.append(current.strip())
        for line in lines:
            draw.text((margin + 10, y), line, fill='#CCCCCC', font=font_body)
            y += 18

        # Canvas preview with highlighted region
        preview_size = 250
        preview_x = width - preview_size - margin
        preview_y = 80

        # Create preview showing current canvas + highlighted region
        preview = cumulative_canvas.copy()
        full_mask = self._get_full_mask_for_substep(substep)

        # Highlight the region to paint
        if full_mask is not None:
            # Create a bright overlay for the region
            overlay = preview.copy()
            overlay[full_mask] = substep.color_rgb
            # Blend with pulsing effect (just show brighter)
            preview = cv2.addWeighted(preview, 0.6, overlay, 0.4, 0)
            # Add border around region
            contours, _ = cv2.findContours(
                full_mask.astype(np.uint8) * 255,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(preview, contours, -1, (255, 215, 0), 2)

        # Resize and add to image
        preview_pil = Image.fromarray(preview)
        preview_pil.thumbnail((preview_size, preview_size), Image.Resampling.LANCZOS)
        draw.rectangle([preview_x - 2, preview_y - 2,
                       preview_x + preview_pil.width + 2, preview_y + preview_pil.height + 2],
                      outline='#FFD700', width=2)
        img.paste(preview_pil, (preview_x, preview_y))

        draw.text((preview_x, preview_y + preview_pil.height + 5),
                 "Paint this area:", fill='#FFD700', font=font_body)

        # Save
        filename = f"substep_{substep.substep_id.replace('.', '_')}.png"
        img.save(os.path.join(output_dir, filename))

    def _create_layers_overview(self, output_dir: str):
        """Create an overview image showing all layers."""
        if not self.layers:
            return

        layer_count = len(self.layers)
        thumb_size = 200
        margin = 15

        width = min(1200, layer_count * (thumb_size + margin) + margin * 2)
        height = thumb_size + 150

        img = Image.new('RGB', (width, height), '#1a1a1a')
        draw = ImageDraw.Draw(img)

        try:
            font_title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
            font_body = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 11)
        except:
            font_title = font_body = ImageFont.load_default()

        draw.text((margin, margin), "PAINTING LAYERS OVERVIEW", fill='#FFD700', font=font_title)

        y = margin + 35
        for i, layer in enumerate(self.layers):
            x = margin + i * (thumb_size + margin)
            if x + thumb_size > width - margin:
                break

            # Layer box
            draw.rectangle([x, y, x + thumb_size, y + thumb_size + 40],
                          outline='#444', fill='#252525')

            # Layer title
            draw.text((x + 5, y + 5), f"Layer {layer.layer_number}", fill='#FFD700', font=font_body)
            title = layer.name[:20] + "..." if len(layer.name) > 20 else layer.name
            draw.text((x + 5, y + 20), title, fill='white', font=font_body)

            # Substep count
            draw.text((x + 5, y + 40), f"{len(layer.substeps)} substeps", fill='#87CEEB', font=font_body)
            draw.text((x + 5, y + 55), layer.total_duration, fill='#666', font=font_body)

        total_substeps = sum(len(l.substeps) for l in self.layers)
        draw.text((margin, height - 30),
                 f"Total: {layer_count} layers, {total_substeps} substeps",
                 fill='white', font=font_body)

        img.save(os.path.join(output_dir, 'overview.png'))


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Bob Ross Style Painting Guide Generator")
        print("Usage: python generator.py <image_path> [output_dir]")
        sys.exit(0)

    image_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "bob_ross_output"

    generator = BobRossGenerator(image_path)
    generator.generate_steps()
    generator.create_step_by_step_guide(output_dir)

    print("\n" + "=" * 60)
    print("Happy Painting!")
    print("=" * 60)
