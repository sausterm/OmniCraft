"""
YOLO + Bob Ross Smart Paint by Numbers

Combines:
1. YOLO semantic segmentation for true object boundaries (Dogs, Trees, Grass, etc.)
2. Scene context analysis (time of day, weather, setting, lighting, mood)
3. Subject-specific, context-aware painting strategies
4. Spatial back-to-front layering (not just luminosity-based)

Result: Each semantic region gets intelligent, context-aware painting treatment
that varies based on WHAT it is and WHEN/WHERE the scene takes place.
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

from ..perception.yolo_segmentation import YOLOSemanticSegmenter, SemanticRegion, YOLO_AVAILABLE
from ..perception.scene_context import SceneContextAnalyzer, SceneContext, TimeOfDay, Weather
from ..perception.layering_strategies import (
    LayeringStrategyEngine, LayerSubstep, SubjectType, classify_subject
)


@dataclass
class PaintingSubstep:
    """A single substep within a semantic layer (e.g., Dog - Shadows)."""
    id: str
    name: str
    parent_region: str
    substep_order: int

    mask: np.ndarray
    coverage: float  # Fraction of parent region

    # Visual properties
    luminosity_range: Tuple[float, float]
    dominant_color: Tuple[int, int, int]
    avg_luminosity: float

    # Bob Ross style guidance
    technique: str  # blocking, layering, blending, detailing, highlighting
    brush_suggestion: str
    stroke_direction: str
    tips: List[str] = field(default_factory=list)
    instruction: str = ""


@dataclass
class SemanticPaintingLayer:
    """A semantic layer (from YOLO) with Bob Ross painting substeps."""
    id: str
    name: str
    category: str  # subject, environment, background
    paint_order: int

    # From YOLO
    mask: np.ndarray
    coverage: float
    confidence: float
    depth_estimate: float
    is_focal: bool

    # Visual properties
    dominant_color: Tuple[int, int, int]
    avg_luminosity: float
    color_palette: List[Tuple[int, int, int]] = field(default_factory=list)

    # Bob Ross substeps (painted in order within this layer)
    substeps: List[PaintingSubstep] = field(default_factory=list)

    # Guidance
    technique: str = "layering"
    bob_ross_tips: List[str] = field(default_factory=list)
    instruction: str = ""


class YOLOBobRossPaint:
    """
    YOLO semantic segmentation + Bob Ross painting methodology.

    For each semantic region:
    1. Analyze the region's color/luminosity distribution
    2. Create Bob Ross style substeps (blocking → layering → details → highlights)
    3. Generate painting instructions in Bob Ross voice
    """

    # Bob Ross brush suggestions by technique
    BRUSH_MAP = {
        "blocking": "2-inch background brush",
        "shadow": "1-inch landscape brush",
        "layering": "1-inch landscape brush",
        "blending": "fan brush",
        "detailing": "#6 filbert brush",
        "highlighting": "script liner brush",
    }

    # Bob Ross stroke suggestions
    STROKE_MAP = {
        "blocking": "X-pattern crisscross strokes",
        "shadow": "gentle push-pull strokes",
        "layering": "firm downward pressure",
        "blending": "light feathery strokes",
        "detailing": "short dabbing motions",
        "highlighting": "pull strokes with thin paint",
    }

    # Bob Ross encouragements
    ENCOURAGEMENTS = [
        "There are no mistakes, only happy accidents.",
        "Let's get a little crazy here.",
        "Just let it happen naturally.",
        "This is your world - you can do anything.",
        "We don't make mistakes, we have happy accidents.",
        "Take your time - there's no pressure.",
        "Let's give this little friend some company.",
        "That's a nice touch right there.",
    ]

    def __init__(
        self,
        image_path: str,
        model_size: str = "m",
        conf_threshold: float = 0.3,
        substeps_per_region: int = 4,  # Number of value substeps per semantic region
    ):
        self.image_path = image_path
        self.model_size = model_size
        self.conf_threshold = conf_threshold
        self.substeps_per_region = substeps_per_region

        # Load image
        self.image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        self.h, self.w = self.image.shape[:2]

        # Precompute luminosity
        hsv = cv2.cvtColor(self.image, cv2.COLOR_RGB2HSV)
        self.luminosity = hsv[:, :, 2] / 255.0

        # LAB for perceptual analysis
        lab = cv2.cvtColor(self.image, cv2.COLOR_RGB2LAB)
        self.lab_l = lab[:, :, 0] / 255.0

        # Initialize YOLO segmenter
        self.segmenter = YOLOSemanticSegmenter(model_size=model_size)

        # Scene context (will be analyzed during process)
        self.scene_context: Optional[SceneContext] = None
        self.strategy_engine: Optional[LayeringStrategyEngine] = None

        # Results
        self.semantic_regions: List[SemanticRegion] = []
        self.painting_layers: List[SemanticPaintingLayer] = []

    def process(self) -> List[SemanticPaintingLayer]:
        """
        Full processing pipeline:
        1. Analyze scene context (day/night, weather, setting, mood)
        2. YOLO semantic segmentation
        3. Enforce exclusive boundaries
        4. Create context-aware, spatially-ordered substeps
        5. Generate painting instructions
        """
        print("=" * 60)
        print("YOLO + BOB ROSS SMART PAINT (Context-Aware)")
        print("=" * 60)

        # Step 1: Scene context analysis
        print("\n[1/6] Analyzing scene context...")
        analyzer = SceneContextAnalyzer(self.image)
        self.scene_context = analyzer.analyze()
        self.strategy_engine = LayeringStrategyEngine(self.scene_context)

        print(f"  Time of day: {self.scene_context.time_of_day.value}")
        print(f"  Weather: {self.scene_context.weather.value}")
        print(f"  Setting: {self.scene_context.setting.value}")
        print(f"  Lighting: {self.scene_context.lighting.value}")
        print(f"  Mood: {self.scene_context.mood.value}")
        print(f"  Light direction: {self.scene_context.light_direction}")

        # Step 2: YOLO segmentation
        print("\n[2/6] Running YOLO semantic segmentation...")
        self.semantic_regions = self.segmenter.segment(
            self.image,
            conf_threshold=self.conf_threshold,
            min_coverage=0.01
        )
        print(f"  Found {len(self.semantic_regions)} semantic regions")

        # Step 3: Enforce exclusive boundaries
        print("\n[3/6] Enforcing exclusive semantic boundaries...")
        self._enforce_exclusive_boundaries()

        # Step 4: Create painting layers with context-aware substeps
        print("\n[4/6] Creating context-aware painting layers...")
        self._create_painting_layers()

        # Step 5: Generate Bob Ross style instructions
        print("\n[5/6] Generating Bob Ross instructions...")
        self._generate_bob_ross_instructions()

        # Step 6: Summary
        print("\n[6/6] Finalizing painting plan...")
        total_substeps = sum(len(l.substeps) for l in self.painting_layers)

        print(f"\n{'='*60}")
        print("PAINTING PLAN COMPLETE")
        print(f"{'='*60}")
        print(f"  Scene: {self.scene_context.time_of_day.value} / {self.scene_context.weather.value}")
        print(f"  {len(self.painting_layers)} semantic regions")
        print(f"  {total_substeps} total painting substeps")
        print("\nPAINTING ORDER (back to front):")

        for layer in self.painting_layers:
            focal = " *** FOCAL POINT ***" if layer.is_focal else ""
            print(f"\n  {layer.paint_order}. {layer.name} ({layer.category}){focal}")
            print(f"     Coverage: {layer.coverage*100:.1f}%")
            for sub in layer.substeps:
                print(f"       → {sub.name} ({sub.technique})")

        return self.painting_layers

    def _enforce_exclusive_boundaries(self):
        """Ensure each pixel belongs to exactly one region."""
        h, w = self.h, self.w
        total_pixels = h * w

        ownership = np.full((h, w), -1, dtype=np.int32)

        # Front layers take priority
        regions_by_depth = sorted(
            enumerate(self.semantic_regions),
            key=lambda x: x[1].depth_estimate,
            reverse=True
        )

        for idx, region in regions_by_depth:
            ownership[region.mask] = idx

        # Fill gaps
        uncovered = (ownership == -1)
        if np.sum(uncovered) > 0:
            from scipy import ndimage
            for _ in range(max(h, w)):
                if np.sum(ownership == -1) == 0:
                    break
                still_uncovered = (ownership == -1)
                for idx, region in enumerate(self.semantic_regions):
                    current_owned = (ownership == idx)
                    dilated = ndimage.binary_dilation(current_owned)
                    new_claims = dilated & still_uncovered
                    ownership[new_claims] = idx
                    still_uncovered = still_uncovered & ~new_claims

        # Update masks
        for idx, region in enumerate(self.semantic_regions):
            region.mask = (ownership == idx)
            region.coverage = np.sum(region.mask) / total_pixels

        # Remove empty
        self.semantic_regions = [r for r in self.semantic_regions if np.sum(r.mask) > 0]

        # Verify
        all_masks = sum(r.mask.astype(int) for r in self.semantic_regions)
        print(f"  Coverage: {100*np.sum(all_masks > 0)/total_pixels:.1f}%")
        print(f"  Overlapping: {np.sum(all_masks > 1)} (should be 0)")

    def _create_painting_layers(self):
        """Convert semantic regions to painting layers with context-aware substeps."""
        self.painting_layers = []

        for i, region in enumerate(self.semantic_regions):
            # Extract color palette for this region
            region_pixels = self.image[region.mask]
            from sklearn.cluster import KMeans

            n_colors = min(5, len(region_pixels) // 100 + 1)
            if len(region_pixels) > 100:
                sample_idx = np.random.choice(len(region_pixels), min(1000, len(region_pixels)), replace=False)
                sample = region_pixels[sample_idx].astype(np.float32)
                kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
                kmeans.fit(sample)
                palette = [tuple(c.astype(int)) for c in kmeans.cluster_centers_]
            else:
                palette = [region.dominant_color]

            # Create painting layer
            layer = SemanticPaintingLayer(
                id=region.id,
                name=region.name,
                category=region.category,
                paint_order=region.paint_order,
                mask=region.mask,
                coverage=region.coverage,
                confidence=region.confidence,
                depth_estimate=region.depth_estimate,
                is_focal=region.is_focal,
                dominant_color=region.dominant_color,
                avg_luminosity=region.avg_luminosity,
                color_palette=palette,
            )

            # Classify subject type and get context-aware strategy
            subject_type = classify_subject(region.name)
            strategy_substeps = self.strategy_engine.get_strategy(
                subject_type,
                region.mask,
                self.image,
                is_focal=region.is_focal
            )

            # Convert strategy substeps to painting substeps with actual masks
            layer.substeps = self._create_substeps_from_strategy(layer, strategy_substeps)

            self.painting_layers.append(layer)
            print(f"  {layer.name} ({subject_type.value}): {len(layer.substeps)} substeps")

    def _create_substeps_from_strategy(
        self,
        layer: SemanticPaintingLayer,
        strategy_substeps: List[LayerSubstep]
    ) -> List[PaintingSubstep]:
        """
        Convert strategy substeps to painting substeps with actual masks.

        Handles different mask_method types:
        - "luminosity": Traditional dark-to-light masking
        - "spatial_back", "spatial_front", "spatial_mid": Back-to-front depth masking
        - "spatial_top", "spatial_bottom": Vertical position masking (for grass, sky)
        - "spatial_left", "spatial_right": Horizontal position masking
        - "full": Entire region
        """
        substeps = []

        for i, strategy_step in enumerate(strategy_substeps):
            # Generate mask based on mask_method
            sub_mask = self._generate_mask_for_strategy(
                layer.mask,
                strategy_step.mask_method,
                strategy_step.mask_params
            )

            if np.sum(sub_mask) == 0:
                continue

            # Get properties for this substep mask
            sub_pixels = self.image[sub_mask]
            sub_color = tuple(np.median(sub_pixels, axis=0).astype(int))
            sub_lum_values = self.luminosity[sub_mask]
            sub_lum = float(np.mean(sub_lum_values))
            lum_range = (float(np.min(sub_lum_values)), float(np.max(sub_lum_values)))

            substep = PaintingSubstep(
                id=f"{layer.id}_sub{i+1}",
                name=f"{layer.name} - {strategy_step.name}",
                parent_region=layer.name,
                substep_order=i + 1,
                mask=sub_mask,
                coverage=np.sum(sub_mask) / max(1, np.sum(layer.mask)),
                luminosity_range=lum_range,
                dominant_color=sub_color,
                avg_luminosity=sub_lum,
                technique=strategy_step.technique,
                brush_suggestion=strategy_step.brush,
                stroke_direction=strategy_step.stroke,
                tips=strategy_step.tips.copy(),
                instruction=strategy_step.description,
            )
            substeps.append(substep)

        return substeps

    def _generate_mask_for_strategy(
        self,
        region_mask: np.ndarray,
        mask_method: str,
        mask_params: Dict
    ) -> np.ndarray:
        """
        Generate a mask based on the strategy's mask_method.

        Spatial methods use position within the region's bounding box.
        Luminosity methods use brightness values.
        """
        if mask_method == "full":
            return region_mask.copy()

        elif mask_method == "luminosity":
            # Traditional luminosity-based masking
            lum_range = mask_params.get("range", (0.0, 1.0))
            region_lum = self.luminosity.copy()
            region_lum[~region_mask] = -1  # Mark non-region pixels

            # Calculate percentiles within the region
            region_lum_values = self.luminosity[region_mask]
            if len(region_lum_values) == 0:
                return np.zeros_like(region_mask)

            lum_min = np.percentile(region_lum_values, lum_range[0] * 100)
            lum_max = np.percentile(region_lum_values, lum_range[1] * 100)

            return region_mask & (self.luminosity >= lum_min) & (self.luminosity <= lum_max)

        elif mask_method in ["spatial_back", "spatial_mid", "spatial_front"]:
            # Back-to-front using luminosity as depth proxy
            # Back = darkest (furthest), Front = brightest (closest)
            return self._create_depth_mask(region_mask, mask_method, mask_params)

        elif mask_method in ["spatial_top", "spatial_bottom", "spatial_middle"]:
            # Vertical position masking
            return self._create_vertical_mask(region_mask, mask_method, mask_params)

        elif mask_method in ["spatial_left", "spatial_right"]:
            # Horizontal position masking
            return self._create_horizontal_mask(region_mask, mask_method, mask_params)

        else:
            # Default to full region
            return region_mask.copy()

    def _create_depth_mask(
        self,
        region_mask: np.ndarray,
        method: str,
        params: Dict
    ) -> np.ndarray:
        """
        Create back-to-front depth mask using luminosity as depth proxy.
        Darker = further back, Brighter = closer to front.
        """
        region_lum = self.luminosity[region_mask]
        if len(region_lum) == 0:
            return np.zeros_like(region_mask)

        lum_min, lum_max = np.min(region_lum), np.max(region_lum)
        lum_range = lum_max - lum_min

        if lum_range < 0.01:
            # Flat luminosity - return full mask for back, empty for others
            return region_mask.copy() if method == "spatial_back" else np.zeros_like(region_mask)

        if method == "spatial_back":
            # Darkest third = back
            depth = params.get("depth", 0.33)
            threshold = lum_min + lum_range * depth
            return region_mask & (self.luminosity <= threshold)

        elif method == "spatial_mid":
            # Middle third
            depth_range = params.get("depth", (0.33, 0.66))
            low_thresh = lum_min + lum_range * depth_range[0]
            high_thresh = lum_min + lum_range * depth_range[1]
            return region_mask & (self.luminosity > low_thresh) & (self.luminosity <= high_thresh)

        elif method == "spatial_front":
            # Brightest third = front
            depth = params.get("depth", 0.66)
            threshold = lum_min + lum_range * depth
            return region_mask & (self.luminosity > threshold)

        return region_mask.copy()

    def _create_vertical_mask(
        self,
        region_mask: np.ndarray,
        method: str,
        params: Dict
    ) -> np.ndarray:
        """
        Create vertical position mask (top/middle/bottom of region).
        Useful for grass, sky, water.
        """
        # Find bounding box of region
        rows = np.any(region_mask, axis=1)
        if not np.any(rows):
            return np.zeros_like(region_mask)

        row_indices = np.where(rows)[0]
        top_row, bottom_row = row_indices[0], row_indices[-1]
        height = bottom_row - top_row + 1

        result = np.zeros_like(region_mask)
        portion = params.get("portion", 0.33)

        if method == "spatial_top":
            # Top portion (distant for grass, upper sky)
            cutoff = top_row + int(height * portion)
            result[:cutoff, :] = region_mask[:cutoff, :]

        elif method == "spatial_bottom":
            # Bottom portion (foreground for grass)
            cutoff = bottom_row - int(height * portion)
            result[cutoff:, :] = region_mask[cutoff:, :]

        elif method == "spatial_middle":
            # Middle portion
            if isinstance(portion, tuple):
                top_p, bottom_p = portion
            else:
                top_p, bottom_p = 0.33, 0.66
            top_cutoff = top_row + int(height * top_p)
            bottom_cutoff = top_row + int(height * bottom_p)
            result[top_cutoff:bottom_cutoff, :] = region_mask[top_cutoff:bottom_cutoff, :]

        return result

    def _create_horizontal_mask(
        self,
        region_mask: np.ndarray,
        method: str,
        params: Dict
    ) -> np.ndarray:
        """
        Create horizontal position mask (left/right of region).
        Useful for mountains, buildings with side lighting.
        """
        # Find bounding box of region
        cols = np.any(region_mask, axis=0)
        if not np.any(cols):
            return np.zeros_like(region_mask)

        col_indices = np.where(cols)[0]
        left_col, right_col = col_indices[0], col_indices[-1]
        width = right_col - left_col + 1

        result = np.zeros_like(region_mask)
        portion = params.get("portion", 0.5)

        if method == "spatial_left":
            cutoff = left_col + int(width * portion)
            result[:, :cutoff] = region_mask[:, :cutoff]

        elif method == "spatial_right":
            cutoff = right_col - int(width * portion)
            result[:, cutoff:] = region_mask[:, cutoff:]

        return result

    def _generate_bob_ross_instructions(self):
        """Generate Bob Ross style instructions for each layer and substep."""
        import random

        for layer in self.painting_layers:
            # Layer-level instruction
            if layer.category == "background":
                layer.instruction = f"Let's start with our {layer.name.lower()} in the background. Remember, we always work from the back forward."
                layer.technique = "blocking"
            elif layer.category == "environment":
                layer.instruction = f"Now let's bring in our {layer.name.lower()}. This is going to be so much fun."
                layer.technique = "layering"
            else:  # subject
                layer.instruction = f"Here comes our little friend - the {layer.name.lower()}. This is the star of our painting."
                layer.technique = "detailing"

            # Layer tips
            if layer.is_focal:
                layer.bob_ross_tips = [
                    f"This {layer.name.lower()} is our focal point - take your time here",
                    "Build up the form with careful value transitions",
                    "Pay attention to the edges - some soft, some defined",
                    random.choice(self.ENCOURAGEMENTS),
                ]
            else:
                layer.bob_ross_tips = [
                    "Don't overwork this area - it's supporting, not starring",
                    "Keep your strokes confident and decisive",
                    random.choice(self.ENCOURAGEMENTS),
                ]

            # Substep instructions
            for sub in layer.substeps:
                sub.instruction = self._create_substep_instruction(layer, sub)
                sub.tips = self._create_substep_tips(sub)

    def _create_substep_instruction(self, layer: SemanticPaintingLayer, sub: PaintingSubstep) -> str:
        """Create Bob Ross style instruction for a substep."""
        parts = []

        # Opening based on technique
        if sub.technique == "blocking":
            parts.append(f"Let's start blocking in the {sub.name.lower()}.")
            parts.append(f"Load your {sub.brush_suggestion} with a thin, dark mix.")
            parts.append("We're just establishing our values here - don't worry about details yet.")
        elif sub.technique == "shadow":
            parts.append(f"Now we'll add the shadows to our {layer.name.lower()}.")
            parts.append(f"Using your {sub.brush_suggestion}, work in the darker areas.")
            parts.append("Remember - shadows give our subject depth and form.")
        elif sub.technique == "layering":
            parts.append(f"Time to build up our {sub.name.lower()}.")
            parts.append(f"With your {sub.brush_suggestion}, use {sub.stroke_direction}.")
            parts.append("Work the paint into the canvas, blending with what's already there.")
        elif sub.technique == "blending":
            parts.append(f"Let's soften and blend our {layer.name.lower()}.")
            parts.append(f"Take your {sub.brush_suggestion} and use gentle {sub.stroke_direction}.")
            parts.append("We're creating smooth transitions between our values.")
        elif sub.technique == "highlighting":
            parts.append(f"Now for the magic - let's add highlights to our {layer.name.lower()}!")
            parts.append(f"Load just a little paint on your {sub.brush_suggestion}.")
            parts.append("Less is more here - just touch where the light would naturally hit.")

        return " ".join(parts)

    def _create_substep_tips(self, sub: PaintingSubstep) -> List[str]:
        """Create tips for a substep."""
        import random
        tips = []

        if sub.technique == "blocking":
            tips.extend([
                "Keep your paint thin at this stage",
                "Don't be afraid of the dark - it creates depth",
            ])
        elif sub.technique == "shadow":
            tips.extend([
                "Shadows aren't black - they have color",
                "Look for the subtle color shifts in the dark areas",
            ])
        elif sub.technique == "layering":
            tips.extend([
                "Let each layer dry slightly before the next",
                "Build up gradually - you can always add more",
            ])
        elif sub.technique == "highlighting":
            tips.extend([
                "Highlights are the icing on the cake",
                "Use your brightest values sparingly",
            ])

        tips.append(random.choice(self.ENCOURAGEMENTS))
        return tips

    def create_step_images(self, output_dir: str) -> Dict[str, List[str]]:
        """
        Create step-by-step progress images with three views per step:
        1. Cumulative: Progressive build-up (existing behavior)
        2. Context: Full image with current step highlighted, rest dimmed
        3. Isolated: Just the current step's region on white canvas

        Returns dict with paths for each view type.
        """
        # Create subdirectories for each view type
        cumulative_dir = os.path.join(output_dir, "cumulative")
        context_dir = os.path.join(output_dir, "context")
        isolated_dir = os.path.join(output_dir, "isolated")

        os.makedirs(cumulative_dir, exist_ok=True)
        os.makedirs(context_dir, exist_ok=True)
        os.makedirs(isolated_dir, exist_ok=True)

        saved_paths = {
            "cumulative": [],
            "context": [],
            "isolated": []
        }

        # Create dimmed version of full image for context view
        dimmed_image = self._create_dimmed_image(self.image, dim_factor=0.35, desaturate=0.7)

        cumulative = np.ones((self.h, self.w, 3), dtype=np.uint8) * 255
        step_num = 0

        for layer in self.painting_layers:
            for sub in layer.substeps:
                step_num += 1
                safe_name = sub.name.replace(' ', '_').replace('-', '_')

                # 1. CUMULATIVE VIEW - Progressive build-up
                cumulative[sub.mask] = self.image[sub.mask]
                cumulative_path = os.path.join(cumulative_dir, f"step_{step_num:02d}_{safe_name}.png")
                cv2.imwrite(cumulative_path, cv2.cvtColor(cumulative, cv2.COLOR_RGB2BGR))
                saved_paths["cumulative"].append(cumulative_path)

                # 2. CONTEXT VIEW - Full image with current step highlighted
                context_image = self._create_context_view(dimmed_image, sub.mask)
                context_path = os.path.join(context_dir, f"step_{step_num:02d}_{safe_name}_context.png")
                cv2.imwrite(context_path, cv2.cvtColor(context_image, cv2.COLOR_RGB2BGR))
                saved_paths["context"].append(context_path)

                # 3. ISOLATED VIEW - Just this step's region on white canvas
                isolated_image = self._create_isolated_view(sub.mask)
                isolated_path = os.path.join(isolated_dir, f"step_{step_num:02d}_{safe_name}_isolated.png")
                cv2.imwrite(isolated_path, cv2.cvtColor(isolated_image, cv2.COLOR_RGB2BGR))
                saved_paths["isolated"].append(isolated_path)

        # Final cumulative
        final_path = os.path.join(cumulative_dir, "final_complete.png")
        cv2.imwrite(final_path, cv2.cvtColor(cumulative, cv2.COLOR_RGB2BGR))
        saved_paths["cumulative"].append(final_path)

        return saved_paths

    def _create_dimmed_image(
        self,
        image: np.ndarray,
        dim_factor: float = 0.4,
        desaturate: float = 0.6
    ) -> np.ndarray:
        """
        Create a dimmed and desaturated version of the image.

        Args:
            image: RGB image
            dim_factor: How much to dim (0=black, 1=original brightness)
            desaturate: How much to desaturate (0=grayscale, 1=original saturation)

        Returns:
            Dimmed RGB image
        """
        # Convert to HSV for saturation control
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)

        # Reduce saturation
        hsv[:, :, 1] = hsv[:, :, 1] * desaturate

        # Reduce value (brightness)
        hsv[:, :, 2] = hsv[:, :, 2] * dim_factor

        # Clip and convert back
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        dimmed = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        return dimmed

    def _create_context_view(
        self,
        dimmed_image: np.ndarray,
        highlight_mask: np.ndarray
    ) -> np.ndarray:
        """
        Create context view: full image with highlighted region at full color,
        rest of image dimmed/desaturated.

        Args:
            dimmed_image: Pre-computed dimmed version of the image
            highlight_mask: Binary mask of region to highlight

        Returns:
            Context view RGB image
        """
        # Start with dimmed image
        context = dimmed_image.copy()

        # Overlay the highlighted region with original colors
        context[highlight_mask] = self.image[highlight_mask]

        # Optional: Add subtle border around highlighted region for clarity
        # Find contours and draw a subtle outline
        mask_uint8 = highlight_mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw a subtle white glow/outline around the region
        if contours:
            # Create a slightly dilated mask for the glow effect
            kernel = np.ones((3, 3), np.uint8)
            dilated = cv2.dilate(mask_uint8, kernel, iterations=2)
            glow_ring = (dilated > 0) & ~highlight_mask

            # Blend a subtle white glow
            context[glow_ring] = (context[glow_ring] * 0.7 + np.array([255, 255, 255]) * 0.3).astype(np.uint8)

        return context

    def _create_isolated_view(self, mask: np.ndarray) -> np.ndarray:
        """
        Create isolated view: just the masked region on a white canvas.

        Args:
            mask: Binary mask of region to isolate

        Returns:
            Isolated view RGB image
        """
        # White canvas
        isolated = np.ones((self.h, self.w, 3), dtype=np.uint8) * 255

        # Copy only the masked region
        isolated[mask] = self.image[mask]

        return isolated

    def create_progress_overview(self, output_dir: str) -> str:
        """Create grid overview showing all painting steps."""
        total_steps = sum(len(l.substeps) for l in self.painting_layers)

        cols = min(5, total_steps + 1)
        rows = (total_steps + 1 + cols - 1) // cols

        thumb_h = 180 if total_steps > 15 else 220
        thumb_w = int(thumb_h * self.w / self.h)
        margin = 6
        label_h = 45

        canvas_w = cols * (thumb_w + margin) + margin
        canvas_h = rows * (thumb_h + label_h + margin) + margin + 70
        canvas = Image.new('RGB', (canvas_w, canvas_h), 'white')
        draw = ImageDraw.Draw(canvas)

        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 9)
            font_title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 13)
        except:
            font = font_title = ImageFont.load_default()

        # Title
        n_regions = len(self.painting_layers)
        title = f"YOLO + BOB ROSS PAINT - {n_regions} Regions, {total_steps} Steps"
        draw.text((margin, margin), title, fill='darkblue', font=font_title)

        subtitle = '"There are no mistakes, only happy accidents"'
        draw.text((margin, margin + 20), subtitle, fill='gray', font=font)

        # Build progress images
        cumulative = np.ones((self.h, self.w, 3), dtype=np.uint8) * 255
        step_num = 0

        for layer in self.painting_layers:
            for sub in layer.substeps:
                cumulative[sub.mask] = self.image[sub.mask]

                thumb = Image.fromarray(cumulative).resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)

                col = step_num % cols
                row = step_num // cols
                x = margin + col * (thumb_w + margin)
                y = margin + 70 + row * (thumb_h + label_h + margin)

                canvas.paste(thumb, (x, y))

                # Label
                short_name = sub.name[:30] if len(sub.name) > 30 else sub.name
                draw.text((x, y + thumb_h + 2), f"{step_num+1}. {short_name}", fill='black', font=font)
                draw.text((x, y + thumb_h + 14), f"({sub.technique})", fill='gray', font=font)

                step_num += 1

        overview_path = os.path.join(output_dir, "progress_overview.png")
        canvas.save(overview_path)
        return overview_path

    def create_layer_guide(self) -> List[Dict]:
        """Generate comprehensive JSON painting guide."""
        guide = []

        # Prep step
        guide.append({
            "step": 0,
            "name": "Canvas Preparation",
            "type": "setup",
            "bob_ross_intro": "Welcome! Today we're going to paint a beautiful scene together. "
                             "Let's start by preparing our canvas with a thin layer of liquid white.",
            "materials": [
                "Canvas prepared with liquid white",
                "Paper towels for cleaning brushes",
                "Odorless thinner",
                "Your palette arranged with colors",
            ],
            "tips": [
                "Make sure your liquid white is thin and even",
                "Have all your brushes clean and ready",
                "Remember - we don't make mistakes, just happy accidents!",
            ]
        })

        # Layer and substep guides
        step = 0
        for layer in self.painting_layers:
            # Layer intro
            guide.append({
                "step": step + 0.5,
                "name": f"Begin {layer.name}",
                "type": "layer_intro",
                "region": layer.name,
                "category": layer.category,
                "bob_ross_intro": layer.instruction,
                "tips": layer.bob_ross_tips,
                "is_focal": layer.is_focal,
            })

            for sub in layer.substeps:
                step += 1
                guide.append({
                    "step": step,
                    "name": sub.name,
                    "type": "paint_substep",
                    "parent_region": layer.name,
                    "category": layer.category,
                    "technique": sub.technique,
                    "brush": sub.brush_suggestion,
                    "strokes": sub.stroke_direction,
                    "coverage": f"{sub.coverage*100:.1f}%",
                    "luminosity_range": list(sub.luminosity_range),
                    "dominant_color": [int(c) for c in sub.dominant_color],
                    "instruction": sub.instruction,
                    "tips": sub.tips,
                    "is_focal": layer.is_focal,
                })

        # Final step
        guide.append({
            "step": step + 1,
            "name": "Final Touches & Sign",
            "type": "finishing",
            "bob_ross_outro": "There we go! Our painting is complete. Step back and admire your work. "
                             "Remember, this is YOUR world - you created something beautiful today.",
            "tips": [
                "Step back and view from a distance",
                "Make any final value adjustments",
                "Add your signature - you've earned it!",
                "Most importantly - be proud of what you've created!",
            ]
        })

        return guide

    def save_all(self, output_dir: str):
        """Save all outputs."""
        os.makedirs(output_dir, exist_ok=True)

        print(f"\nSaving to {output_dir}/")
        print("-" * 50)

        # Reference
        ref_path = os.path.join(output_dir, "colored_reference.png")
        cv2.imwrite(ref_path, cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR))
        print("  colored_reference.png")

        # Steps with three view types
        steps_dir = os.path.join(output_dir, "steps")
        step_paths = self.create_step_images(steps_dir)
        n_cumulative = len(step_paths["cumulative"])
        n_context = len(step_paths["context"])
        n_isolated = len(step_paths["isolated"])
        print(f"  steps/")
        print(f"    cumulative/ - {n_cumulative} images (progressive build-up)")
        print(f"    context/    - {n_context} images (highlighted in full image)")
        print(f"    isolated/   - {n_isolated} images (region on white canvas)")

        # Overview
        self.create_progress_overview(output_dir)
        print("  progress_overview.png")

        # Guide
        guide = self.create_layer_guide()
        guide_path = os.path.join(output_dir, "painting_guide.json")
        with open(guide_path, 'w') as f:
            json.dump(guide, f, indent=2)
        print("  painting_guide.json")

        # Scene analysis with context
        analysis = {
            "method": "yolo_bob_ross_context_aware",
            "scene_context": {
                "time_of_day": self.scene_context.time_of_day.value,
                "weather": self.scene_context.weather.value,
                "setting": self.scene_context.setting.value,
                "lighting": self.scene_context.lighting.value,
                "mood": self.scene_context.mood.value,
                "light_direction": self.scene_context.light_direction,
                "color_temperature": round(self.scene_context.color_temperature, 3),
                "avg_luminosity": round(self.scene_context.avg_luminosity, 3),
                "contrast": round(self.scene_context.contrast, 3),
                "saturation": round(self.scene_context.saturation, 3),
            },
            "num_regions": len(self.painting_layers),
            "total_substeps": sum(len(l.substeps) for l in self.painting_layers),
            "regions": [
                {
                    "name": l.name,
                    "subject_type": classify_subject(l.name).value,
                    "category": l.category,
                    "coverage": round(l.coverage, 4),
                    "confidence": round(l.confidence, 2),
                    "is_focal": l.is_focal,
                    "substeps": len(l.substeps),
                    "substep_details": [
                        {
                            "name": s.name,
                            "technique": s.technique,
                            "coverage": round(s.coverage, 3),
                        }
                        for s in l.substeps
                    ],
                }
                for l in self.painting_layers
            ]
        }
        analysis_path = os.path.join(output_dir, "scene_analysis.json")
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        print("  scene_analysis.json")

        print("-" * 50)
        print("Done! Happy painting!")


def process_image(
    image_path: str,
    output_dir: str = None,
    model_size: str = "m",
    conf_threshold: float = 0.2,
    substeps: int = 4
) -> YOLOBobRossPaint:
    """Main entry point."""
    if output_dir is None:
        base = os.path.splitext(os.path.basename(image_path))[0]
        output_dir = f"output/{base}_yolo_bob_ross"

    painter = YOLOBobRossPaint(
        image_path,
        model_size=model_size,
        conf_threshold=conf_threshold,
        substeps_per_region=substeps
    )
    painter.process()
    painter.save_all(output_dir)

    return painter


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python yolo_bob_ross_paint.py <image_path> [output_dir]")
        sys.exit(0)

    image_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    process_image(image_path, output_dir)
