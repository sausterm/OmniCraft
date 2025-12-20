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
import logging
from PIL import Image, ImageDraw, ImageFont
import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)

# Vision module imports (segmentation + analysis)
from artisan.vision.segmentation.yolo_segmentation import YOLOSemanticSegmenter, SemanticRegion, YOLO_AVAILABLE
from artisan.vision.analysis.scene_context import SceneContextAnalyzer, SceneContext, TimeOfDay, Weather
from artisan.vision.analysis.layering_strategies import (
    LayeringStrategyEngine, LayerSubstep, SubjectType, classify_subject
)
from artisan.vision.analysis.scene_analyzer import (
    SceneAnalyzer, SceneAnalysisResult, LightingRole, ValueProgression, DepthLayer, SceneType
)
from artisan.paint.planning.painting_planner import PaintingPlanner, PaintingSubstepPlan


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

    # Scene analysis info (from SceneAnalyzer)
    lighting_role: Optional[str] = None  # emitter, silhouette, reflector, etc.
    value_progression: Optional[str] = None  # dark_to_light, glow_then_edges, etc.
    depth_layer: Optional[str] = None  # background, midground, foreground, focal


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

    # Available painting styles
    PAINT_STYLES = ["photo", "oil", "impressionist", "poster", "watercolor"]

    def __init__(
        self,
        image_path: str,
        model_size: str = "m",
        conf_threshold: float = 0.3,
        substeps_per_region: int = 4,  # Number of value substeps per semantic region
        paint_style: str = "photo",  # photo, oil, impressionist, poster, watercolor
        simplify: int = 0,  # 0=none, 1=light, 2=medium, 3=heavy detail reduction
    ):
        self.image_path = image_path
        self.model_size = model_size
        self.conf_threshold = conf_threshold
        self.substeps_per_region = substeps_per_region
        self.paint_style = paint_style if paint_style in self.PAINT_STYLES else "photo"
        self.simplify = max(0, min(3, simplify))  # Clamp to 0-3

        # Load original image
        self.original_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        self.h, self.w = self.original_image.shape[:2]

        # Apply simplification first (reduces detail for cleaner YOLO detection)
        if self.simplify > 0:
            print(f"  Simplifying image (level {self.simplify})...")
            self.image = self._simplify_image(self.original_image, self.simplify)
        else:
            self.image = self.original_image.copy()

        # Apply painterly style transformation
        if self.paint_style != "photo":
            print(f"  Applying '{self.paint_style}' style...")
            self.image = self._apply_paint_style(self.image, self.paint_style)

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

        # Advanced scene analysis (from SceneAnalyzer)
        self.scene_analysis: Optional[SceneAnalysisResult] = None
        self.use_advanced_analysis: bool = True  # Use SceneAnalyzer + PaintingPlanner

        # Results
        self.semantic_regions: List[SemanticRegion] = []
        self.painting_layers: List[SemanticPaintingLayer] = []

    def _simplify_image(self, image: np.ndarray, level: int) -> np.ndarray:
        """
        Reduce image detail to help YOLO detect fewer, larger regions.

        This is applied BEFORE paint styles and helps reduce hyperrealism
        that causes too many small objects to be detected.

        Levels:
        - 1: Light - subtle smoothing, preserves most detail
        - 2: Medium - noticeable smoothing, merges small details
        - 3: Heavy - significant simplification, painterly base
        """
        result = image.copy()

        if level >= 1:
            # Light: Edge-preserving filter - smooths while keeping edges
            result = cv2.edgePreservingFilter(result, flags=1, sigma_s=40, sigma_r=0.4)

        if level >= 2:
            # Medium: Add bilateral filter for more smoothing
            result = cv2.bilateralFilter(result, d=9, sigmaColor=50, sigmaSpace=50)
            # Light color quantization to merge similar colors
            result = self._quantize_colors(result, n_colors=32)

        if level >= 3:
            # Heavy: More aggressive smoothing and quantization
            result = cv2.bilateralFilter(result, d=11, sigmaColor=75, sigmaSpace=75)
            result = cv2.edgePreservingFilter(result, flags=1, sigma_s=60, sigma_r=0.5)
            result = self._quantize_colors(result, n_colors=20)

        return result

    def _apply_paint_style(self, image: np.ndarray, style: str) -> np.ndarray:
        """
        Apply painterly style transformation to the image.

        Styles:
        - oil: Rich colors, visible brush strokes, classic oil painting look
        - impressionist: Soft edges, posterized colors, dreamy quality
        - poster: Flat color areas with strong edges, paint-by-numbers look
        - watercolor: Soft, transparent, with bleeding edges
        """
        if style == "oil":
            return self._style_oil_paint(image)
        elif style == "impressionist":
            return self._style_impressionist(image)
        elif style == "poster":
            return self._style_poster(image)
        elif style == "watercolor":
            return self._style_watercolor(image)
        else:
            return image.copy()

    def _style_oil_paint(self, image: np.ndarray) -> np.ndarray:
        """
        Oil painting effect: rich colors, visible brush strokes.
        Uses bilateral filter for smoothing + edge enhancement.
        """
        # Strong bilateral filter for smooth color regions
        smooth = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
        smooth = cv2.bilateralFilter(smooth, d=9, sigmaColor=75, sigmaSpace=75)

        # Enhance saturation for richer colors
        hsv = cv2.cvtColor(smooth, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.2, 0, 255)  # Boost saturation
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.05, 0, 255)  # Slight brightness boost
        smooth = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        # Quantize colors slightly for painterly effect
        smooth = self._quantize_colors(smooth, n_colors=24)

        # Add subtle edge darkening for depth
        edges = cv2.Canny(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), 50, 150)
        edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
        edges = cv2.GaussianBlur(edges, (3, 3), 0)

        # Darken edges slightly
        result = smooth.astype(np.float32)
        edge_mask = edges.astype(np.float32) / 255.0
        result = result * (1 - edge_mask[:, :, np.newaxis] * 0.3)

        return np.clip(result, 0, 255).astype(np.uint8)

    def _style_impressionist(self, image: np.ndarray) -> np.ndarray:
        """
        Impressionist effect: soft, dreamy, with visible color patches.
        """
        # Use OpenCV stylization for base effect
        stylized = cv2.stylization(image, sigma_s=60, sigma_r=0.45)

        # Soften further with edge-preserving filter
        soft = cv2.edgePreservingFilter(stylized, flags=1, sigma_s=60, sigma_r=0.4)

        # Subtle color quantization
        result = self._quantize_colors(soft, n_colors=20)

        # Slight desaturation for dreamy quality
        hsv = cv2.cvtColor(result, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = hsv[:, :, 1] * 0.9
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        return result

    def _style_poster(self, image: np.ndarray) -> np.ndarray:
        """
        Poster/paint-by-numbers effect: flat color areas with defined edges.
        """
        # Strong color quantization
        quantized = self._quantize_colors(image, n_colors=12)

        # Bilateral filter for flat regions
        flat = cv2.bilateralFilter(quantized, d=15, sigmaColor=100, sigmaSpace=100)

        # Edge detection
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 80, 200)
        edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)

        # Combine flat colors with dark edges
        result = flat.copy()
        result[edges > 0] = result[edges > 0] * 0.3  # Darken edges

        return result.astype(np.uint8)

    def _style_watercolor(self, image: np.ndarray) -> np.ndarray:
        """
        Watercolor effect: soft edges, transparent look, bleeding colors.
        """
        # Multiple passes of edge-preserving filter for soft effect
        soft = cv2.edgePreservingFilter(image, flags=2, sigma_s=80, sigma_r=0.5)
        soft = cv2.edgePreservingFilter(soft, flags=2, sigma_s=60, sigma_r=0.4)

        # Slight color quantization
        soft = self._quantize_colors(soft, n_colors=16)

        # Reduce contrast slightly for transparent look
        soft = soft.astype(np.float32)
        mean = np.mean(soft)
        soft = (soft - mean) * 0.85 + mean  # Reduce contrast

        # Lighten overall for watercolor transparency
        soft = soft * 0.95 + 255 * 0.05

        # Add subtle paper texture effect (slight noise)
        noise = np.random.normal(0, 3, soft.shape).astype(np.float32)
        soft = soft + noise

        return np.clip(soft, 0, 255).astype(np.uint8)

    def _quantize_colors(self, image: np.ndarray, n_colors: int = 16) -> np.ndarray:
        """Reduce image to n_colors using k-means clustering."""
        from sklearn.cluster import MiniBatchKMeans

        # Reshape for clustering
        pixels = image.reshape(-1, 3).astype(np.float32)

        # K-means clustering
        kmeans = MiniBatchKMeans(n_clusters=n_colors, random_state=42, n_init=3)
        labels = kmeans.fit_predict(pixels)
        centers = kmeans.cluster_centers_

        # Reconstruct image
        quantized = centers[labels].reshape(image.shape)
        return quantized.astype(np.uint8)

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

        # Build regions dict for SceneAnalyzer
        regions_dict = {region.name: region.mask for region in self.semantic_regions}
        subject_types = {region.name: classify_subject(region.name) for region in self.semantic_regions}

        # Run advanced scene analysis if enabled
        if self.use_advanced_analysis and regions_dict:
            print("  Running advanced scene analysis...")
            analyzer = SceneAnalyzer(self.image, self.scene_context)
            self.scene_analysis = analyzer.analyze(regions_dict)

            print(f"    Scene type: {self.scene_analysis.scene_type.value}")
            print(f"    Backlit: {self.scene_analysis.is_backlit}")
            print(f"    Light sources: {len(self.scene_analysis.light_sources)}")

            # Print lighting roles for each region
            for name, analysis in self.scene_analysis.region_analyses.items():
                print(f"    {name}: {analysis.lighting_role.value} -> {analysis.value_progression.value}")

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

            # Get scene analysis info for this region (if available)
            region_analysis = None
            if self.scene_analysis and region.name in self.scene_analysis.region_analyses:
                region_analysis = self.scene_analysis.region_analyses[region.name]

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
                # Add scene analysis info
                lighting_role=region_analysis.lighting_role.value if region_analysis else None,
                value_progression=region_analysis.value_progression.value if region_analysis else None,
                depth_layer=region_analysis.depth_layer.value if region_analysis else None,
            )

            # Generate substeps using advanced planner or fallback to strategy engine
            subject_type = classify_subject(region.name)

            if self.use_advanced_analysis and region_analysis:
                # Use the advanced PaintingPlanner for context-aware substeps
                layer.substeps = self._create_substeps_from_scene_analysis(
                    layer, region_analysis, subject_type
                )
            else:
                # Fallback to original LayeringStrategyEngine
                strategy_substeps = self.strategy_engine.get_strategy(
                    subject_type,
                    region.mask,
                    self.image,
                    is_focal=region.is_focal
                )
                layer.substeps = self._create_substeps_from_strategy(layer, strategy_substeps)

            self.painting_layers.append(layer)
            role_info = f" [{layer.lighting_role}]" if layer.lighting_role else ""
            print(f"  {layer.name} ({subject_type.value}){role_info}: {len(layer.substeps)} substeps")

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

        Ensures ALL pixels in the region are captured (no gaps).
        """
        substeps = []

        # Track which pixels have been assigned to ensure full coverage
        assigned_pixels = np.zeros_like(layer.mask, dtype=bool)

        for i, strategy_step in enumerate(strategy_substeps):
            # Generate mask based on mask_method
            sub_mask = self._generate_mask_for_strategy(
                layer.mask,
                strategy_step.mask_method,
                strategy_step.mask_params
            )

            if np.sum(sub_mask) == 0:
                continue

            # Track assigned pixels
            assigned_pixels |= sub_mask

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

        # CRITICAL: Capture any unassigned pixels and add them to the appropriate substep
        unassigned = layer.mask & ~assigned_pixels
        if np.sum(unassigned) > 0:
            self._distribute_unassigned_pixels(substeps, unassigned, layer)

        return substeps

    def _create_substeps_from_scene_analysis(
        self,
        layer: SemanticPaintingLayer,
        region_analysis,  # RegionAnalysis from scene_analyzer
        subject_type: SubjectType
    ) -> List[PaintingSubstep]:
        """
        Create substeps using advanced scene analysis.

        Uses the region's lighting role and value progression to determine:
        - Correct order of values (dark-to-light vs light-to-dark vs glow-then-edges)
        - Appropriate techniques and tips based on lighting context

        Ensures ALL pixels in the region are captured (no gaps).
        """
        from artisan.vision.analysis.scene_analyzer import ValueProgression, LightingRole

        substeps = []
        value_progression = region_analysis.value_progression
        lighting_role = region_analysis.lighting_role
        is_focal = layer.is_focal

        # Get value ranges based on progression type
        num_substeps = 5 if is_focal else 4
        value_ranges = self._get_value_ranges_for_progression(value_progression, num_substeps)

        # Get substep configurations based on progression and lighting role
        configs = self._get_substep_configs_for_progression(
            value_progression, lighting_role, subject_type, num_substeps
        )

        # Track which pixels have been assigned to ensure full coverage
        assigned_pixels = np.zeros_like(layer.mask, dtype=bool)

        for i, (value_range, config) in enumerate(zip(value_ranges, configs)):
            # Generate mask for this value range
            sub_mask = self._generate_mask_for_strategy(
                layer.mask,
                "luminosity",
                {"range": value_range}
            )

            if np.sum(sub_mask) == 0:
                continue

            # Track assigned pixels
            assigned_pixels |= sub_mask

            # Get properties for this substep mask
            sub_pixels = self.image[sub_mask]
            sub_color = tuple(np.median(sub_pixels, axis=0).astype(int))
            sub_lum_values = self.luminosity[sub_mask]
            sub_lum = float(np.mean(sub_lum_values))
            lum_range = (float(np.min(sub_lum_values)), float(np.max(sub_lum_values)))

            # Generate instruction based on lighting context
            instruction = self._generate_context_instruction(
                layer.name, config, value_progression, lighting_role, i, num_substeps
            )

            # Generate tips based on lighting role
            tips = self._generate_context_tips(config, lighting_role, region_analysis.depth_layer, i, num_substeps)

            substep = PaintingSubstep(
                id=f"{layer.id}_sub{i+1}",
                name=f"{layer.name} - {config['name']}",
                parent_region=layer.name,
                substep_order=i + 1,
                mask=sub_mask,
                coverage=np.sum(sub_mask) / max(1, np.sum(layer.mask)),
                luminosity_range=lum_range,
                dominant_color=sub_color,
                avg_luminosity=sub_lum,
                technique=config['technique'],
                brush_suggestion=config['brush'],
                stroke_direction=config['strokes'],
                tips=tips,
                instruction=instruction,
            )
            substeps.append(substep)

        # CRITICAL: Capture any unassigned pixels and add them to the appropriate substep
        unassigned = layer.mask & ~assigned_pixels
        if np.sum(unassigned) > 0:
            self._distribute_unassigned_pixels(substeps, unassigned, layer)

        return substeps

    def _distribute_unassigned_pixels(
        self,
        substeps: List[PaintingSubstep],
        unassigned: np.ndarray,
        layer: SemanticPaintingLayer
    ):
        """
        Distribute unassigned pixels to the most appropriate existing substep
        based on their luminosity values.
        """
        if not substeps or np.sum(unassigned) == 0:
            return

        # Get luminosity of unassigned pixels
        unassigned_lum = self.luminosity[unassigned]

        # For each unassigned pixel, find the substep with the closest luminosity range
        unassigned_coords = np.where(unassigned)

        for idx in range(len(unassigned_coords[0])):
            y, x = unassigned_coords[0][idx], unassigned_coords[1][idx]
            pixel_lum = self.luminosity[y, x]

            # Find best matching substep based on luminosity range
            best_substep = None
            best_distance = float('inf')

            for substep in substeps:
                lum_min, lum_max = substep.luminosity_range
                # Distance is 0 if within range, otherwise distance to nearest edge
                if lum_min <= pixel_lum <= lum_max:
                    distance = 0
                else:
                    distance = min(abs(pixel_lum - lum_min), abs(pixel_lum - lum_max))

                if distance < best_distance:
                    best_distance = distance
                    best_substep = substep

            # Add pixel to best matching substep
            if best_substep is not None:
                best_substep.mask[y, x] = True

        # Update coverage for all substeps
        total_region_pixels = max(1, np.sum(layer.mask))
        for substep in substeps:
            substep.coverage = np.sum(substep.mask) / total_region_pixels

    def _get_value_ranges_for_progression(
        self,
        progression,  # ValueProgression enum
        num_substeps: int
    ) -> List[Tuple[float, float]]:
        """Get value ranges for each substep based on progression type."""
        from artisan.vision.analysis.scene_analyzer import ValueProgression

        if progression == ValueProgression.DARK_TO_LIGHT:
            # Standard: dark values first, light values last
            ranges = []
            step_size = 1.0 / num_substeps
            for i in range(num_substeps):
                start = i * step_size
                end = min(1.0, (i + 1) * step_size + 0.1)  # 10% overlap
                ranges.append((start, end))
            return ranges

        elif progression == ValueProgression.LIGHT_TO_DARK:
            # Emission: light values first (glow), then darker edges
            ranges = []
            step_size = 1.0 / num_substeps
            for i in range(num_substeps):
                # Reverse order: start from bright
                start = 1.0 - (i + 1) * step_size
                end = 1.0 - i * step_size + 0.1
                ranges.append((max(0, start), min(1.0, end)))
            return ranges

        elif progression == ValueProgression.GLOW_THEN_EDGES:
            # Aurora/fire: bright core first, then mid, then dark edges
            if num_substeps == 3:
                return [(0.7, 1.0), (0.35, 0.75), (0.0, 0.4)]
            elif num_substeps == 4:
                return [(0.75, 1.0), (0.5, 0.8), (0.25, 0.55), (0.0, 0.3)]
            elif num_substeps >= 5:
                return [(0.8, 1.0), (0.6, 0.85), (0.4, 0.65), (0.2, 0.45), (0.0, 0.25)]
            else:
                return [(0.6, 1.0), (0.0, 0.65)]

        elif progression == ValueProgression.SILHOUETTE_RIM:
            # Silhouette: dark mass first (most of it), then rim highlights
            if num_substeps == 3:
                return [(0.0, 0.3), (0.25, 0.6), (0.7, 1.0)]  # Dark, mid, rim
            elif num_substeps == 4:
                return [(0.0, 0.25), (0.2, 0.45), (0.4, 0.7), (0.75, 1.0)]
            elif num_substeps >= 5:
                return [(0.0, 0.2), (0.15, 0.35), (0.3, 0.5), (0.45, 0.7), (0.75, 1.0)]
            else:
                return [(0.0, 0.5), (0.5, 1.0)]  # Simple: dark mass, then rim

        elif progression == ValueProgression.REFLECTION:
            # Reflections: base, then reflected image
            if num_substeps >= 4:
                return [(0.0, 0.3), (0.25, 0.5), (0.45, 0.75), (0.7, 1.0)]
            elif num_substeps == 3:
                return [(0.0, 0.35), (0.3, 0.65), (0.6, 1.0)]
            else:
                return [(0.0, 0.5), (0.4, 1.0)]

        elif progression == ValueProgression.MUTED_FLAT:
            # Shadow areas: compressed value range
            if num_substeps >= 4:
                return [(0.0, 0.25), (0.2, 0.4), (0.35, 0.55), (0.5, 0.7)]
            elif num_substeps == 3:
                return [(0.0, 0.35), (0.25, 0.55), (0.45, 0.7)]
            else:
                return [(0.0, 0.4), (0.3, 0.6)]

        # Default fallback to dark-to-light
        return self._get_value_ranges_for_progression(ValueProgression.DARK_TO_LIGHT, num_substeps)

    def _get_substep_configs_for_progression(
        self,
        progression,  # ValueProgression enum
        lighting_role,  # LightingRole enum
        subject_type: SubjectType,
        num_substeps: int
    ) -> List[Dict]:
        """Get configuration for each substep based on progression and lighting."""
        from artisan.vision.analysis.scene_analyzer import ValueProgression, LightingRole

        if progression == ValueProgression.LIGHT_TO_DARK:
            return self._configs_light_to_dark(num_substeps)
        elif progression == ValueProgression.GLOW_THEN_EDGES:
            return self._configs_glow_then_edges(num_substeps)
        elif progression == ValueProgression.SILHOUETTE_RIM:
            return self._configs_silhouette_rim(num_substeps)
        elif progression == ValueProgression.REFLECTION:
            return self._configs_reflection(num_substeps)
        elif progression == ValueProgression.MUTED_FLAT:
            return self._configs_muted_flat(num_substeps)
        else:  # DARK_TO_LIGHT (standard)
            return self._configs_dark_to_light(num_substeps)

    def _configs_dark_to_light(self, n: int) -> List[Dict]:
        """Standard dark-to-light substep configs."""
        configs = [
            {'name': 'Dark Values', 'technique': 'blocking', 'brush': '1-inch brush', 'strokes': 'establish shadows'},
            {'name': 'Shadow Mid-tones', 'technique': 'layering', 'brush': '1-inch brush', 'strokes': 'build form'},
            {'name': 'Light Mid-tones', 'technique': 'layering', 'brush': 'filbert', 'strokes': 'blend values'},
            {'name': 'Light Values', 'technique': 'blending', 'brush': 'fan brush', 'strokes': 'soft transitions'},
            {'name': 'Highlights', 'technique': 'highlighting', 'brush': 'script liner', 'strokes': 'sparingly'},
        ]
        return configs[:n]

    def _configs_light_to_dark(self, n: int) -> List[Dict]:
        """Light-to-dark substep configs (for emitters like sky/aurora)."""
        configs = [
            {'name': 'Core Glow', 'technique': 'blocking', 'brush': '2-inch brush', 'strokes': 'soft coverage'},
            {'name': 'Light Falloff', 'technique': 'layering', 'brush': '2-inch brush', 'strokes': 'blend outward'},
            {'name': 'Mid Values', 'technique': 'layering', 'brush': '1-inch brush', 'strokes': 'transition zones'},
            {'name': 'Dark Transition', 'technique': 'blending', 'brush': 'fan brush', 'strokes': 'feather edges'},
            {'name': 'Edge Definition', 'technique': 'detailing', 'brush': 'filbert', 'strokes': 'crisp where needed'},
        ]
        return configs[:n]

    def _configs_glow_then_edges(self, n: int) -> List[Dict]:
        """Glow-then-edges configs (for aurora, fire, neon)."""
        configs = [
            {'name': 'Brightest Core', 'technique': 'blocking', 'brush': 'fan brush', 'strokes': 'soft center glow'},
            {'name': 'Inner Glow', 'technique': 'layering', 'brush': 'fan brush', 'strokes': 'blend from core'},
            {'name': 'Mid Glow', 'technique': 'layering', 'brush': '1-inch brush', 'strokes': 'feather outward'},
            {'name': 'Outer Glow', 'technique': 'blending', 'brush': '1-inch brush', 'strokes': 'soft edges'},
            {'name': 'Dark Boundary', 'technique': 'detailing', 'brush': 'filbert', 'strokes': 'define shape'},
        ]
        return configs[:n]

    def _configs_silhouette_rim(self, n: int) -> List[Dict]:
        """Silhouette with rim light configs (for backlit subjects)."""
        configs = [
            {'name': 'Core Shadow', 'technique': 'blocking', 'brush': '1-inch brush', 'strokes': 'establish mass'},
            {'name': 'Shadow Variation', 'technique': 'layering', 'brush': 'filbert', 'strokes': 'subtle form'},
            {'name': 'Edge Transition', 'technique': 'layering', 'brush': 'fan brush', 'strokes': 'soft edges'},
            {'name': 'Rim Light', 'technique': 'highlighting', 'brush': 'script liner', 'strokes': 'edge highlights'},
            {'name': 'Bright Rim', 'technique': 'highlighting', 'brush': 'script liner', 'strokes': 'backlight glow'},
        ]
        return configs[:n]

    def _configs_reflection(self, n: int) -> List[Dict]:
        """Reflection configs (for water, glass, metal)."""
        configs = [
            {'name': 'Deep Base', 'technique': 'blocking', 'brush': '2-inch brush', 'strokes': 'horizontal coverage'},
            {'name': 'Reflection Darks', 'technique': 'layering', 'brush': '1-inch brush', 'strokes': 'inverted image'},
            {'name': 'Reflection Lights', 'technique': 'layering', 'brush': '1-inch brush', 'strokes': 'muted copy'},
            {'name': 'Surface Texture', 'technique': 'blending', 'brush': 'fan brush', 'strokes': 'horizontal ripples'},
            {'name': 'Sparkle', 'technique': 'highlighting', 'brush': 'palette knife', 'strokes': 'broken horizontal'},
        ]
        return configs[:n]

    def _configs_muted_flat(self, n: int) -> List[Dict]:
        """Muted/flat configs (for shadow areas)."""
        configs = [
            {'name': 'Shadow Core', 'technique': 'blocking', 'brush': '1-inch brush', 'strokes': 'soft coverage'},
            {'name': 'Shadow Variation', 'technique': 'layering', 'brush': 'filbert', 'strokes': 'subtle changes'},
            {'name': 'Shadow Mid', 'technique': 'blending', 'brush': 'fan brush', 'strokes': 'soft transitions'},
            {'name': 'Shadow Edge', 'technique': 'blending', 'brush': 'fan brush', 'strokes': 'blend to light'},
        ]
        return configs[:n]

    def _generate_context_instruction(
        self,
        region_name: str,
        config: Dict,
        progression,  # ValueProgression
        lighting_role,  # LightingRole
        step_index: int,
        total_steps: int
    ) -> str:
        """Generate painting instruction based on lighting context."""
        from artisan.vision.analysis.scene_analyzer import ValueProgression, LightingRole

        if step_index == 0:
            # First step instructions
            if progression == ValueProgression.LIGHT_TO_DARK:
                return f"Start with the brightest values in {region_name}. This establishes the light source - the glow that everything else relates to."
            elif progression == ValueProgression.GLOW_THEN_EDGES:
                return f"Begin with the glowing core of {region_name}. Use soft strokes to establish where the light is strongest."
            elif progression == ValueProgression.SILHOUETTE_RIM:
                return f"Block in the dark mass of {region_name}. This is your silhouette shape against the light."
            else:
                return f"Start with the darkest values in {region_name}. Establish your shadows first - they create the foundation."

        elif step_index == total_steps - 1:
            # Last step instructions
            if lighting_role == LightingRole.SILHOUETTE:
                return f"Add rim lighting to {region_name}. These edge highlights show the backlight wrapping around the form."
            elif lighting_role == LightingRole.EMITTER:
                return f"Define the edges of {region_name}. Keep them soft where the glow fades into darkness."
            elif lighting_role == LightingRole.REFLECTOR:
                return f"Add sparkle and surface highlights to {region_name}. Broken strokes suggest the reflective quality."
            else:
                return f"Add final highlights to {region_name}. Use sparingly - these are the brightest points where light directly hits."

        else:
            # Middle steps
            return f"Build up the {config['name'].lower()} in {region_name}. Work {config['strokes']}."

    def _generate_context_tips(
        self,
        config: Dict,
        lighting_role,  # LightingRole
        depth_layer,  # DepthLayer
        step_index: int,
        total_steps: int
    ) -> List[str]:
        """Generate tips based on lighting context."""
        from artisan.vision.analysis.scene_analyzer import LightingRole, DepthLayer
        import random

        tips = []

        # Lighting role specific tips
        if lighting_role == LightingRole.EMITTER:
            if step_index == 0:
                tips.append("Establish the glow first - this is your light source")
            tips.append("Keep edges soft where light fades")

        elif lighting_role == LightingRole.SILHOUETTE:
            if step_index == 0:
                tips.append("Dark shapes against light - keep values low")
            if step_index == total_steps - 1:
                tips.append("Rim light is warm where backlight hits edges")

        elif lighting_role == LightingRole.REFLECTOR:
            tips.append("Reflections are darker and less saturated than source")
            tips.append("Keep strokes horizontal for water")

        elif lighting_role == LightingRole.SHADOW:
            tips.append("Shadows have color - usually cool blues/purples")
            tips.append("Keep contrast low in shadow areas")

        elif lighting_role == LightingRole.RECEIVER_LIT:
            tips.append("This area receives direct light - higher contrast")

        # Depth layer tips
        if depth_layer == DepthLayer.BACKGROUND:
            tips.append("Less detail - atmospheric perspective softens distant elements")
        elif depth_layer == DepthLayer.FOCAL:
            tips.append("This is your focal point - give it the most attention")
        elif depth_layer in [DepthLayer.FAR_MIDGROUND]:
            tips.append("Medium detail - not too sharp, not too soft")

        # Technique tips
        if config['technique'] == 'blocking':
            tips.append("Work quickly to cover the area - don't overwork")
        elif config['technique'] == 'highlighting':
            tips.append("Less is more - highlights are the finishing touch")

        # Add encouragement
        tips.append(random.choice(self.ENCOURAGEMENTS))

        return tips[:4]  # Limit to 4 tips

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

        Uses PERCENTILE-based thresholds so each depth layer gets roughly
        equal pixel counts, rather than VALUE-based thresholds which can
        create uneven coverage when luminosity distribution is skewed.
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
            # Darkest pixels (by percentile) = back
            depth = params.get("depth", 0.33)
            # Use percentile so we get ~33% of PIXELS, not 33% of value range
            threshold = np.percentile(region_lum, depth * 100)
            return region_mask & (self.luminosity <= threshold)

        elif method == "spatial_mid":
            # Middle pixels (by percentile)
            depth_range = params.get("depth", (0.33, 0.66))
            low_thresh = np.percentile(region_lum, depth_range[0] * 100)
            high_thresh = np.percentile(region_lum, depth_range[1] * 100)
            return region_mask & (self.luminosity > low_thresh) & (self.luminosity <= high_thresh)

        elif method == "spatial_front":
            # Brightest pixels (by percentile) = front
            depth = params.get("depth", 0.66)
            threshold = np.percentile(region_lum, depth * 100)
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

        Uses CUMULATIVE PIXEL COUNT to find cutoff rows, ensuring each
        portion gets roughly equal pixel counts regardless of region shape.
        """
        total_pixels = np.sum(region_mask)
        if total_pixels == 0:
            return np.zeros_like(region_mask)

        # Calculate cumulative pixel count by row (top to bottom)
        pixels_per_row = np.sum(region_mask, axis=1)
        cumulative_pixels = np.cumsum(pixels_per_row)

        result = np.zeros_like(region_mask)
        portion = params.get("portion", 0.33)

        if method == "spatial_top":
            # Top portion - find row where we hit `portion` of total pixels
            target_pixels = total_pixels * portion
            cutoff_row = np.searchsorted(cumulative_pixels, target_pixels)
            result[:cutoff_row + 1, :] = region_mask[:cutoff_row + 1, :]

        elif method == "spatial_bottom":
            # Bottom portion - find row where remaining pixels = `portion` of total
            target_pixels = total_pixels * (1 - portion)
            cutoff_row = np.searchsorted(cumulative_pixels, target_pixels)
            result[cutoff_row:, :] = region_mask[cutoff_row:, :]

        elif method == "spatial_middle":
            # Middle portion - between two percentile cutoffs
            if isinstance(portion, tuple):
                top_p, bottom_p = portion
            else:
                top_p, bottom_p = 0.33, 0.66
            top_target = total_pixels * top_p
            bottom_target = total_pixels * bottom_p
            top_row = np.searchsorted(cumulative_pixels, top_target)
            bottom_row = np.searchsorted(cumulative_pixels, bottom_target)
            result[top_row:bottom_row + 1, :] = region_mask[top_row:bottom_row + 1, :]

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

        For painterly styles (oil, impressionist, watercolor), applies edge blending
        to simulate wet-on-wet painting where colors blend into each other.

        Returns dict with paths for each view type.
        """
        logger.info("Creating step images...")

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

        # Determine if we should apply painterly blending
        blend_styles = ["oil", "impressionist", "watercolor"]
        should_blend = self.paint_style in blend_styles

        cumulative = np.ones((self.h, self.w, 3), dtype=np.uint8) * 255
        painted_mask = np.zeros((self.h, self.w), dtype=bool)  # Track what's been painted
        step_num = 0

        total_substeps = sum(len(l.substeps) for l in self.painting_layers)
        for layer in self.painting_layers:
            for sub in layer.substeps:
                step_num += 1
                safe_name = sub.name.replace(' ', '_').replace('-', '_')
                logger.info(f"  Step {step_num}/{total_substeps}: {sub.name} ({sub.technique})")

                # 1. CUMULATIVE VIEW - Progressive build-up with optional blending
                cumulative[sub.mask] = self.image[sub.mask]

                # Apply edge blending for painterly styles
                if should_blend:
                    cumulative = self._blend_painted_edges(
                        cumulative, sub.mask, painted_mask,
                        blend_radius=8 if self.paint_style == "oil" else 12
                    )

                # Update painted mask
                painted_mask |= sub.mask

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

        logger.info(f"Step images complete: {len(saved_paths['cumulative'])} cumulative, {len(saved_paths['context'])} context, {len(saved_paths['isolated'])} isolated")
        return saved_paths

    def _blend_painted_edges(
        self,
        canvas: np.ndarray,
        current_mask: np.ndarray,
        previous_painted: np.ndarray,
        blend_radius: int = 8
    ) -> np.ndarray:
        """
        Blend edges between newly painted area and previously painted areas.
        Simulates wet-on-wet oil painting where colors blend at boundaries.

        Args:
            canvas: Current cumulative canvas
            current_mask: Mask of area just painted
            previous_painted: Mask of all previously painted areas
            blend_radius: Size of blending zone in pixels

        Returns:
            Canvas with blended edges
        """
        # Find boundary between current and previous painted areas
        # We want to blend where current_mask meets previous_painted
        boundary_zone = cv2.dilate(
            current_mask.astype(np.uint8),
            np.ones((blend_radius, blend_radius), np.uint8),
            iterations=1
        ).astype(bool)

        # Only blend in the overlap with previously painted areas
        blend_zone = boundary_zone & previous_painted & ~current_mask

        if not np.any(blend_zone):
            return canvas

        # Create a smooth blending mask using distance transform
        # Distance from the edge of current_mask
        dist_from_current = cv2.distanceTransform(
            (~current_mask).astype(np.uint8),
            cv2.DIST_L2, 5
        )

        # Normalize to 0-1 within blend radius
        blend_weights = np.clip(dist_from_current / blend_radius, 0, 1)

        # Apply Gaussian blur to the canvas in the blend zone
        blurred = cv2.GaussianBlur(canvas, (blend_radius * 2 + 1, blend_radius * 2 + 1), 0)

        # Blend: use more of the blurred version near the boundary
        result = canvas.copy().astype(np.float32)

        # Only apply in the blend zone
        for c in range(3):
            blend_amount = blend_weights * 0.7  # Max 70% blur
            result[:, :, c] = np.where(
                blend_zone,
                canvas[:, :, c] * (1 - blend_amount) + blurred[:, :, c] * blend_amount,
                result[:, :, c]
            )

        return np.clip(result, 0, 255).astype(np.uint8)

    def create_region_focused_images(self, output_dir: str) -> Dict[str, List[str]]:
        """
        Create region-focused step images that complete each region before moving to next.

        For each semantic region:
        1. region_progress: Shows just this region building up (all substeps)
        2. region_in_context: Shows this region building up with rest of canvas dimmed
        3. canvas_by_region: Full canvas view, completing one region at a time

        This is an ALTERNATIVE view to the standard layer-by-layer approach.
        Use when you want to focus on one subject/area at a time.

        Returns dict with paths organized by region.
        """
        # Create output directories
        region_progress_dir = os.path.join(output_dir, "region_progress")
        region_context_dir = os.path.join(output_dir, "region_in_context")
        canvas_by_region_dir = os.path.join(output_dir, "canvas_by_region")

        os.makedirs(region_progress_dir, exist_ok=True)
        os.makedirs(region_context_dir, exist_ok=True)
        os.makedirs(canvas_by_region_dir, exist_ok=True)

        saved_paths = {
            "region_progress": [],
            "region_in_context": [],
            "canvas_by_region": []
        }

        # Create dimmed version for context views
        dimmed_image = self._create_dimmed_image(self.image, dim_factor=0.35, desaturate=0.7)

        # Determine if we should apply painterly blending
        blend_styles = ["oil", "impressionist", "watercolor"]
        should_blend = self.paint_style in blend_styles
        blend_radius = 8 if self.paint_style == "oil" else 12

        # Canvas that builds up region by region
        canvas = np.ones((self.h, self.w, 3), dtype=np.uint8) * 255
        canvas_painted_mask = np.zeros((self.h, self.w), dtype=bool)
        global_step = 0

        for layer_idx, layer in enumerate(self.painting_layers):
            # Create per-region subdirectory
            region_safe_name = layer.name.replace(' ', '_').replace('-', '_')
            region_subdir = os.path.join(region_progress_dir, f"{layer_idx+1:02d}_{region_safe_name}")
            os.makedirs(region_subdir, exist_ok=True)

            # Region builds up independently (isolated view of just this region)
            region_canvas = np.ones((self.h, self.w, 3), dtype=np.uint8) * 255
            region_painted_mask = np.zeros((self.h, self.w), dtype=bool)

            # Context view: dimmed background + this region building up
            # Start with canvas so far (completed previous regions)
            context_base = canvas.copy()
            # Dim the parts that aren't this region
            other_regions_mask = np.ones((self.h, self.w), dtype=bool)
            other_regions_mask[layer.mask] = False
            context_base[other_regions_mask] = (context_base[other_regions_mask] * 0.5).astype(np.uint8)
            context_painted_mask = np.zeros((self.h, self.w), dtype=bool)

            for sub_idx, sub in enumerate(layer.substeps):
                global_step += 1
                sub_safe_name = sub.name.replace(' ', '_').replace('-', '_')

                # 1. REGION PROGRESS - Just this region building up
                region_canvas[sub.mask] = self.image[sub.mask]
                if should_blend:
                    region_canvas = self._blend_painted_edges(
                        region_canvas, sub.mask, region_painted_mask, blend_radius
                    )
                region_painted_mask |= sub.mask
                region_path = os.path.join(region_subdir, f"step_{sub_idx+1:02d}_{sub_safe_name}.png")
                cv2.imwrite(region_path, cv2.cvtColor(region_canvas, cv2.COLOR_RGB2BGR))
                saved_paths["region_progress"].append(region_path)

                # 2. REGION IN CONTEXT - This region building with rest dimmed
                context_view = context_base.copy()
                context_view[sub.mask] = self.image[sub.mask]
                if should_blend:
                    context_view = self._blend_painted_edges(
                        context_view, sub.mask, context_painted_mask, blend_radius
                    )
                context_painted_mask |= sub.mask
                context_path = os.path.join(region_context_dir,
                    f"step_{global_step:02d}_{region_safe_name}_{sub_idx+1:02d}.png")
                cv2.imwrite(context_path, cv2.cvtColor(context_view, cv2.COLOR_RGB2BGR))
                saved_paths["region_in_context"].append(context_path)

                # Update context base for next substep
                context_base[sub.mask] = self.image[sub.mask]

                # 3. CANVAS BY REGION - Full canvas, completing regions one at a time
                canvas[sub.mask] = self.image[sub.mask]
                if should_blend:
                    canvas = self._blend_painted_edges(
                        canvas, sub.mask, canvas_painted_mask, blend_radius
                    )
                canvas_painted_mask |= sub.mask
                canvas_path = os.path.join(canvas_by_region_dir,
                    f"step_{global_step:02d}_{region_safe_name}_{sub_idx+1:02d}.png")
                cv2.imwrite(canvas_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
                saved_paths["canvas_by_region"].append(canvas_path)

            # Save completed region
            completed_path = os.path.join(region_subdir, "completed.png")
            cv2.imwrite(completed_path, cv2.cvtColor(region_canvas, cv2.COLOR_RGB2BGR))

        # Final complete canvas
        final_path = os.path.join(canvas_by_region_dir, "final_complete.png")
        cv2.imwrite(final_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        saved_paths["canvas_by_region"].append(final_path)

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
                # Scene analysis info
                "lighting_role": layer.lighting_role,
                "value_progression": layer.value_progression,
                "depth_layer": layer.depth_layer,
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

        logger.info(f"Saving outputs to {output_dir}/")
        print(f"\nSaving to {output_dir}/")
        print("-" * 50)

        # Reference
        ref_path = os.path.join(output_dir, "colored_reference.png")
        cv2.imwrite(ref_path, cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR))
        print("  colored_reference.png")

        # Steps with three view types (layer-by-layer progression)
        steps_dir = os.path.join(output_dir, "steps")
        step_paths = self.create_step_images(steps_dir)
        n_cumulative = len(step_paths["cumulative"])
        n_context = len(step_paths["context"])
        n_isolated = len(step_paths["isolated"])
        print(f"  steps/ (layer-by-layer)")
        print(f"    cumulative/ - {n_cumulative} images (progressive build-up)")
        print(f"    context/    - {n_context} images (highlighted in full image)")
        print(f"    isolated/   - {n_isolated} images (region on white canvas)")

        # Region-focused steps (complete each region before moving to next)
        region_steps_dir = os.path.join(output_dir, "steps_by_region")
        region_paths = self.create_region_focused_images(region_steps_dir)
        n_progress = len(region_paths["region_progress"])
        n_context_region = len(region_paths["region_in_context"])
        n_canvas = len(region_paths["canvas_by_region"])
        print(f"  steps_by_region/ (region-focused)")
        print(f"    region_progress/   - {n_progress} images (each region isolated)")
        print(f"    region_in_context/ - {n_context_region} images (region highlighted)")
        print(f"    canvas_by_region/  - {n_canvas} images (full canvas, region by region)")

        # Overview
        self.create_progress_overview(output_dir)
        print("  progress_overview.png")

        # Guide
        logger.info("Generating painting instructions guide...")
        guide = self.create_layer_guide()
        guide_path = os.path.join(output_dir, "painting_guide.json")
        with open(guide_path, 'w') as f:
            json.dump(guide, f, indent=2)
        logger.info(f"Painting guide generated with {len(guide)} steps")
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
                    # New: Scene analysis info
                    "lighting_role": l.lighting_role,
                    "value_progression": l.value_progression,
                    "depth_layer": l.depth_layer,
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

        # Add advanced scene analysis if available
        if self.scene_analysis:
            analysis["advanced_scene_analysis"] = {
                "scene_type": self.scene_analysis.scene_type.value,
                "is_backlit": self.scene_analysis.is_backlit,
                "primary_light_direction": list(self.scene_analysis.primary_light_direction),
                "light_sources": [
                    {
                        "type": ls.type.value,
                        "intensity": round(ls.intensity, 3),
                        "color_temperature": round(ls.color_temperature, 3),
                        "is_primary": ls.is_primary,
                        "position": list(ls.position),
                    }
                    for ls in self.scene_analysis.light_sources
                ],
                "painting_order": self.scene_analysis.painting_order,
                "notes": self.scene_analysis.notes,
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
    substeps: int = 4,
    paint_style: str = "photo",  # photo, oil, impressionist, poster, watercolor
    simplify: int = 0  # 0=none, 1=light, 2=medium, 3=heavy
) -> YOLOBobRossPaint:
    """
    Main entry point.

    Args:
        image_path: Path to input image
        output_dir: Output directory (auto-generated if None)
        model_size: YOLO model size (n, s, m, l, x)
        conf_threshold: YOLO confidence threshold
        substeps: Number of value substeps per region
        paint_style: Painting style effect to apply:
            - photo: Original image (no effect)
            - oil: Rich colors, visible brush strokes
            - impressionist: Soft, dreamy, posterized colors
            - poster: Flat colors with strong edges (paint-by-numbers look)
            - watercolor: Soft, transparent, bleeding edges
        simplify: Detail reduction level (0-3):
            - 0: None (original detail)
            - 1: Light (subtle smoothing)
            - 2: Medium (merges small details)
            - 3: Heavy (significant simplification)
    """
    if output_dir is None:
        base = os.path.splitext(os.path.basename(image_path))[0]
        output_dir = f"output/{base}_yolo_bob_ross"

    painter = YOLOBobRossPaint(
        image_path,
        model_size=model_size,
        conf_threshold=conf_threshold,
        substeps_per_region=substeps,
        paint_style=paint_style,
        simplify=simplify
    )
    painter.process()
    painter.save_all(output_dir)

    return painter


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python yolo_bob_ross_paint.py <image_path> [output_dir] [--style=STYLE] [--simplify=LEVEL]")
        print("Styles: photo (default), oil, impressionist, poster, watercolor")
        print("Simplify: 0 (none), 1 (light), 2 (medium), 3 (heavy)")
        sys.exit(0)

    image_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else None

    # Parse arguments
    paint_style = "photo"
    simplify = 0
    for arg in sys.argv:
        if arg.startswith("--style="):
            paint_style = arg.split("=")[1]
        elif arg.startswith("--simplify="):
            simplify = int(arg.split("=")[1])

    process_image(image_path, output_dir, paint_style=paint_style, simplify=simplify)
