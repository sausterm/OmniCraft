"""
Semantic Paint by Numbers - Depth-ordered, object-aware painting layers.

This creates painting layers that follow proper artistic principles:
1. Paint from back to front (depth perspective)
2. Identify distinct objects semantically (hills, trees, grass, dogs)
3. Group by spatial region, then consider color/texture/lighting
4. Generate student-friendly names and instructions

The fundamental principle: paint back-to-front.
- Furthest elements first (sky, distant hills)
- Then middle ground (trees, grass)
- Then foreground subjects (dogs, people)
- Details and refinements last
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import json
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field

# Updated imports for deprecated location
from artisan.vision.segmentation.semantic import SemanticSegmenter, SegmentationResult, Segment, SegmentType
from artisan.vision.analysis.scene_builder import SceneGraphBuilder
from artisan.vision.segmentation.subject_detector import SubjectDetector
from artisan.vision.analysis.depth_ordering import DepthAnalyzer, DepthZone, SpatialLayer, create_depth_ordered_layers
from artisan.core.scene_graph import SceneGraph, Entity, EntityType
from artisan.core.art_principles import ArtPrinciplesEngine, TECHNIQUES
from artisan.core.constraints import ArtConstraints, Medium, Style, SkillLevel, SubjectDomain


@dataclass
class SubLayer:
    """A sub-layer within a semantic layer, for building up the painting."""
    id: str
    name: str                            # e.g., "Dog - Dark Values"
    parent_layer: str                    # Parent semantic layer name
    sub_order: int                       # Order within parent (1, 2, 3...)

    mask: np.ndarray                     # Binary mask for this sub-layer
    coverage: float                      # Fraction of parent layer

    # Visual properties
    luminosity_range: Tuple[float, float]  # (min, max) luminosity
    dominant_color: Tuple[int, int, int]
    avg_luminosity: float

    # Painting guidance
    technique: str = "layering"
    tips: List[str] = field(default_factory=list)
    instruction: str = ""


@dataclass
class PaintingLayer:
    """A layer to be painted, with all metadata needed for instructions."""
    id: str
    name: str                            # Student-friendly name like "Distant Hills"
    order: int                           # Painting order (1 = first)

    # Spatial/Depth
    mask: np.ndarray                     # Binary mask
    coverage: float                      # Fraction of image
    depth_zone: DepthZone                # FAR_BACKGROUND, BACKGROUND, MIDGROUND, etc.
    depth_value: float                   # 0.0 = furthest, 1.0 = nearest

    # Visual properties
    dominant_color: Tuple[int, int, int]
    avg_luminosity: float
    color_palette: List[Tuple[int, int, int]] = field(default_factory=list)

    # Semantic identification
    semantic_type: str = "region"        # sky, hills, trees, grass, subject, etc.
    description: str = ""                # What this layer represents

    # Painting guidance
    technique: str = "blocking"          # Primary technique
    tips: List[str] = field(default_factory=list)
    brush_direction: Optional[str] = None

    # For subject/focal layers
    is_focal: bool = False
    sub_regions: List[Dict] = field(default_factory=list)

    # SUB-LAYERS: Internal breakdown for building up this layer
    sub_layers: List[SubLayer] = field(default_factory=list)

    # Student instruction
    instruction: str = ""                # Full painting instruction for this layer


class SemanticPaintByNumbers:
    """
    Depth-ordered, semantically-aware paint-by-numbers.

    Key principles:
    1. Paint back-to-front (furthest → nearest)
    2. Identify distinct objects holistically (color + depth + texture + context)
    3. Group by spatial region/object, not just color
    4. Generate student-friendly names and instructions

    The approach:
    - First: segment the image into regions
    - Second: analyze each region for depth cues (position, saturation, detail)
    - Third: identify what each region represents semantically
    - Fourth: order by depth (background first, subjects last)
    - Fifth: generate meaningful instructions for each layer
    """

    def __init__(
        self,
        image_path: str,
        use_sam: bool = False,
        target_layers: int = 12,
        min_coverage: float = 0.01,
        skill_level: str = "intermediate"
    ):
        """
        Initialize semantic paint-by-numbers.

        Args:
            image_path: Path to input image
            use_sam: Whether to use SAM (requires torch + segment-anything)
            target_layers: Target number of painting layers
            min_coverage: Minimum coverage for a layer
            skill_level: beginner/intermediate/advanced
        """
        self.image_path = image_path
        self.use_sam = use_sam
        self.target_layers = target_layers
        self.min_coverage = min_coverage
        self.skill_level = skill_level

        # Load image
        self.image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        self.h, self.w = self.image.shape[:2]

        # Initialize components
        self.segmenter = SemanticSegmenter(device="cpu")
        self.scene_builder = SceneGraphBuilder()
        self.subject_detector = SubjectDetector()
        self.depth_analyzer = DepthAnalyzer()
        self.art_engine = ArtPrinciplesEngine()

        # Results
        self.segmentation: Optional[SegmentationResult] = None
        self.scene_graph: Optional[SceneGraph] = None
        self.subject_analysis = None
        self.spatial_layers: List[SpatialLayer] = []  # From depth analysis
        self.painting_layers: List[PaintingLayer] = []

    def process(self) -> List[PaintingLayer]:
        """
        Run complete analysis and generate depth-ordered painting layers.

        The key insight: we analyze by SPATIAL DEPTH first, then refine
        by color/texture within depth zones. This gives us proper
        back-to-front painting order.

        Returns:
            List of PaintingLayers ordered for painting (back to front)
        """
        print("=" * 60)
        print("SEMANTIC PAINT BY NUMBERS - Depth-Ordered")
        print("=" * 60)

        # Step 1: Semantic segmentation
        print("\n[1/6] Running semantic segmentation...")
        self.segmentation = self.segmenter.segment(
            self.image,
            min_area_ratio=self.min_coverage / 2,
            max_segments=self.target_layers * 3,
            merge_similar=True
        )
        print(f"  Found {self.segmentation.total_segments} segments ({self.segmentation.method})")

        # Step 2: Build scene graph
        print("\n[2/6] Building scene graph...")
        self.scene_graph = self.scene_builder.build(self.segmentation, self.image)
        print(f"  Created {len(self.scene_graph.entities)} entities")

        # Step 3: Detect subjects
        print("\n[3/6] Analyzing subjects...")
        self.subject_analysis = self.subject_detector.analyze(self.image, self.segmentation)
        print(f"  Domain: {self.subject_analysis.primary_domain.value}")
        print(f"  Complexity: {self.subject_analysis.scene_complexity:.2f}")

        # Step 4: DEPTH ANALYSIS - This is the key addition
        print("\n[4/6] Analyzing spatial depth...")
        segments_for_depth = [
            {'mask': seg.mask} for seg in self.segmentation.segments
        ]
        self.spatial_layers = self.depth_analyzer.analyze_depth(
            self.image,
            segments_for_depth
        )
        print(f"  Identified {len(self.spatial_layers)} depth-ordered regions:")
        for sl in self.spatial_layers:
            print(f"    - {sl.name} ({sl.depth_zone.name}): depth={sl.depth_value:.2f}")

        # Step 4b: SUBJECT SEPARATION - Ensure detected subjects get dedicated layers
        print("\n[4b/6] Separating detected subjects...")
        self.spatial_layers = self._separate_subjects_from_environment()

        # Step 5: Convert spatial layers to painting layers with semantic enrichment
        print("\n[5/6] Creating painting layers with semantic context...")
        self.painting_layers = self._create_depth_ordered_layers()
        print(f"  Created {len(self.painting_layers)} painting layers")

        # Step 5b: ENFORCE EXCLUSIVE BOUNDARIES - Critical for proper isolation
        print("\n[5b/6] Enforcing exclusive semantic boundaries...")
        self._enforce_exclusive_boundaries()

        # Step 5c: CREATE SUB-LAYERS within each semantic layer
        print("\n[5c/6] Creating sub-layers for each semantic region...")
        self._create_sub_layers()

        # Step 6: Generate student-friendly instructions
        print("\n[6/6] Generating painting instructions...")
        self._enrich_with_subject_context()
        self._generate_instructions()

        print("\nFINAL PAINTING ORDER (back to front):")
        for layer in self.painting_layers:
            zone_name = layer.depth_zone.name.replace('_', ' ').title()
            print(f"  {layer.order}. {layer.name}")
            print(f"     └─ Zone: {zone_name}, Coverage: {layer.coverage*100:.1f}%, Technique: {layer.technique}")

        return self.painting_layers

    def _separate_subjects_from_environment(self) -> List[SpatialLayer]:
        """
        Ensure detected subjects get their own dedicated layers.

        This prevents dogs from being merged with grass, people with background, etc.
        """
        if not self.subject_analysis or not self.subject_analysis.subjects:
            return self.spatial_layers

        new_layers = []
        subject_masks = []

        # Collect all subject masks
        for subject in self.subject_analysis.subjects:
            subject_masks.append({
                'mask': subject.mask,
                'category': subject.category.value,
                'domain': subject.domain.value,
            })

        # Process each spatial layer
        for layer in self.spatial_layers:
            # Check if this layer significantly overlaps with any subject
            layer_contains_subject = False
            subject_in_layer = None

            for subj in subject_masks:
                overlap = np.sum(layer.mask & subj['mask'])
                layer_total = np.sum(layer.mask)
                subj_total = np.sum(subj['mask'])

                # If subject covers significant portion of layer OR layer covers significant portion of subject
                if layer_total > 0 and (overlap > layer_total * 0.2 or overlap > subj_total * 0.5):
                    layer_contains_subject = True
                    subject_in_layer = subj
                    break

            if layer_contains_subject and subject_in_layer:
                # Split this layer into subject and non-subject parts
                subj_mask = subject_in_layer['mask']

                # Subject portion
                subject_mask = layer.mask & subj_mask
                if np.sum(subject_mask) > 0:
                    # Analyze subject colors
                    subj_pixels = self.image[subject_mask]
                    subj_color = tuple(np.median(subj_pixels, axis=0).astype(int))

                    # Calculate subject depth (slightly in front of environment)
                    subj_y = np.where(subject_mask)[0]
                    subj_depth = layer.depth_value + 0.05  # Subjects slightly forward

                    # Create subject layer
                    subject_name = subject_in_layer['category'].replace('_', ' ').title()
                    # Make "Other Animal" more friendly
                    if subject_name == "Other Animal":
                        subject_name = "Dog"  # Default to dog for pet-type subjects
                    subject_layer = SpatialLayer(
                        id=f"subject_{layer.id}",
                        name=subject_name,
                        depth_zone=DepthZone.FOREGROUND if layer.depth_zone.value >= 2 else layer.depth_zone,
                        depth_value=min(0.95, subj_depth),
                        mask=subject_mask,
                        coverage=np.sum(subject_mask) / (self.h * self.w),
                        y_top=subj_y.min() / self.h if len(subj_y) > 0 else layer.y_top,
                        y_bottom=subj_y.max() / self.h if len(subj_y) > 0 else layer.y_bottom,
                        dominant_color=subj_color,
                        avg_luminosity=self._calc_luminosity(subject_mask),
                        description=f"Paint the {subject_name.lower()} carefully - this is the focal point.",
                        semantic_type="subject",
                        technique="layering",
                        tips=[
                            f"This is your {subject_name.lower()} - the heart of your painting",
                            "Build up form with careful value transitions",
                            "Pay attention to the edges and details",
                            "Take your time - this is what viewers will look at"
                        ],
                    )
                    new_layers.append(subject_layer)

                # Environment portion (what's left)
                env_mask = layer.mask & ~subj_mask
                if np.sum(env_mask) > self.h * self.w * 0.01:  # Only if significant
                    env_pixels = self.image[env_mask]
                    env_color = tuple(np.median(env_pixels, axis=0).astype(int))

                    env_layer = SpatialLayer(
                        id=f"env_{layer.id}",
                        name=self._infer_environment_name(env_mask, layer),
                        depth_zone=layer.depth_zone,
                        depth_value=layer.depth_value,
                        mask=env_mask,
                        coverage=np.sum(env_mask) / (self.h * self.w),
                        y_top=layer.y_top,
                        y_bottom=layer.y_bottom,
                        dominant_color=env_color,
                        avg_luminosity=self._calc_luminosity(env_mask),
                        description=layer.description,
                        semantic_type="environment",
                        technique=layer.technique,
                        tips=layer.tips,
                    )
                    new_layers.append(env_layer)
            else:
                # No subject in this layer, keep as-is
                new_layers.append(layer)

        # Re-sort by depth
        new_layers = sorted(new_layers, key=lambda l: l.depth_value)

        # Re-assign paint order
        for i, layer in enumerate(new_layers):
            layer.paint_order = i + 1

        print(f"  After subject separation: {len(new_layers)} layers")
        for sl in new_layers:
            print(f"    - {sl.name} ({sl.semantic_type}): depth={sl.depth_value:.2f}")

        return new_layers

    def _calc_luminosity(self, mask: np.ndarray) -> float:
        """Calculate average luminosity for a masked region."""
        hsv = cv2.cvtColor(self.image, cv2.COLOR_RGB2HSV)
        return float(np.mean(hsv[mask, 2]) / 255.0)

    def _enforce_exclusive_boundaries(self):
        """
        Ensure each pixel belongs to exactly ONE layer.

        This is critical for proper semantic isolation:
        1. Remove overlaps - pixels in multiple layers go to the FRONTMOST layer
        2. Fill gaps - uncovered pixels go to the nearest layer
        3. Verify 100% coverage with 0% overlap
        """
        h, w = self.h, self.w
        total_pixels = h * w

        # Create ownership map: which layer owns each pixel (-1 = uncovered)
        ownership = np.full((h, w), -1, dtype=np.int32)

        # Assign pixels to layers in REVERSE depth order (front to back)
        # This way, frontmost layers take priority for overlapping pixels
        layers_by_depth = sorted(
            enumerate(self.painting_layers),
            key=lambda x: x[1].depth_value,
            reverse=True  # Front first
        )

        for layer_idx, layer in layers_by_depth:
            # This layer claims all pixels in its mask that aren't already claimed
            # by a layer in front of it
            ownership[layer.mask] = layer_idx

        # Find uncovered pixels
        uncovered_mask = (ownership == -1)
        uncovered_count = np.sum(uncovered_mask)

        if uncovered_count > 0:
            print(f"  Assigning {uncovered_count} uncovered pixels ({100*uncovered_count/total_pixels:.1f}%)...")

            # Assign uncovered pixels to nearest layer by spatial proximity
            # Use distance transform to find nearest labeled pixel
            from scipy import ndimage

            # Create a labeled image from current ownership
            labeled = ownership.copy()
            labeled[uncovered_mask] = -1

            # For each uncovered pixel, find the nearest owned pixel and copy its label
            # We'll do this iteratively by dilating each layer's mask
            for _ in range(max(h, w)):  # Max iterations
                if np.sum(ownership == -1) == 0:
                    break

                # Dilate each layer's current ownership
                still_uncovered = (ownership == -1)
                for layer_idx, layer in enumerate(self.painting_layers):
                    current_owned = (ownership == layer_idx)
                    # Dilate by 1 pixel
                    dilated = ndimage.binary_dilation(current_owned)
                    # Claim newly reachable uncovered pixels
                    new_claims = dilated & still_uncovered
                    ownership[new_claims] = layer_idx
                    still_uncovered = still_uncovered & ~new_claims

        # Now rebuild each layer's mask from the ownership map
        for layer_idx, layer in enumerate(self.painting_layers):
            new_mask = (ownership == layer_idx)
            old_count = np.sum(layer.mask)
            new_count = np.sum(new_mask)

            layer.mask = new_mask
            layer.coverage = new_count / total_pixels

            if abs(new_count - old_count) > 100:  # Significant change
                print(f"    {layer.name}: {old_count} → {new_count} pixels")

        # Verify results
        all_masks = np.zeros((h, w), dtype=np.int32)
        for layer in self.painting_layers:
            all_masks += layer.mask.astype(np.int32)

        overlapping = np.sum(all_masks > 1)
        uncovered = np.sum(all_masks == 0)

        print(f"  Final coverage: {100*(total_pixels-uncovered)/total_pixels:.1f}%")
        print(f"  Overlapping pixels: {overlapping} (should be 0)")
        print(f"  Uncovered pixels: {uncovered} (should be 0)")

        # Remove empty layers
        non_empty = [l for l in self.painting_layers if np.sum(l.mask) > 0]
        removed = len(self.painting_layers) - len(non_empty)
        if removed > 0:
            print(f"  Removed {removed} empty layer(s)")
            self.painting_layers = non_empty

            # Re-number paint order
            for i, layer in enumerate(self.painting_layers):
                layer.order = i + 1

    def _create_sub_layers(self):
        """
        Break each semantic layer into luminosity-based sub-layers.

        Painting order within each semantic layer:
        1. Dark values first (shadows, deepest tones)
        2. Mid values (main body, base colors)
        3. Light values (highlights, brightest areas)

        This creates the "layers of layers" structure.
        """
        # Convert image to HSV for luminosity analysis
        hsv = cv2.cvtColor(self.image, cv2.COLOR_RGB2HSV)
        luminosity = hsv[:, :, 2] / 255.0  # Value channel normalized

        total_sub_layers = 0

        for layer in self.painting_layers:
            # Get luminosity values within this layer's mask
            layer_luminosity = luminosity[layer.mask]

            if len(layer_luminosity) == 0:
                continue

            # Determine number of sub-layers based on layer size and complexity
            # Focal subjects get more sub-layers
            if layer.is_focal or layer.semantic_type == "subject":
                num_sub = 4  # darks, mid-darks, mid-lights, lights
            elif layer.coverage > 0.2:
                num_sub = 3  # darks, mids, lights
            else:
                num_sub = 2  # darks, lights

            # Calculate luminosity percentiles for this layer
            percentiles = np.linspace(0, 100, num_sub + 1)
            thresholds = np.percentile(layer_luminosity, percentiles)

            sub_layer_names = {
                2: ["Dark Values", "Light Values"],
                3: ["Dark Values", "Mid Tones", "Light Values"],
                4: ["Deep Shadows", "Dark Tones", "Light Tones", "Highlights"],
            }

            sub_layer_techniques = {
                "Dark Values": "blocking",
                "Deep Shadows": "shadow",
                "Dark Tones": "layering",
                "Mid Tones": "layering",
                "Light Tones": "layering",
                "Light Values": "highlight",
                "Highlights": "highlight",
            }

            sub_layer_tips = {
                "Dark Values": ["Establish the darkest areas first", "Use thin layers for rich darks"],
                "Deep Shadows": ["Build shadows gradually", "Don't go too dark too fast"],
                "Dark Tones": ["Connect shadows to mid-tones", "Blend edges carefully"],
                "Mid Tones": ["This is the main body color", "Cover the largest area"],
                "Light Tones": ["Build toward the highlights", "Keep edges soft"],
                "Light Values": ["Add lighter values on top", "Don't overwork"],
                "Highlights": ["Add brightest spots last", "Use sparingly for impact"],
            }

            names = sub_layer_names.get(num_sub, ["Values"])
            layer.sub_layers = []

            for i in range(num_sub):
                lum_min = thresholds[i]
                lum_max = thresholds[i + 1]

                # Create mask for this luminosity range within the semantic layer
                sub_mask = layer.mask & (luminosity >= lum_min / 255.0) & (luminosity <= lum_max / 255.0)

                # Handle edge case for last sub-layer (include max value)
                if i == num_sub - 1:
                    sub_mask = layer.mask & (luminosity >= lum_min / 255.0)

                sub_pixel_count = np.sum(sub_mask)
                if sub_pixel_count == 0:
                    continue

                # Get colors for this sub-layer
                sub_pixels = self.image[sub_mask]
                sub_color = tuple(np.median(sub_pixels, axis=0).astype(int))
                sub_lum = np.mean(luminosity[sub_mask])

                sub_name = names[i] if i < len(names) else f"Value {i+1}"

                sub_layer = SubLayer(
                    id=f"{layer.id}_sub{i+1}",
                    name=f"{layer.name} - {sub_name}",
                    parent_layer=layer.name,
                    sub_order=i + 1,
                    mask=sub_mask,
                    coverage=sub_pixel_count / np.sum(layer.mask),
                    luminosity_range=(lum_min / 255.0, lum_max / 255.0),
                    dominant_color=sub_color,
                    avg_luminosity=sub_lum,
                    technique=sub_layer_techniques.get(sub_name, "layering"),
                    tips=sub_layer_tips.get(sub_name, ["Apply carefully"]),
                )

                layer.sub_layers.append(sub_layer)
                total_sub_layers += 1

            if layer.sub_layers:
                print(f"  {layer.name}: {len(layer.sub_layers)} sub-layers")

        print(f"  Total: {total_sub_layers} sub-layers across {len(self.painting_layers)} semantic layers")

    def _infer_environment_name(self, mask: np.ndarray, original_layer: SpatialLayer) -> str:
        """Infer a name for an environment region based on color and position."""
        y_coords = np.where(mask)[0]
        if len(y_coords) == 0:
            return original_layer.name

        y_center = np.mean(y_coords) / self.h

        # Get dominant color
        pixels = self.image[mask]
        avg_color = np.mean(pixels, axis=0)

        # Green detection (grass)
        is_green = avg_color[1] > avg_color[0] * 1.2 and avg_color[1] > avg_color[2] * 1.1

        # Blue detection (sky)
        is_blue = avg_color[2] > avg_color[0] * 1.2 and avg_color[2] > avg_color[1]

        # Position and color based naming
        if y_center < 0.3 and is_blue:
            return "Sky"
        elif y_center < 0.4:
            return "Distant Hills" if not is_green else "Distant Trees"
        elif y_center < 0.6:
            return "Tree Line" if avg_color.mean() < 80 else "Field"
        elif is_green:
            return "Grass Field"
        else:
            return "Ground"

    def _create_depth_ordered_layers(self) -> List[PaintingLayer]:
        """
        Convert depth-analyzed spatial layers into painting layers.

        The spatial layers are already ordered by depth (back to front).
        We enrich them with color palette and semantic context.
        """
        layers = []

        for i, spatial in enumerate(self.spatial_layers):
            # Skip tiny layers
            if spatial.coverage < self.min_coverage:
                continue

            # Extract color palette from this region
            masked_pixels = self.image[spatial.mask]
            if len(masked_pixels) == 0:
                continue

            color_palette = self._extract_color_palette(masked_pixels, num_colors=5)

            # Create painting layer from spatial layer
            layer = PaintingLayer(
                id=spatial.id,
                name=spatial.name,
                order=spatial.paint_order,
                mask=spatial.mask,
                coverage=spatial.coverage,
                depth_zone=spatial.depth_zone,
                depth_value=spatial.depth_value,
                dominant_color=spatial.dominant_color,
                avg_luminosity=spatial.avg_luminosity,
                color_palette=color_palette,
                semantic_type=spatial.semantic_type,
                description=spatial.description,
                technique=spatial.technique,
                tips=spatial.tips.copy(),
                is_focal=spatial.depth_zone in [DepthZone.FOREGROUND, DepthZone.NEAR_FOREGROUND],
            )

            layers.append(layer)

        # Merge if we have too many
        if len(layers) > self.target_layers:
            layers = self._merge_by_depth_zone(layers)

        return layers

    def _extract_color_palette(self, pixels: np.ndarray, num_colors: int = 5) -> List[Tuple[int, int, int]]:
        """Extract dominant colors from pixels using k-means."""
        from sklearn.cluster import KMeans

        if len(pixels) < num_colors:
            return [tuple(np.median(pixels, axis=0).astype(int))]

        # Subsample for speed
        if len(pixels) > 1000:
            indices = np.random.choice(len(pixels), 1000, replace=False)
            sample = pixels[indices]
        else:
            sample = pixels

        try:
            kmeans = KMeans(n_clusters=min(num_colors, len(sample)), random_state=42, n_init=10)
            kmeans.fit(sample)
            colors = [tuple(c.astype(int)) for c in kmeans.cluster_centers_]
            return colors
        except:
            return [tuple(np.median(pixels, axis=0).astype(int))]

    def _merge_by_depth_zone(self, layers: List[PaintingLayer]) -> List[PaintingLayer]:
        """Merge layers within the same depth zone if we have too many."""
        while len(layers) > self.target_layers:
            # Find smallest non-focal layer
            candidates = [l for l in layers if not l.is_focal]
            if not candidates:
                break

            smallest = min(candidates, key=lambda l: l.coverage)

            # Find best layer to merge into (same depth zone, similar luminosity)
            others = [l for l in layers if l.id != smallest.id]
            same_zone = [l for l in others if l.depth_zone == smallest.depth_zone]

            if same_zone:
                merge_candidates = same_zone
            else:
                merge_candidates = others

            if not merge_candidates:
                break

            best_match = min(merge_candidates, key=lambda l: (
                abs(l.depth_value - smallest.depth_value) * 2 +
                abs(l.avg_luminosity - smallest.avg_luminosity)
            ))

            # Merge masks and update coverage
            best_match.mask = best_match.mask | smallest.mask
            best_match.coverage = np.sum(best_match.mask) / (self.h * self.w)

            layers = [l for l in layers if l.id != smallest.id]

        # Re-number order
        for i, layer in enumerate(layers):
            layer.order = i + 1

        return layers

    def _enrich_with_subject_context(self):
        """Enrich layer names and descriptions with detected subject info."""
        if not self.subject_analysis:
            return

        for layer in self.painting_layers:
            # Check if this layer contains any detected subjects
            for subject in self.subject_analysis.subjects:
                overlap = np.sum(layer.mask & subject.mask)
                total = np.sum(layer.mask)
                if total > 0 and overlap > total * 0.3:
                    # This layer contains a subject
                    # Use category (SubjectCategory) not subject_type
                    subject_name = subject.category.value.replace('_', ' ').title()

                    # Update name if it's generic
                    if 'subject' in layer.name.lower() or 'foreground' in layer.name.lower():
                        layer.name = f"{subject_name}"
                        layer.semantic_type = "subject"

                    # Add subject-specific tips
                    if subject.recommended_techniques:
                        layer.tips.extend(subject.recommended_techniques[:2])

                    layer.is_focal = True
                    break

    def _generate_instructions(self):
        """Generate student-friendly painting instructions for each layer."""
        for layer in self.painting_layers:
            layer.instruction = self._create_layer_instruction(layer)

    def _create_layer_instruction(self, layer: PaintingLayer) -> str:
        """Create a complete painting instruction for a layer."""
        parts = []

        # Start with the action
        zone_intro = {
            DepthZone.FAR_BACKGROUND: "Starting with the furthest elements",
            DepthZone.BACKGROUND: "Moving to the background",
            DepthZone.MIDGROUND: "Now working on the middle ground",
            DepthZone.FOREGROUND: "Moving forward to the main subject area",
            DepthZone.NEAR_FOREGROUND: "Finally, the closest elements",
        }

        intro = zone_intro.get(layer.depth_zone, "Next")
        parts.append(f"{intro}, paint the {layer.name.lower()}.")

        # Add technique guidance
        if layer.technique == "wet_blend":
            parts.append("Work quickly while the paint is wet to blend colors smoothly.")
        elif layer.technique == "blocking":
            parts.append("Use broad strokes to establish the shapes. Don't worry about details yet.")
        elif layer.technique == "layering":
            parts.append("Build up the form gradually with careful value transitions.")
        elif layer.technique == "shadow":
            parts.append("Build dark values slowly. You may need multiple thin coats for rich darks.")

        # Add luminosity guidance
        if layer.avg_luminosity > 0.7:
            parts.append("This is a bright area - mix with white and apply with confident strokes.")
        elif layer.avg_luminosity < 0.3:
            parts.append("This is a dark area - take your time building up the darks.")

        # Add focal area guidance
        if layer.is_focal:
            parts.append("This is a focal area of your painting - take your time and pay attention to details.")

        # Add description if available
        if layer.description and layer.description not in ' '.join(parts):
            parts.append(layer.description)

        return ' '.join(parts)

    def create_step_images(self, output_dir: str, use_sub_layers: bool = True) -> List[str]:
        """
        Create step-by-step images showing layers being painted.

        If use_sub_layers is True, each semantic layer is built up through
        its sub-layers (darks → mids → lights), maintaining semantic isolation.

        Returns:
            List of saved file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        saved_paths = []

        # Start with white canvas
        cumulative = np.ones((self.h, self.w, 3), dtype=np.uint8) * 255

        step_num = 0

        for layer in self.painting_layers:
            if use_sub_layers and layer.sub_layers:
                # Paint this semantic layer through its sub-layers
                for sub in layer.sub_layers:
                    step_num += 1
                    # Paint ONLY this sub-layer's pixels
                    cumulative[sub.mask] = self.image[sub.mask]

                    # Save progress image
                    safe_name = sub.name.replace(' ', '_').replace('-', '_')
                    progress_path = os.path.join(output_dir, f"step_{step_num:02d}_{safe_name}.png")
                    cv2.imwrite(progress_path, cv2.cvtColor(cumulative, cv2.COLOR_RGB2BGR))
                    saved_paths.append(progress_path)
            else:
                # Paint entire semantic layer at once (fallback)
                step_num += 1
                cumulative[layer.mask] = self.image[layer.mask]

                safe_name = layer.name.replace(' ', '_')
                progress_path = os.path.join(output_dir, f"step_{step_num:02d}_{safe_name}.png")
                cv2.imwrite(progress_path, cv2.cvtColor(cumulative, cv2.COLOR_RGB2BGR))
                saved_paths.append(progress_path)

        # Save final
        final_path = os.path.join(output_dir, "final_complete.png")
        cv2.imwrite(final_path, cv2.cvtColor(cumulative, cv2.COLOR_RGB2BGR))
        saved_paths.append(final_path)

        return saved_paths

    def create_progress_overview(self, output_dir: str, use_sub_layers: bool = True) -> str:
        """Create grid showing all progress steps with sub-layers."""
        # Count total steps (including sub-layers)
        total_steps = 0
        step_info = []  # [(name, is_sub_layer, parent_name), ...]

        for layer in self.painting_layers:
            if use_sub_layers and layer.sub_layers:
                for sub in layer.sub_layers:
                    total_steps += 1
                    step_info.append((sub.name, True, layer.name))
            else:
                total_steps += 1
                step_info.append((layer.name, False, None))

        # Calculate grid layout
        cols = min(5, total_steps + 1)  # More columns for sub-layers
        rows = (total_steps + 1 + cols - 1) // cols

        # Thumbnail size (smaller for more steps)
        thumb_h = 200 if total_steps > 12 else 250
        thumb_w = int(thumb_h * self.w / self.h)
        margin = 8
        label_h = 35

        # Create canvas
        canvas_w = cols * (thumb_w + margin) + margin
        canvas_h = rows * (thumb_h + label_h + margin) + margin + 50  # Extra for title
        canvas = Image.new('RGB', (canvas_w, canvas_h), 'white')
        draw = ImageDraw.Draw(canvas)

        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 10)
            font_title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        except:
            font = font_title = ImageFont.load_default()

        # Title
        n_semantic = len(self.painting_layers)
        title = f"SEMANTIC PAINT BY NUMBERS - {n_semantic} Regions, {total_steps} Steps"
        draw.text((margin, margin), title, fill='darkblue', font=font_title)

        # Build cumulative images using sub-layers
        cumulative = np.ones((self.h, self.w, 3), dtype=np.uint8) * 255

        step_num = 0
        for layer in self.painting_layers:
            if use_sub_layers and layer.sub_layers:
                for sub in layer.sub_layers:
                    cumulative[sub.mask] = self.image[sub.mask]

                    # Create thumbnail
                    thumb = Image.fromarray(cumulative).resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)

                    # Position
                    col = step_num % cols
                    row = step_num // cols
                    x = margin + col * (thumb_w + margin)
                    y = margin + 50 + row * (thumb_h + label_h + margin)

                    canvas.paste(thumb, (x, y))

                    # Label (truncate for space)
                    short_name = sub.name[:25] if len(sub.name) > 25 else sub.name
                    label = f"{step_num + 1}. {short_name}"
                    draw.text((x, y + thumb_h + 2), label, fill='black', font=font)

                    step_num += 1
            else:
                cumulative[layer.mask] = self.image[layer.mask]

                thumb = Image.fromarray(cumulative).resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)

                col = step_num % cols
                row = step_num // cols
                x = margin + col * (thumb_w + margin)
                y = margin + 50 + row * (thumb_h + label_h + margin)

                canvas.paste(thumb, (x, y))

                label = f"{step_num + 1}. {layer.name[:20]}"
                draw.text((x, y + thumb_h + 2), label, fill='black', font=font)

                step_num += 1

        # Save
        overview_path = os.path.join(output_dir, "progress_overview.png")
        canvas.save(overview_path)

        return overview_path

    def create_layer_guide(self) -> List[Dict]:
        """Generate layer-by-layer painting guide with depth-aware ordering."""
        guide = []

        # Preparation step
        depth_zones_present = set(l.depth_zone.name for l in self.painting_layers)
        guide.append({
            "step": 0,
            "name": "Preparation",
            "type": "setup",
            "description": f"Prepare your canvas and materials. This painting has {len(self.painting_layers)} layers, painted from back to front.",
            "technique": "setup",
            "painting_order": "We'll paint from the furthest elements (background) to the nearest (foreground subjects).",
            "depth_zones": list(depth_zones_present),
            "tips": [
                f"Subject domain: {self.subject_analysis.primary_domain.value if self.subject_analysis else 'mixed'}",
                f"Scene complexity: {self.subject_analysis.scene_complexity:.1f}/1.0" if self.subject_analysis else "",
                "Have clean water and paper towels ready",
                "Lay out your paints in organized fashion",
                "We'll work back-to-front: background first, then subjects"
            ]
        })

        # Group layers by depth zone for clarity in guide
        current_zone = None

        for layer in self.painting_layers:
            entry = {
                "step": layer.order,
                "name": layer.name,
                "type": "paint_layer",
                "depth_zone": layer.depth_zone.name,
                "depth_value": round(float(layer.depth_value), 2),
                "semantic_type": layer.semantic_type,
                "coverage": f"{layer.coverage * 100:.1f}%",
                "dominant_color": [int(c) for c in layer.dominant_color],
                "color_palette": [[int(c) for c in color] for color in layer.color_palette[:3]],
                "avg_luminosity": round(float(layer.avg_luminosity), 2),
                "technique": layer.technique,
                "is_focal": layer.is_focal,
                "tips": layer.tips,
                "instruction": layer.instruction,
                "description": layer.description
            }

            # Add zone transition note if we're entering a new zone
            if layer.depth_zone != current_zone:
                zone_name = layer.depth_zone.name.replace('_', ' ').title()
                entry["zone_transition"] = f"Now entering: {zone_name}"
                current_zone = layer.depth_zone

            guide.append(entry)

        # Finishing step
        guide.append({
            "step": len(self.painting_layers) + 1,
            "name": "Final Details & Refinement",
            "type": "finishing",
            "description": "Step back, assess, and add final touches",
            "technique": "detail",
            "tips": [
                "Step back and assess the overall composition",
                "Check that depth is clear - background should recede, subjects should pop",
                "Add highlights to the brightest areas",
                "Deepen shadows where needed for contrast",
                "Refine edges - soften background edges, sharpen subject edges",
                "Sign your work!"
            ]
        })

        return guide

    def save_all(self, output_dir: str):
        """Save all outputs."""
        os.makedirs(output_dir, exist_ok=True)

        print(f"\nSaving outputs to {output_dir}/")
        print("-" * 50)

        # Colored reference
        ref_path = os.path.join(output_dir, "colored_reference.png")
        cv2.imwrite(ref_path, cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR))
        print(f"  colored_reference.png - Original image")

        # Segmentation visualization
        seg_viz = self._create_segmentation_viz()
        seg_path = os.path.join(output_dir, "segmentation.png")
        cv2.imwrite(seg_path, cv2.cvtColor(seg_viz, cv2.COLOR_RGB2BGR))
        print(f"  segmentation.png - Semantic segmentation")

        # Step images
        steps_dir = os.path.join(output_dir, "steps")
        step_paths = self.create_step_images(steps_dir)
        print(f"  steps/ - {len(step_paths)} progress images")

        # Progress overview
        overview_path = self.create_progress_overview(output_dir)
        print(f"  progress_overview.png - Grid of all steps")

        # Layer guide JSON
        guide = self.create_layer_guide()
        guide_path = os.path.join(output_dir, "layer_guide.json")
        with open(guide_path, 'w') as f:
            json.dump(guide, f, indent=2, default=str)
        print(f"  layer_guide.json - Layer-by-layer instructions")

        # Scene analysis JSON with depth ordering info
        analysis = {
            "domain": self.subject_analysis.primary_domain.value if self.subject_analysis else "mixed",
            "complexity": float(self.subject_analysis.scene_complexity) if self.subject_analysis else 0.5,
            "num_layers": len(self.painting_layers),
            "segmentation_method": self.segmentation.method if self.segmentation else "unknown",
            "painting_approach": "back_to_front",
            "depth_zones_used": list(set(l.depth_zone.name for l in self.painting_layers)),
            "layers": [
                {
                    "order": l.order,
                    "name": l.name,
                    "coverage": round(float(l.coverage), 4),
                    "depth_zone": l.depth_zone.name,
                    "depth_value": round(float(l.depth_value), 3),
                    "technique": l.technique,
                    "semantic_type": l.semantic_type,
                    "is_focal": l.is_focal,
                    "dominant_color": [int(c) for c in l.dominant_color],
                    "instruction": l.instruction[:100] + "..." if len(l.instruction) > 100 else l.instruction,
                }
                for l in self.painting_layers
            ]
        }
        analysis_path = os.path.join(output_dir, "scene_analysis.json")
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"  scene_analysis.json - Scene understanding data")

        print("-" * 50)
        print(f"Done! {len(os.listdir(output_dir))} items generated.")

        return output_dir

    def _create_segmentation_viz(self) -> np.ndarray:
        """Create visualization of painting layers."""
        viz = self.image.copy().astype(np.float32)

        # Color palette for layers
        np.random.seed(42)
        colors = np.random.randint(50, 255, (len(self.painting_layers), 3))

        for i, layer in enumerate(self.painting_layers):
            color = colors[i]
            alpha = 0.3

            for c in range(3):
                viz[:, :, c] = np.where(
                    layer.mask,
                    viz[:, :, c] * (1 - alpha) + color[c] * alpha,
                    viz[:, :, c]
                )

            # Draw boundary
            mask_uint8 = layer.mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(viz.astype(np.uint8), contours, -1, (255, 255, 255), 1)

        return viz.astype(np.uint8)


def process_image(
    image_path: str,
    output_dir: str = None,
    use_sam: bool = False,
    target_layers: int = 12
) -> SemanticPaintByNumbers:
    """
    Main entry point for semantic paint-by-numbers.

    Args:
        image_path: Path to input image
        output_dir: Output directory (default: based on image name)
        use_sam: Whether to use SAM segmentation
        target_layers: Target number of painting layers

    Returns:
        SemanticPaintByNumbers instance
    """
    if output_dir is None:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_dir = f"output/{base_name}_semantic"

    spbn = SemanticPaintByNumbers(
        image_path,
        use_sam=use_sam,
        target_layers=target_layers
    )
    spbn.process()
    spbn.save_all(output_dir)

    return spbn


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Semantic Paint by Numbers")
        print("=" * 40)
        print("Usage: python semantic_paint_by_numbers.py <image_path> [output_dir] [--sam]")
        sys.exit(0)

    image_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else None
    use_sam = '--sam' in sys.argv

    process_image(image_path, output_dir, use_sam=use_sam)
