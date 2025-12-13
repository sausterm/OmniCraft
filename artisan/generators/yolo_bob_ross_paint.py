"""
YOLO + Bob Ross Smart Paint by Numbers

Combines:
1. YOLO semantic segmentation for true object boundaries (Dogs, Trees, Grass, etc.)
2. Bob Ross style layer analysis WITHIN each semantic region
3. Back-to-front painting with progressive value building

Result: Each semantic region (Dog, Trees, etc.) gets its own full painting treatment
with darks→midtones→highlights, following Bob Ross methodology.
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

from ..perception.yolo_segmentation import YOLOSemanticSegmenter, SemanticRegion, YOLO_AVAILABLE


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

        # Results
        self.semantic_regions: List[SemanticRegion] = []
        self.painting_layers: List[SemanticPaintingLayer] = []

    def process(self) -> List[SemanticPaintingLayer]:
        """
        Full processing pipeline:
        1. YOLO semantic segmentation
        2. Enforce exclusive boundaries
        3. Create Bob Ross substeps for each semantic region
        4. Generate painting instructions
        """
        print("=" * 60)
        print("YOLO + BOB ROSS SMART PAINT")
        print("=" * 60)

        # Step 1: YOLO segmentation
        print("\n[1/5] Running YOLO semantic segmentation...")
        self.semantic_regions = self.segmenter.segment(
            self.image,
            conf_threshold=self.conf_threshold,
            min_coverage=0.01
        )
        print(f"  Found {len(self.semantic_regions)} semantic regions")

        # Step 2: Enforce exclusive boundaries
        print("\n[2/5] Enforcing exclusive semantic boundaries...")
        self._enforce_exclusive_boundaries()

        # Step 3: Create painting layers with Bob Ross substeps
        print("\n[3/5] Creating Bob Ross painting layers...")
        self._create_painting_layers()

        # Step 4: Generate Bob Ross style instructions
        print("\n[4/5] Generating Bob Ross instructions...")
        self._generate_bob_ross_instructions()

        # Step 5: Summary
        print("\n[5/5] Finalizing painting plan...")
        total_substeps = sum(len(l.substeps) for l in self.painting_layers)

        print(f"\n{'='*60}")
        print("PAINTING PLAN COMPLETE")
        print(f"{'='*60}")
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
        """Convert semantic regions to painting layers with Bob Ross substeps."""
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

            # Create Bob Ross substeps
            layer.substeps = self._create_bob_ross_substeps(layer)

            self.painting_layers.append(layer)
            print(f"  {layer.name}: {len(layer.substeps)} substeps")

    def _create_bob_ross_substeps(self, layer: SemanticPaintingLayer) -> List[PaintingSubstep]:
        """
        Create Bob Ross style substeps for a semantic region.

        Bob Ross methodology:
        1. Block in the darks (establish shadows)
        2. Add the mid-tones (main colors)
        3. Build up the lights
        4. Add highlights and details last
        """
        substeps = []
        region_lum = self.luminosity[layer.mask]

        if len(region_lum) == 0:
            return substeps

        # Determine number of substeps based on region importance
        if layer.is_focal:
            n_steps = min(self.substeps_per_region + 1, 5)  # More steps for focal
        elif layer.coverage > 0.3:
            n_steps = self.substeps_per_region
        else:
            n_steps = max(2, self.substeps_per_region - 1)

        # Bob Ross step names and techniques
        step_configs = {
            2: [
                ("Dark Values", "blocking", 0.0, 0.5),
                ("Highlights", "highlighting", 0.5, 1.0),
            ],
            3: [
                ("Shadows", "shadow", 0.0, 0.33),
                ("Mid-Tones", "layering", 0.33, 0.67),
                ("Highlights", "highlighting", 0.67, 1.0),
            ],
            4: [
                ("Deep Shadows", "blocking", 0.0, 0.25),
                ("Shadow Tones", "shadow", 0.25, 0.5),
                ("Light Tones", "layering", 0.5, 0.75),
                ("Highlights", "highlighting", 0.75, 1.0),
            ],
            5: [
                ("Deep Shadows", "blocking", 0.0, 0.2),
                ("Shadow Areas", "shadow", 0.2, 0.4),
                ("Mid-Tones", "layering", 0.4, 0.6),
                ("Light Areas", "blending", 0.6, 0.8),
                ("Bright Highlights", "highlighting", 0.8, 1.0),
            ],
        }

        configs = step_configs.get(n_steps, step_configs[4])

        # Calculate luminosity percentiles for this region
        percentiles = np.percentile(region_lum, [c[2]*100 for c in configs] + [100])

        for i, (name, technique, pct_min, pct_max) in enumerate(configs):
            lum_min = percentiles[i]
            lum_max = percentiles[i + 1] if i + 1 < len(percentiles) else 1.0

            # Create mask for this luminosity range within the semantic region
            if i == len(configs) - 1:
                sub_mask = layer.mask & (self.luminosity >= lum_min)
            else:
                sub_mask = layer.mask & (self.luminosity >= lum_min) & (self.luminosity < lum_max)

            if np.sum(sub_mask) == 0:
                continue

            # Get properties
            sub_pixels = self.image[sub_mask]
            sub_color = tuple(np.median(sub_pixels, axis=0).astype(int))
            sub_lum = np.mean(self.luminosity[sub_mask])

            substep = PaintingSubstep(
                id=f"{layer.id}_sub{i+1}",
                name=f"{layer.name} - {name}",
                parent_region=layer.name,
                substep_order=i + 1,
                mask=sub_mask,
                coverage=np.sum(sub_mask) / np.sum(layer.mask),
                luminosity_range=(float(lum_min), float(lum_max)),
                dominant_color=sub_color,
                avg_luminosity=float(sub_lum),
                technique=technique,
                brush_suggestion=self.BRUSH_MAP.get(technique, "1-inch brush"),
                stroke_direction=self.STROKE_MAP.get(technique, "natural strokes"),
            )
            substeps.append(substep)

        return substeps

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

    def create_step_images(self, output_dir: str) -> List[str]:
        """Create step-by-step progress images."""
        os.makedirs(output_dir, exist_ok=True)
        saved_paths = []

        cumulative = np.ones((self.h, self.w, 3), dtype=np.uint8) * 255
        step_num = 0

        for layer in self.painting_layers:
            for sub in layer.substeps:
                step_num += 1
                cumulative[sub.mask] = self.image[sub.mask]

                safe_name = sub.name.replace(' ', '_').replace('-', '_')
                path = os.path.join(output_dir, f"step_{step_num:02d}_{safe_name}.png")
                cv2.imwrite(path, cv2.cvtColor(cumulative, cv2.COLOR_RGB2BGR))
                saved_paths.append(path)

        # Final
        final_path = os.path.join(output_dir, "final_complete.png")
        cv2.imwrite(final_path, cv2.cvtColor(cumulative, cv2.COLOR_RGB2BGR))
        saved_paths.append(final_path)

        return saved_paths

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

        # Steps
        steps_dir = os.path.join(output_dir, "steps")
        step_paths = self.create_step_images(steps_dir)
        print(f"  steps/ - {len(step_paths)} images")

        # Overview
        self.create_progress_overview(output_dir)
        print("  progress_overview.png")

        # Guide
        guide = self.create_layer_guide()
        guide_path = os.path.join(output_dir, "painting_guide.json")
        with open(guide_path, 'w') as f:
            json.dump(guide, f, indent=2)
        print("  painting_guide.json")

        # Scene analysis
        analysis = {
            "method": "yolo_bob_ross",
            "num_regions": len(self.painting_layers),
            "total_substeps": sum(len(l.substeps) for l in self.painting_layers),
            "regions": [
                {
                    "name": l.name,
                    "category": l.category,
                    "coverage": round(l.coverage, 4),
                    "confidence": round(l.confidence, 2),
                    "is_focal": l.is_focal,
                    "substeps": len(l.substeps),
                    "techniques": [s.technique for s in l.substeps],
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
