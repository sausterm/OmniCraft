"""
YOLO Smart Paint by Numbers - True semantic segmentation with smart layering.

Combines:
1. YOLO for detecting known objects (dogs, people, etc.) with precise masks
2. Color/position analysis for environment (sky, grass, trees)
3. Smart sub-layering within each semantic region (darks → lights)

This gives us:
- True semantic boundaries (dog pixels ONLY in dog layer)
- Intelligent environment segmentation
- Progressive building within each region
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import json
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field

from ..perception.yolo_segmentation import YOLOSemanticSegmenter, SemanticRegion, YOLO_AVAILABLE


@dataclass
class SubLayer:
    """A sub-layer within a semantic region for progressive painting."""
    id: str
    name: str
    parent_name: str
    sub_order: int
    mask: np.ndarray
    coverage: float  # Fraction of parent
    luminosity_range: Tuple[float, float]
    dominant_color: Tuple[int, int, int]
    avg_luminosity: float
    technique: str = "layering"
    tips: List[str] = field(default_factory=list)


@dataclass
class SmartSemanticLayer:
    """A semantic layer with sub-layers for progressive painting."""
    id: str
    name: str
    category: str  # subject, environment, background
    paint_order: int

    mask: np.ndarray
    coverage: float
    confidence: float
    depth_estimate: float

    dominant_color: Tuple[int, int, int]
    avg_luminosity: float
    is_focal: bool

    sub_layers: List[SubLayer] = field(default_factory=list)

    technique: str = "layering"
    tips: List[str] = field(default_factory=list)
    instruction: str = ""


class YOLOSmartPaintByNumbers:
    """
    Smart paint-by-numbers using YOLO semantic segmentation.

    Each semantic region (Dog, Trees, Grass, Sky) becomes a layer,
    and each layer is built up through luminosity-based sub-layers.
    """

    def __init__(
        self,
        image_path: str,
        model_size: str = "m",
        conf_threshold: float = 0.3,
        min_coverage: float = 0.01
    ):
        """
        Initialize YOLO smart paint-by-numbers.

        Args:
            image_path: Path to input image
            model_size: YOLO model size (n/s/m/l/x)
            conf_threshold: Minimum detection confidence
            min_coverage: Minimum coverage for environment regions
        """
        self.image_path = image_path
        self.model_size = model_size
        self.conf_threshold = conf_threshold
        self.min_coverage = min_coverage

        # Load image
        self.image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        self.h, self.w = self.image.shape[:2]

        # Initialize segmenter
        self.segmenter = YOLOSemanticSegmenter(model_size=model_size)

        # Results
        self.semantic_regions: List[SemanticRegion] = []
        self.layers: List[SmartSemanticLayer] = []

    def process(self) -> List[SmartSemanticLayer]:
        """
        Run complete pipeline:
        1. YOLO semantic segmentation
        2. Enforce exclusive boundaries
        3. Create sub-layers within each semantic region
        4. Generate painting instructions
        """
        print("=" * 60)
        print("YOLO SMART PAINT BY NUMBERS")
        print("=" * 60)

        # Step 1: Semantic segmentation
        print("\n[1/4] Running YOLO semantic segmentation...")
        self.semantic_regions = self.segmenter.segment(
            self.image,
            conf_threshold=self.conf_threshold,
            min_coverage=self.min_coverage
        )
        print(f"  Found {len(self.semantic_regions)} semantic regions")

        # Step 2: Enforce exclusive boundaries
        print("\n[2/4] Enforcing exclusive semantic boundaries...")
        self._enforce_exclusive_boundaries()

        # Step 3: Convert to layers with sub-layers
        print("\n[3/4] Creating smart layers with sub-layers...")
        self._create_smart_layers()

        # Step 4: Generate instructions
        print("\n[4/4] Generating painting instructions...")
        self._generate_instructions()

        # Print summary
        total_sub = sum(len(l.sub_layers) for l in self.layers)
        print(f"\nFINAL RESULT:")
        print(f"  {len(self.layers)} semantic layers")
        print(f"  {total_sub} total sub-layers")
        print("\nPAINTING ORDER (back to front):")
        for layer in self.layers:
            focal = " [FOCAL]" if layer.is_focal else ""
            print(f"  {layer.paint_order}. {layer.name} ({layer.category}){focal}")
            print(f"     Coverage: {layer.coverage*100:.1f}%, Sub-layers: {len(layer.sub_layers)}")

        return self.layers

    def _enforce_exclusive_boundaries(self):
        """Ensure each pixel belongs to exactly one region."""
        h, w = self.h, self.w
        total_pixels = h * w

        # Create ownership map
        ownership = np.full((h, w), -1, dtype=np.int32)

        # Assign in reverse depth order (front layers take priority)
        regions_by_depth = sorted(
            enumerate(self.semantic_regions),
            key=lambda x: x[1].depth_estimate,
            reverse=True
        )

        for idx, region in regions_by_depth:
            ownership[region.mask] = idx

        # Fill uncovered pixels
        uncovered = (ownership == -1)
        uncovered_count = np.sum(uncovered)

        if uncovered_count > 0:
            print(f"  Assigning {uncovered_count} uncovered pixels ({100*uncovered_count/total_pixels:.1f}%)...")
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

        # Update region masks
        for idx, region in enumerate(self.semantic_regions):
            new_mask = (ownership == idx)
            region.mask = new_mask
            region.coverage = np.sum(new_mask) / total_pixels

        # Remove empty regions
        self.semantic_regions = [r for r in self.semantic_regions if np.sum(r.mask) > 0]

        # Verify
        all_masks = sum(r.mask.astype(int) for r in self.semantic_regions)
        overlap = np.sum(all_masks > 1)
        uncovered = np.sum(all_masks == 0)
        print(f"  Coverage: {100*(total_pixels-uncovered)/total_pixels:.1f}%")
        print(f"  Overlapping: {overlap} (should be 0)")

    def _create_smart_layers(self):
        """Convert semantic regions to smart layers with sub-layers."""
        hsv = cv2.cvtColor(self.image, cv2.COLOR_RGB2HSV)
        luminosity = hsv[:, :, 2] / 255.0

        self.layers = []

        for i, region in enumerate(self.semantic_regions):
            # Create base layer
            layer = SmartSemanticLayer(
                id=region.id,
                name=region.name,
                category=region.category,
                paint_order=region.paint_order,
                mask=region.mask,
                coverage=region.coverage,
                confidence=region.confidence,
                depth_estimate=region.depth_estimate,
                dominant_color=region.dominant_color,
                avg_luminosity=region.avg_luminosity,
                is_focal=region.is_focal,
            )

            # Create sub-layers based on luminosity
            layer.sub_layers = self._create_sub_layers(layer, luminosity)

            self.layers.append(layer)
            print(f"  {layer.name}: {len(layer.sub_layers)} sub-layers")

    def _create_sub_layers(self, layer: SmartSemanticLayer, luminosity: np.ndarray) -> List[SubLayer]:
        """Create luminosity-based sub-layers within a semantic layer."""
        sub_layers = []

        layer_lum = luminosity[layer.mask]
        if len(layer_lum) == 0:
            return sub_layers

        # Number of sub-layers based on layer type
        if layer.is_focal:
            num_sub = 4  # More detail for focal areas
        elif layer.coverage > 0.3:
            num_sub = 3
        else:
            num_sub = 2

        # Calculate luminosity thresholds
        percentiles = np.linspace(0, 100, num_sub + 1)
        thresholds = np.percentile(layer_lum, percentiles)

        # Sub-layer names
        names = {
            2: ["Dark Values", "Light Values"],
            3: ["Shadows", "Mid Tones", "Highlights"],
            4: ["Deep Shadows", "Dark Tones", "Light Tones", "Highlights"],
        }

        techniques = {
            "Deep Shadows": "shadow",
            "Shadows": "shadow",
            "Dark Values": "blocking",
            "Dark Tones": "layering",
            "Mid Tones": "layering",
            "Light Tones": "layering",
            "Light Values": "highlight",
            "Highlights": "highlight",
        }

        tips = {
            "Deep Shadows": ["Build shadows gradually", "Use thin layers"],
            "Shadows": ["Establish dark areas first", "Don't go too dark too fast"],
            "Dark Values": ["Block in the darkest areas", "Create depth"],
            "Dark Tones": ["Connect shadows to mid-tones", "Blend carefully"],
            "Mid Tones": ["This is the main body", "Cover the largest area"],
            "Light Tones": ["Build toward highlights", "Keep edges soft"],
            "Light Values": ["Add lighter values", "Don't overwork"],
            "Highlights": ["Add brightest spots last", "Use sparingly"],
        }

        sub_names = names.get(num_sub, ["Values"])

        for i in range(num_sub):
            lum_min = thresholds[i] / 255.0
            lum_max = thresholds[i + 1] / 255.0 if i < num_sub - 1 else 1.0

            # Create mask for this luminosity range
            if i == num_sub - 1:
                sub_mask = layer.mask & (luminosity >= lum_min)
            else:
                sub_mask = layer.mask & (luminosity >= lum_min) & (luminosity < lum_max)

            if np.sum(sub_mask) == 0:
                continue

            # Get properties
            sub_pixels = self.image[sub_mask]
            sub_color = tuple(np.median(sub_pixels, axis=0).astype(int))
            sub_lum = np.mean(luminosity[sub_mask])

            sub_name = sub_names[i] if i < len(sub_names) else f"Value {i+1}"

            sub_layer = SubLayer(
                id=f"{layer.id}_sub{i+1}",
                name=f"{layer.name} - {sub_name}",
                parent_name=layer.name,
                sub_order=i + 1,
                mask=sub_mask,
                coverage=np.sum(sub_mask) / np.sum(layer.mask),
                luminosity_range=(lum_min, lum_max),
                dominant_color=sub_color,
                avg_luminosity=sub_lum,
                technique=techniques.get(sub_name, "layering"),
                tips=tips.get(sub_name, ["Apply carefully"]),
            )
            sub_layers.append(sub_layer)

        return sub_layers

    def _generate_instructions(self):
        """Generate painting instructions for each layer."""
        for layer in self.layers:
            parts = []

            if layer.category == "background":
                parts.append(f"Start with the {layer.name.lower()} in the background.")
            elif layer.category == "environment":
                parts.append(f"Paint the {layer.name.lower()}.")
            else:
                parts.append(f"Now paint the {layer.name.lower()} - this is a focal subject.")

            if layer.is_focal:
                parts.append("Take your time with this area - it's what viewers will focus on.")

            layer.instruction = " ".join(parts)

            # Add technique tips
            if layer.category == "subject":
                layer.tips = [
                    "Build up form gradually",
                    "Pay attention to edges and details",
                    "Use the sub-layers to create depth"
                ]
            elif layer.category == "background":
                layer.tips = [
                    "Keep edges soft",
                    "Less detail than foreground",
                    "Work quickly"
                ]
            else:
                layer.tips = [
                    "Connect background to foreground",
                    "Medium level of detail"
                ]

    def create_step_images(self, output_dir: str) -> List[str]:
        """Create step-by-step progress images using sub-layers."""
        os.makedirs(output_dir, exist_ok=True)
        saved_paths = []

        cumulative = np.ones((self.h, self.w, 3), dtype=np.uint8) * 255
        step_num = 0

        for layer in self.layers:
            for sub in layer.sub_layers:
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
        """Create grid overview of all steps."""
        total_steps = sum(len(l.sub_layers) for l in self.layers)

        cols = min(5, total_steps + 1)
        rows = (total_steps + 1 + cols - 1) // cols

        thumb_h = 200 if total_steps > 12 else 250
        thumb_w = int(thumb_h * self.w / self.h)
        margin = 8
        label_h = 40

        canvas_w = cols * (thumb_w + margin) + margin
        canvas_h = rows * (thumb_h + label_h + margin) + margin + 60
        canvas = Image.new('RGB', (canvas_w, canvas_h), 'white')
        draw = ImageDraw.Draw(canvas)

        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 10)
            font_title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        except:
            font = font_title = ImageFont.load_default()

        # Title
        title = f"YOLO SMART PAINT - {len(self.layers)} Regions, {total_steps} Steps"
        draw.text((margin, margin), title, fill='darkblue', font=font_title)

        # Build images
        cumulative = np.ones((self.h, self.w, 3), dtype=np.uint8) * 255
        step_num = 0

        for layer in self.layers:
            for sub in layer.sub_layers:
                cumulative[sub.mask] = self.image[sub.mask]

                thumb = Image.fromarray(cumulative).resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)

                col = step_num % cols
                row = step_num // cols
                x = margin + col * (thumb_w + margin)
                y = margin + 60 + row * (thumb_h + label_h + margin)

                canvas.paste(thumb, (x, y))

                # Label
                short_name = sub.name[:28] if len(sub.name) > 28 else sub.name
                draw.text((x, y + thumb_h + 2), f"{step_num+1}. {short_name}", fill='black', font=font)

                step_num += 1

        overview_path = os.path.join(output_dir, "progress_overview.png")
        canvas.save(overview_path)
        return overview_path

    def create_layer_guide(self) -> List[Dict]:
        """Generate JSON painting guide."""
        guide = []

        # Prep step
        guide.append({
            "step": 0,
            "name": "Preparation",
            "type": "setup",
            "description": f"This painting has {len(self.layers)} semantic regions, painted back-to-front.",
            "regions": [l.name for l in self.layers],
        })

        step = 0
        for layer in self.layers:
            for sub in layer.sub_layers:
                step += 1
                guide.append({
                    "step": step,
                    "name": sub.name,
                    "parent": layer.name,
                    "category": layer.category,
                    "type": "paint_sub_layer",
                    "coverage": f"{sub.coverage*100:.1f}%",
                    "luminosity_range": [float(x) for x in sub.luminosity_range],
                    "dominant_color": [int(x) for x in sub.dominant_color],
                    "technique": sub.technique,
                    "tips": sub.tips,
                    "is_focal": layer.is_focal,
                })

        # Final step
        guide.append({
            "step": step + 1,
            "name": "Final Details",
            "type": "finishing",
            "tips": [
                "Step back and assess",
                "Add final highlights",
                "Sharpen edges on focal areas",
                "Sign your work!"
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
        guide_path = os.path.join(output_dir, "layer_guide.json")
        with open(guide_path, 'w') as f:
            json.dump(guide, f, indent=2)
        print("  layer_guide.json")

        # Scene analysis
        analysis = {
            "method": "yolo_smart",
            "num_regions": len(self.layers),
            "total_sub_layers": sum(len(l.sub_layers) for l in self.layers),
            "regions": [
                {
                    "name": l.name,
                    "category": l.category,
                    "coverage": round(l.coverage, 4),
                    "confidence": round(l.confidence, 2),
                    "is_focal": l.is_focal,
                    "sub_layers": len(l.sub_layers),
                }
                for l in self.layers
            ]
        }
        analysis_path = os.path.join(output_dir, "scene_analysis.json")
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        print("  scene_analysis.json")

        print("-" * 50)
        print(f"Done!")


def process_image(image_path: str, output_dir: str = None, model_size: str = "n") -> YOLOSmartPaintByNumbers:
    """Main entry point."""
    if output_dir is None:
        base = os.path.splitext(os.path.basename(image_path))[0]
        output_dir = f"output/{base}_yolo_smart"

    painter = YOLOSmartPaintByNumbers(image_path, model_size=model_size)
    painter.process()
    painter.save_all(output_dir)

    return painter


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python yolo_smart_paint.py <image_path> [output_dir]")
        sys.exit(0)

    image_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    process_image(image_path, output_dir)
