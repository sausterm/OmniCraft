"""
Smart Paint by Numbers - Combines traditional paint-by-numbers with intelligent technique analysis.

Generates:
1. Traditional paint-by-numbers templates (numbered regions)
2. Technique analysis (glazing, blending, layering detection)
3. Layer-by-layer painting instructions with ORGANIC layer boundaries
4. Visual technique guides

This creates a complete painting instruction kit that goes beyond simple "fill by number"
to teach actual painting techniques appropriate for each image.

Organic Segmentation:
- Instead of arbitrary horizontal cuts, layers follow natural image boundaries
- Groups regions by color similarity and luminosity
- Orders layers back-to-front for proper painting sequence
"""

import os
import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import json
from typing import List, Dict, Optional

# Updated imports for deprecated location
from artisan.core.paint_by_numbers import PaintByNumbers
from artisan.analysis.technique_analyzer import TechniqueAnalyzer, Technique
from artisan.analysis.technique_visualizer import TechniqueVisualizer
from artisan.analysis.organic_segmentation import segment_into_natural_layers


class SmartPaintByNumbers:
    """
    Enhanced paint-by-numbers that combines numbered templates with technique guidance
    and organic layer detection for natural painting order.
    """

    def __init__(self, image_path: str, n_colors: int = 15, min_region_size: int = 50,
                 use_organic_layers: bool = True, target_layers: int = 12):
        """
        Initialize smart paint-by-numbers.

        Args:
            image_path: Path to input image
            n_colors: Number of colors in palette
            min_region_size: Minimum pixels per region
            use_organic_layers: Whether to use organic layer detection
            target_layers: Target number of organic layers (8-15 recommended)
        """
        self.image_path = image_path
        self.n_colors = n_colors
        self.min_region_size = min_region_size
        self.use_organic_layers = use_organic_layers
        self.target_layers = target_layers

        # Initialize both systems
        self.pbn = PaintByNumbers(image_path, n_colors, min_region_size)
        self.analyzer = TechniqueAnalyzer(image_path)
        self.analysis = None
        self.visualizer = None

        # Organic layer detection results
        self.organic_layers: List[Dict] = []
        self.original_rgb = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

    def process(self):
        """Run complete analysis and template generation."""
        print("=" * 60)
        print("SMART PAINT BY NUMBERS")
        print("=" * 60)

        # Step 1: Analyze techniques
        print("\n[1/5] Analyzing painting techniques...")
        self.analysis = self.analyzer.full_analysis(self.n_colors)
        self._print_analysis_summary()

        # Step 2: Detect organic layers (natural boundaries)
        if self.use_organic_layers:
            print("\n[2/5] Detecting organic layers...")
            self.organic_layers = segment_into_natural_layers(
                self.original_rgb,
                target_layers=self.target_layers,
                min_coverage=0.005
            )
            print(f"  Found {len(self.organic_layers)} natural layers:")
            for layer in self.organic_layers:
                coverage_pct = layer['coverage'] * 100
                print(f"    - {layer['name']}: {coverage_pct:.1f}% coverage, technique: {layer['technique']}")
        else:
            print("\n[2/5] Skipping organic layer detection (disabled)")
            self.organic_layers = []

        # Step 3: Generate paint-by-numbers template
        print("\n[3/5] Generating paint-by-numbers template...")
        self.pbn.quantize_colors()
        self.pbn.create_regions()
        self.pbn.create_template()

        # Step 4: Initialize visualizer
        print("\n[4/5] Creating technique visualizations...")
        self.visualizer = TechniqueVisualizer(self.analyzer)

        # Step 5: Generate enhanced template with technique hints
        print("\n[5/5] Generating enhanced outputs...")

        return self.analysis

    def _print_analysis_summary(self):
        """Print analysis summary to console."""
        a = self.analysis
        print(f"  Pattern detected: {a.dominant_pattern.upper()}")
        print(f"  Difficulty: {'*' * a.overall_difficulty} ({a.overall_difficulty}/5)")
        print(f"  Dark background: {a.has_dark_background} ({a.dark_background_percentage:.0%})")
        print(f"  Glow effects: {a.has_glow_effects}")
        print(f"  Gradient intensity: {a.gradient_intensity:.2f}")
        print(f"  Edge softness: {a.edge_softness:.2f}")
        print(f"  Recommended layers: {a.layer_count}")

        if a.technique_zones:
            print(f"  Techniques to use:")
            for zone in a.technique_zones:
                print(f"    - {zone.technique.value.replace('_', ' ').title()}")

    def create_enhanced_template(self) -> np.ndarray:
        """
        Create an enhanced template that shows both numbers AND technique zones.

        Returns:
            Enhanced template image as numpy array
        """
        # Start with standard template
        template = self.pbn.template.copy()
        h, w = template.shape[:2]

        # Create technique zone overlay (very subtle)
        overlay = np.zeros((h, w, 3), dtype=np.float32)

        technique_colors = {
            Technique.STANDARD_FILL: (0.4, 0.4, 0.4),
            Technique.GLAZING: (1.0, 0.8, 0.0),
            Technique.WET_ON_WET: (0.0, 0.6, 1.0),
            Technique.LAYERED_BUILDUP: (0.6, 0.2, 1.0),
            Technique.GRADIENT_BLEND: (0.0, 1.0, 0.6),
        }

        for zone in self.analysis.technique_zones:
            if zone.technique in technique_colors:
                color = technique_colors[zone.technique]
                overlay[zone.region_mask] = color

        # Blend very subtly (just tint the white areas)
        white_mask = np.all(template > 240, axis=2)
        alpha = 0.15
        for c in range(3):
            template[:, :, c] = np.where(
                white_mask,
                template[:, :, c] * (1 - alpha) + overlay[:, :, c] * 255 * alpha,
                template[:, :, c]
            )

        return template.astype(np.uint8)

    def create_combo_output(self, output_dir: str) -> Image.Image:
        """
        Create a combined output showing template + technique info side by side.
        """
        # Get components
        template = Image.fromarray(self.pbn.template)
        colored = Image.fromarray(self.pbn.quantized_image)
        technique_map = self.visualizer.create_technique_map()

        # Calculate layout
        margin = 20
        thumb_height = 300

        # Scale images to same height
        aspect = template.width / template.height
        thumb_width = int(thumb_height * aspect)

        template_thumb = template.resize((thumb_width, thumb_height), Image.Resampling.LANCZOS)
        colored_thumb = colored.resize((thumb_width, thumb_height), Image.Resampling.LANCZOS)

        # Total width
        total_width = thumb_width * 2 + margin * 3
        total_height = thumb_height + 400  # Extra space for instructions

        combo = Image.new('RGB', (total_width, total_height), 'white')
        draw = ImageDraw.Draw(combo)

        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
            font_title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        except:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
                font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
            except:
                font = font_title = ImageFont.load_default()

        # Title
        title = f"SMART PAINT BY NUMBERS - {self.analysis.dominant_pattern.upper()}"
        draw.text((margin, margin // 2), title, fill='black', font=font_title)

        # Place images
        y = margin + 20
        combo.paste(template_thumb, (margin, y))
        combo.paste(colored_thumb, (margin * 2 + thumb_width, y))

        # Labels
        draw.text((margin, y + thumb_height + 5), "Template", fill='gray', font=font)
        draw.text((margin * 2 + thumb_width, y + thumb_height + 5), "Reference", fill='gray', font=font)

        # Instructions section
        y = thumb_height + margin + 40
        draw.text((margin, y), "PAINTING APPROACH:", fill='darkblue', font=font_title)
        y += 25

        # Add key insights
        insights = []
        if self.analysis.has_dark_background:
            insights.append("1. Start with a DARK BASE - paint entire canvas dark first")
        if self.analysis.has_glow_effects:
            insights.append("2. Use GLAZING - thin transparent layers for glow effects")
        if self.analysis.gradient_intensity > 0.15:
            insights.append("3. BLEND while wet - don't let paint dry between adjacent colors")

        if not insights:
            insights.append("1. Paint largest regions first")
            insights.append("2. Work from background to foreground")
            insights.append("3. Add details last")

        for insight in insights:
            draw.text((margin, y), insight, fill='black', font=font)
            y += 20

        # Technique summary
        y += 15
        draw.text((margin, y), "TECHNIQUES USED:", fill='darkblue', font=font_title)
        y += 25

        for zone in self.analysis.technique_zones:
            tech_name = zone.technique.value.replace('_', ' ').title()
            draw.text((margin, y), f"  - {tech_name}: {zone.instructions[:80]}...", fill='black', font=font)
            y += 18

        return combo

    def create_organic_layer_visualization(self) -> np.ndarray:
        """
        Create a visualization showing all organic layers with color-coded masks.

        Returns:
            Visualization image as numpy array (H, W, 3)
        """
        if not self.organic_layers:
            return self.original_rgb.copy()

        h, w = self.original_rgb.shape[:2]
        visualization = self.original_rgb.copy().astype(np.float32)

        # Color palette for layer overlays
        layer_colors = [
            (255, 100, 100),  # Red
            (100, 255, 100),  # Green
            (100, 100, 255),  # Blue
            (255, 255, 100),  # Yellow
            (255, 100, 255),  # Magenta
            (100, 255, 255),  # Cyan
            (255, 180, 100),  # Orange
            (180, 100, 255),  # Purple
        ]

        for i, layer in enumerate(self.organic_layers):
            color = layer_colors[i % len(layer_colors)]
            mask = layer['mask']

            # Apply semi-transparent color overlay
            alpha = 0.3
            for c in range(3):
                visualization[:, :, c] = np.where(
                    mask,
                    visualization[:, :, c] * (1 - alpha) + color[c] * alpha,
                    visualization[:, :, c]
                )

            # Draw mask boundary
            mask_uint8 = mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(visualization.astype(np.uint8), contours, -1, color, 2)

        return visualization.astype(np.uint8)

    def create_layer_by_layer_guide(self) -> List[Dict]:
        """
        Generate a detailed layer-by-layer painting guide based on organic layers.

        Returns:
            List of layer instruction dictionaries
        """
        instructions = []

        # Step 0: Preparation
        instructions.append({
            "step": 0,
            "name": "Preparation",
            "type": "setup",
            "description": f"Prepare your canvas and materials. This painting has {len(self.organic_layers)} organic layers.",
            "colors": [],
            "technique": "setup",
            "tips": [
                f"Pattern detected: {self.analysis.dominant_pattern.upper()}",
                f"Difficulty: {self.analysis.overall_difficulty}/5",
                "Have clean water and paper towels ready",
            ]
        })

        # Generate instructions for each organic layer
        for i, layer in enumerate(self.organic_layers):
            # Get dominant colors in this layer
            mask = layer['mask']
            layer_pixels = self.original_rgb[mask]

            if len(layer_pixels) > 0:
                # Find average color
                avg_color = np.mean(layer_pixels, axis=0).astype(int).tolist()
            else:
                avg_color = [128, 128, 128]

            # Determine technique based on layer properties
            technique = layer.get('technique', 'layer')
            technique_tips = self._get_technique_tips(technique, layer)

            instructions.append({
                "step": i + 1,
                "name": layer['name'],
                "type": "paint_layer",
                "description": f"Paint layer {i + 1}: {layer['name']}",
                "coverage": f"{layer['coverage'] * 100:.1f}%",
                "y_range": layer['y_range'],
                "avg_luminosity": layer['avg_luminosity'],
                "dominant_color": avg_color,
                "technique": technique,
                "tips": technique_tips,
            })

        # Final step: Details
        instructions.append({
            "step": len(self.organic_layers) + 1,
            "name": "Final Details",
            "type": "finishing",
            "description": "Add final highlights and details",
            "colors": [],
            "technique": "detail",
            "tips": [
                "Step back and assess the overall composition",
                "Add highlights to the brightest areas",
                "Deepen shadows where needed",
                "Sign your work!"
            ]
        })

        return instructions

    def _get_technique_tips(self, technique: str, layer: Dict) -> List[str]:
        """Generate technique-specific tips for a layer."""
        tips = []

        if technique == 'highlight':
            tips.append("Apply thin, translucent strokes")
            tips.append("Build up brightness gradually")
            tips.append("Use lighter pressure for softer highlights")
        elif technique == 'blend':
            tips.append("Work quickly while paint is wet")
            tips.append("Use a clean brush to soften edges")
            tips.append("Blend adjacent colors where they meet")
        elif technique == 'silhouette':
            tips.append("Use solid, opaque strokes")
            tips.append("Maintain clean edges")
            tips.append("This layer provides contrast and depth")
        else:  # 'layer' or default
            tips.append("Apply even coverage")
            tips.append("Follow the natural boundaries")
            tips.append("Let dry before adding details")

        # Add luminosity-based tips
        if layer['avg_luminosity'] > 0.7:
            tips.append("This is a bright area - use lighter paint mixtures")
        elif layer['avg_luminosity'] < 0.3:
            tips.append("This is a dark area - may need multiple coats for opacity")

        return tips

    def create_organic_step_images(self, output_dir: str, progress_only: bool = False) -> List[str]:
        """
        Create step-by-step images showing layers being painted in order.

        Generates:
        - progress_XX.png: Clean cumulative progress (what the painting looks like)
        - layer_XX.png: Just the current layer being added (isolated) - skipped if progress_only

        Args:
            output_dir: Directory to save step images
            progress_only: If True, only save progress images (skip isolated layer images)

        Returns:
            List of saved file paths
        """
        if not self.organic_layers:
            return []

        h, w = self.original_rgb.shape[:2]
        saved_paths = []

        # Start with white canvas
        cumulative = np.ones((h, w, 3), dtype=np.uint8) * 255

        for i, layer in enumerate(self.organic_layers):
            mask = layer['mask']

            # Paint this layer onto the cumulative image
            cumulative[mask] = self.original_rgb[mask]

            # Save clean progress image (cumulative, no outlines)
            progress_path = os.path.join(output_dir, f"progress_{i+1:02d}.png")
            cv2.imwrite(progress_path, cv2.cvtColor(cumulative, cv2.COLOR_RGB2BGR))
            saved_paths.append(progress_path)

            if not progress_only:
                # Save isolated layer image (just this layer on white)
                layer_img = np.ones((h, w, 3), dtype=np.uint8) * 255
                layer_img[mask] = self.original_rgb[mask]
                layer_path = os.path.join(output_dir, f"layer_{i+1:02d}_{layer['name']}.png")
                cv2.imwrite(layer_path, cv2.cvtColor(layer_img, cv2.COLOR_RGB2BGR))
                saved_paths.append(layer_path)

        # Save final complete image
        final_path = os.path.join(output_dir, "final_complete.png")
        cv2.imwrite(final_path, cv2.cvtColor(cumulative, cv2.COLOR_RGB2BGR))
        saved_paths.append(final_path)

        return saved_paths

    def create_progress_overview(self, output_dir: str) -> str:
        """
        Create a single image showing all progress steps in a grid.

        Returns:
            Path to the saved overview image
        """
        if not self.organic_layers:
            return None

        from PIL import Image, ImageDraw, ImageFont

        n_layers = len(self.organic_layers)
        h, w = self.original_rgb.shape[:2]

        # Calculate grid layout (aim for roughly square)
        cols = min(4, n_layers + 1)  # +1 for final
        rows = (n_layers + 1 + cols - 1) // cols

        # Thumbnail size
        thumb_h = 250
        thumb_w = int(thumb_h * w / h)
        margin = 10
        label_h = 25

        # Create canvas
        canvas_w = cols * (thumb_w + margin) + margin
        canvas_h = rows * (thumb_h + label_h + margin) + margin
        canvas = Image.new('RGB', (canvas_w, canvas_h), 'white')
        draw = ImageDraw.Draw(canvas)

        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        except:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
            except:
                font = ImageFont.load_default()

        # Generate thumbnails
        cumulative = np.ones((h, w, 3), dtype=np.uint8) * 255

        for i, layer in enumerate(self.organic_layers):
            mask = layer['mask']
            cumulative[mask] = self.original_rgb[mask]

            # Create thumbnail
            thumb = Image.fromarray(cumulative).resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)

            # Calculate position
            col = i % cols
            row = i // cols
            x = margin + col * (thumb_w + margin)
            y = margin + row * (thumb_h + label_h + margin)

            # Paste thumbnail
            canvas.paste(thumb, (x, y))

            # Add label
            label = f"Step {i+1}: {layer['name']}"
            draw.text((x, y + thumb_h + 2), label[:30], fill='black', font=font)

        # Add final complete image
        final_idx = n_layers
        col = final_idx % cols
        row = final_idx // cols
        x = margin + col * (thumb_w + margin)
        y = margin + row * (thumb_h + label_h + margin)

        final_thumb = Image.fromarray(cumulative).resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        canvas.paste(final_thumb, (x, y))
        draw.text((x, y + thumb_h + 2), "COMPLETE", fill='darkgreen', font=font)

        # Save
        overview_path = os.path.join(output_dir, "progress_overview.png")
        canvas.save(overview_path)

        return overview_path

    def save_all(self, output_dir: str, minimal: bool = False):
        """
        Save all outputs to a directory.

        Args:
            output_dir: Directory for outputs
            minimal: If True, only save essential files (progress images, guide, reference)
        """
        os.makedirs(output_dir, exist_ok=True)

        print(f"\nSaving outputs to {output_dir}/")
        print("-" * 50)

        # Always save colored reference
        colored_path = os.path.join(output_dir, "colored_reference.png")
        cv2.imwrite(colored_path, cv2.cvtColor(self.pbn.quantized_image, cv2.COLOR_RGB2BGR))
        print(f"  colored_reference.png - Completed colored version")

        if not minimal:
            # 1. Standard template
            template_path = os.path.join(output_dir, "template.png")
            cv2.imwrite(template_path, cv2.cvtColor(self.pbn.template, cv2.COLOR_RGB2BGR))
            print(f"  template.png - Numbered paint-by-numbers template")

            # 2. Enhanced template (with technique tints)
            enhanced = self.create_enhanced_template()
            enhanced_path = os.path.join(output_dir, "template_enhanced.png")
            cv2.imwrite(enhanced_path, cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR))
            print(f"  template_enhanced.png - Template with technique zone hints")

            # 4. Technique visualizations
            self.visualizer.generate_all_outputs(output_dir)

            # 5. Combo output
            combo = self.create_combo_output(output_dir)
            combo.save(os.path.join(output_dir, "combo_guide.png"))
            print(f"  combo_guide.png - Combined template + instructions")

        # 6. Organic layer visualization and step-by-step guide
        if self.organic_layers:
            if not minimal:
                # Organic layer overview
                organic_viz = self.create_organic_layer_visualization()
                organic_viz_path = os.path.join(output_dir, "organic_layers.png")
                cv2.imwrite(organic_viz_path, cv2.cvtColor(organic_viz, cv2.COLOR_RGB2BGR))
                print(f"  organic_layers.png - Organic layer visualization")

            # Step-by-step layer images
            steps_dir = os.path.join(output_dir, "steps")
            os.makedirs(steps_dir, exist_ok=True)
            step_paths = self.create_organic_step_images(steps_dir, progress_only=minimal)
            n_progress = len([p for p in step_paths if 'progress_' in p])
            if minimal:
                print(f"  steps/ - {n_progress} progress images")
            else:
                n_layers = len([p for p in step_paths if 'layer_' in p])
                print(f"  steps/ - {n_progress} progress images + {n_layers} layer images")

            # Progress overview grid
            overview_path = self.create_progress_overview(output_dir)
            print(f"  progress_overview.png - Grid showing all steps")

            # Organic layer guide JSON
            organic_guide = self.create_layer_by_layer_guide()
            organic_guide_path = os.path.join(output_dir, "organic_layer_guide.json")
            with open(organic_guide_path, 'w') as f:
                json.dump(organic_guide, f, indent=2, default=str)
            print(f"  organic_layer_guide.json - Layer-by-layer painting guide")

        if not minimal:
            # 7. Color guide JSON (enhanced with techniques and organic layers)
            matched = self.pbn.match_to_paint_colors()
            enhanced_guide = {
                "n_colors": self.n_colors,
                "pattern": self.analysis.dominant_pattern,
                "difficulty": self.analysis.overall_difficulty,
                "has_dark_background": self.analysis.has_dark_background,
                "has_glow_effects": self.analysis.has_glow_effects,
                "recommended_base_color": self.analysis.recommended_base,
                "organic_layers": [
                    {
                        "name": layer['name'],
                        "priority": layer['priority'],
                        "coverage": layer['coverage'],
                        "technique": layer['technique'],
                        "avg_luminosity": layer['avg_luminosity'],
                        "y_range": layer['y_range'],
                    }
                    for layer in self.organic_layers
                ] if self.organic_layers else [],
                "techniques": [
                    {
                        "name": z.technique.value,
                        "layer_order": z.layer_order,
                        "instructions": z.instructions,
                        "brush_direction": z.brush_direction
                    }
                    for z in self.analysis.technique_zones
                ],
                "colors": matched,
                "layer_instructions": self.analyzer.generate_layer_instructions(self.analysis)
            }

            guide_path = os.path.join(output_dir, "painting_guide.json")
            with open(guide_path, 'w') as f:
                json.dump(enhanced_guide, f, indent=2, default=str)
            print(f"  painting_guide.json - Complete painting guide data")

        print("-" * 50)
        print(f"Done! {len(os.listdir(output_dir))} files generated.")

        return output_dir


def process_image(image_path: str, n_colors: int = 15, output_dir: str = None):
    """
    Main entry point for smart paint-by-numbers processing.

    Args:
        image_path: Path to input image
        n_colors: Number of colors
        output_dir: Output directory (default: based on image name)

    Returns:
        SmartPaintByNumbers instance
    """
    if output_dir is None:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_dir = f"output_{base_name}_n{n_colors}"

    spbn = SmartPaintByNumbers(image_path, n_colors)
    spbn.process()
    spbn.save_all(output_dir)

    return spbn


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Smart Paint by Numbers")
        print("=" * 40)
        print("Usage: python smart_paint_by_numbers.py <image_path> [n_colors] [output_dir]")
        print()
        print("Examples:")
        print("  python smart_paint_by_numbers.py aurora.png")
        print("  python smart_paint_by_numbers.py aurora.png 15")
        print("  python smart_paint_by_numbers.py aurora.png 15 my_output")
        sys.exit(0)

    image_path = sys.argv[1]
    n_colors = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    output_dir = sys.argv[3] if len(sys.argv) > 3 else None

    process_image(image_path, n_colors, output_dir)
