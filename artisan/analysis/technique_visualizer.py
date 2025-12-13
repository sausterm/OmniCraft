"""
Technique Visualizer - Generate visual instruction guides for painting techniques.

Creates:
- Technique zone overlay maps
- Layer-by-layer visual guides
- Brush direction indicators
- Printable instruction sheets
"""

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
import os

from .technique_analyzer import TechniqueAnalyzer, Technique, ImageAnalysis


class TechniqueVisualizer:
    """Generates visual instruction guides for painting techniques."""

    # Color coding for different techniques
    TECHNIQUE_COLORS = {
        Technique.STANDARD_FILL: (100, 100, 100),      # Gray
        Technique.GLAZING: (255, 200, 0),               # Gold
        Technique.WET_ON_WET: (0, 150, 255),            # Blue
        Technique.DRY_BRUSH: (255, 100, 100),           # Red
        Technique.LAYERED_BUILDUP: (150, 50, 255),      # Purple
        Technique.GRADIENT_BLEND: (0, 255, 150),        # Teal
    }

    TECHNIQUE_LABELS = {
        Technique.STANDARD_FILL: "Fill",
        Technique.GLAZING: "Glaze",
        Technique.WET_ON_WET: "Wet Blend",
        Technique.DRY_BRUSH: "Dry Brush",
        Technique.LAYERED_BUILDUP: "Layer",
        Technique.GRADIENT_BLEND: "Blend",
    }

    def __init__(self, analyzer: TechniqueAnalyzer):
        """
        Initialize visualizer with an analyzer.

        Args:
            analyzer: TechniqueAnalyzer instance with completed analysis
        """
        self.analyzer = analyzer
        self.analysis = analyzer.full_analysis()
        self.width = analyzer.width
        self.height = analyzer.height

    def create_technique_overlay(self, opacity: float = 0.4) -> np.ndarray:
        """
        Create a technique zone overlay using OUTLINES instead of color fills.
        This preserves the original image colors while showing technique boundaries.

        Args:
            opacity: Overlay opacity (0-1)

        Returns:
            RGBA overlay image
        """
        # Start with transparent
        overlay = np.zeros((self.height, self.width, 4), dtype=np.uint8)

        for zone in self.analysis.technique_zones:
            color = self.TECHNIQUE_COLORS[zone.technique]
            mask = zone.region_mask.astype(np.uint8)

            # Create outline instead of fill - find boundary of the mask
            kernel = np.ones((5, 5), np.uint8)
            dilated = cv2.dilate(mask, kernel, iterations=2)
            eroded = cv2.erode(mask, kernel, iterations=1)
            boundary = dilated - eroded

            # Apply color only to boundary (outline)
            overlay[boundary > 0, 0] = color[0]
            overlay[boundary > 0, 1] = color[1]
            overlay[boundary > 0, 2] = color[2]
            overlay[boundary > 0, 3] = int(255 * 0.8)  # Solid outline

        return overlay

    def create_technique_map(self, save_path: str = None) -> Image.Image:
        """
        Create a complete technique map with legend.
        Shows original image with subtle technique zone outlines.

        Args:
            save_path: Optional path to save the image

        Returns:
            PIL Image of the technique map
        """
        # Show original with only outline overlay (preserves original colors)
        original = Image.fromarray(self.analyzer.original_rgb)
        overlay = Image.fromarray(self.create_technique_overlay(0.8))

        # Blend - outlines only, original colors preserved
        composite = Image.alpha_composite(original.convert('RGBA'), overlay)

        # Add legend
        legend_height = 120
        final_width = self.width
        final_height = self.height + legend_height

        final = Image.new('RGB', (final_width, final_height), 'white')
        final.paste(composite.convert('RGB'), (0, 0))

        draw = ImageDraw.Draw(final)

        # Load font
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
            font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 11)
            font_title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        except:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
                font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
                font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
            except:
                font = font_small = font_title = ImageFont.load_default()

        # Draw title
        title = f"TECHNIQUE MAP - {self.analysis.dominant_pattern.upper()}"
        draw.text((10, self.height + 5), title, fill='black', font=font_title)

        # Draw difficulty stars
        difficulty = f"Difficulty: {'*' * self.analysis.overall_difficulty}"
        draw.text((self.width - 150, self.height + 5), difficulty, fill='black', font=font)

        # Draw legend swatches
        legend_y = self.height + 35
        x_pos = 10
        techniques_in_image = set(z.technique for z in self.analysis.technique_zones)

        for technique in techniques_in_image:
            color = self.TECHNIQUE_COLORS[technique]
            label = self.TECHNIQUE_LABELS[technique]

            # Draw swatch
            draw.rectangle([x_pos, legend_y, x_pos + 25, legend_y + 20], fill=color, outline='black')

            # Draw label
            draw.text((x_pos + 30, legend_y + 2), label, fill='black', font=font_small)

            x_pos += 100

        # Draw layer info
        layer_text = f"Layers: {self.analysis.layer_count} | "
        layer_text += "Dark BG " if self.analysis.has_dark_background else ""
        layer_text += "Glow " if self.analysis.has_glow_effects else ""
        draw.text((10, legend_y + 30), layer_text, fill='gray', font=font_small)

        # Draw pattern-specific tip
        tips = {
            "aurora": "TIP: Use glazing - thin transparent layers over black",
            "sunset": "TIP: Work wet-on-wet for smooth color transitions",
            "fire": "TIP: Build from dark to bright, add white last",
            "standard": "TIP: Paint largest areas first, details last",
        }
        tip = tips.get(self.analysis.dominant_pattern, "")
        draw.text((10, legend_y + 50), tip, fill='darkblue', font=font_small)

        if save_path:
            final.save(save_path)
            print(f"Technique map saved to {save_path}")

        return final

    def create_brush_direction_map(self, save_path: str = None) -> Image.Image:
        """
        Create a map showing recommended brush directions.
        """
        # Start with dimmed original
        original = self.analyzer.original_rgb.copy().astype(np.float32)
        dimmed = (original * 0.4).astype(np.uint8)
        img = Image.fromarray(dimmed)
        draw = ImageDraw.Draw(img)

        # Draw arrows for each zone with direction
        for zone in self.analysis.technique_zones:
            if zone.brush_direction is None:
                continue

            # Find region centroid
            mask = zone.region_mask
            y_coords, x_coords = np.where(mask)
            if len(y_coords) == 0:
                continue

            center_x = int(np.mean(x_coords))
            center_y = int(np.mean(y_coords))

            # Draw direction arrows
            arrow_len = min(self.width, self.height) // 8
            color = self.TECHNIQUE_COLORS[zone.technique]

            if zone.brush_direction == "vertical" or zone.brush_direction == "vertical_down":
                self._draw_arrow(draw, center_x, center_y - arrow_len//2,
                               center_x, center_y + arrow_len//2, color)
            elif zone.brush_direction == "vertical_up":
                self._draw_arrow(draw, center_x, center_y + arrow_len//2,
                               center_x, center_y - arrow_len//2, color)
            elif zone.brush_direction == "horizontal":
                self._draw_arrow(draw, center_x - arrow_len//2, center_y,
                               center_x + arrow_len//2, center_y, color)
            elif zone.brush_direction == "radial":
                # Draw multiple arrows radiating out
                for angle in [0, 45, 90, 135]:
                    rad = np.radians(angle)
                    dx = int(arrow_len//2 * np.cos(rad))
                    dy = int(arrow_len//2 * np.sin(rad))
                    self._draw_arrow(draw, center_x, center_y,
                                   center_x + dx, center_y + dy, color)

        if save_path:
            img.save(save_path)
            print(f"Brush direction map saved to {save_path}")

        return img

    def _draw_arrow(self, draw, x1, y1, x2, y2, color, width=3):
        """Draw an arrow on the image."""
        draw.line([(x1, y1), (x2, y2)], fill=color, width=width)

        # Draw arrowhead
        angle = np.arctan2(y2 - y1, x2 - x1)
        arrow_size = 10

        x3 = x2 - arrow_size * np.cos(angle - np.pi/6)
        y3 = y2 - arrow_size * np.sin(angle - np.pi/6)
        x4 = x2 - arrow_size * np.cos(angle + np.pi/6)
        y4 = y2 - arrow_size * np.sin(angle + np.pi/6)

        draw.polygon([(x2, y2), (x3, y3), (x4, y4)], fill=color)

    def create_layer_guide(self, save_path: str = None) -> Image.Image:
        """
        Create a visual layer-by-layer painting guide.
        """
        instructions = self.analyzer.generate_layer_instructions(self.analysis)

        # Calculate layout
        thumb_size = 200
        margin = 20
        text_width = 400

        n_layers = len(instructions)
        cols = 2
        rows = (n_layers + cols - 1) // cols

        img_width = (thumb_size + text_width + margin * 3) * cols
        img_height = margin + rows * (thumb_size + margin * 2)

        img = Image.new('RGB', (img_width, max(800, img_height)), 'white')
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
            font_title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        except:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
                font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
            except:
                font = font_title = ImageFont.load_default()

        # Title
        draw.text((margin, margin//2), f"LAYER-BY-LAYER GUIDE: {self.analysis.dominant_pattern.upper()}",
                 fill='black', font=font_title)

        col_width = (thumb_size + text_width + margin * 3)

        for i, inst in enumerate(instructions):
            col = i % cols
            row = i // cols

            x = margin + col * col_width
            y = margin * 2 + row * (thumb_size + margin * 2)

            # Draw layer thumbnail
            thumb = self._create_layer_thumbnail(inst, thumb_size)
            img.paste(thumb, (x, y))

            # Draw text
            text_x = x + thumb_size + margin

            # Layer name
            draw.text((text_x, y), inst['name'], fill='darkblue', font=font_title)

            # Technique
            tech_text = f"Technique: {inst['technique'].replace('_', ' ').title()}"
            draw.text((text_x, y + 20), tech_text, fill='gray', font=font)

            # Instructions (wrapped)
            instr_text = inst['instructions'][:300]
            if len(inst['instructions']) > 300:
                instr_text += "..."

            # Simple word wrap
            words = instr_text.split()
            lines = []
            current_line = ""
            for word in words:
                if len(current_line + word) < 45:
                    current_line += word + " "
                else:
                    lines.append(current_line)
                    current_line = word + " "
            lines.append(current_line)

            for j, line in enumerate(lines[:8]):  # Max 8 lines
                draw.text((text_x, y + 40 + j * 16), line.strip(), fill='black', font=font)

            # Colors
            if inst['colors']:
                color_y = y + thumb_size - 25
                draw.text((text_x, color_y), "Colors:", fill='gray', font=font)
                for k, color in enumerate(inst['colors'][:5]):
                    cx = text_x + 50 + k * 25
                    if isinstance(color, tuple) and len(color) == 3:
                        draw.rectangle([cx, color_y, cx + 20, color_y + 20],
                                     fill=color, outline='black')

        if save_path:
            img.save(save_path)
            print(f"Layer guide saved to {save_path}")

        return img

    def _create_layer_thumbnail(self, instruction: dict, size: int) -> Image.Image:
        """Create a thumbnail representing a layer."""
        img = Image.new('RGB', (size, size), 'white')
        draw = ImageDraw.Draw(img)

        technique = instruction.get('technique', 'setup')
        colors = instruction.get('colors', [])

        if technique == 'setup':
            # Draw setup icon
            draw.rectangle([10, 10, size-10, size-10], outline='gray', width=2)
            draw.text((size//2 - 30, size//2 - 10), "SETUP", fill='gray')
        elif technique == 'finishing':
            # Draw checkmark
            draw.rectangle([10, 10, size-10, size-10], outline='green', width=2)
            draw.text((size//2 - 30, size//2 - 10), "DONE", fill='green')
        elif technique == 'standard_fill':
            # Fill with base color
            if colors:
                draw.rectangle([10, 10, size-10, size-10], fill=colors[0], outline='black')
            else:
                draw.rectangle([10, 10, size-10, size-10], fill='darkgray', outline='black')
        elif technique == 'glazing':
            # Show layered effect
            if colors:
                draw.rectangle([10, 10, size-10, size-10], fill=(30, 30, 30))
                for i, color in enumerate(colors):
                    alpha_offset = i * 20
                    draw.ellipse([20 + alpha_offset, 20 + alpha_offset,
                                size - 20 - alpha_offset, size - 20 - alpha_offset],
                               fill=color, outline=None)
        elif technique in ['wet_on_wet', 'gradient_blend']:
            # Show gradient
            if len(colors) >= 2:
                for y in range(size):
                    ratio = y / size
                    r = int(colors[0][0] * (1-ratio) + colors[-1][0] * ratio)
                    g = int(colors[0][1] * (1-ratio) + colors[-1][1] * ratio)
                    b = int(colors[0][2] * (1-ratio) + colors[-1][2] * ratio)
                    draw.line([(10, y), (size-10, y)], fill=(r, g, b))
            draw.rectangle([10, 10, size-10, size-10], outline='black', width=1)
        elif technique == 'layered_buildup':
            # Show concentric layers
            if colors:
                draw.rectangle([10, 10, size-10, size-10], fill=colors[0] if colors else 'black')
                for i, color in enumerate(colors[1:], 1):
                    inset = i * 15
                    draw.ellipse([10 + inset, 10 + inset, size - 10 - inset, size - 10 - inset],
                               fill=color)

        return img

    def create_printable_guide(self, save_path: str = None) -> Image.Image:
        """
        Create a complete printable instruction sheet.
        """
        instructions = self.analyzer.generate_layer_instructions(self.analysis)

        # A4-ish proportions at screen res
        page_width = 800
        page_height = 1100

        img = Image.new('RGB', (page_width, page_height), 'white')
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 11)
            font_bold = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
            font_title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
        except:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
                font_bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
                font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
            except:
                font = font_bold = font_title = ImageFont.load_default()

        margin = 40
        y = margin

        # Header
        draw.text((margin, y), "PAINTING TECHNIQUE GUIDE", fill='black', font=font_title)
        y += 30

        # Pattern info
        pattern_info = f"Pattern: {self.analysis.dominant_pattern.upper()}  |  "
        pattern_info += f"Difficulty: {'*' * self.analysis.overall_difficulty}  |  "
        pattern_info += f"Layers: {self.analysis.layer_count}"
        draw.text((margin, y), pattern_info, fill='gray', font=font)
        y += 25

        # Horizontal line
        draw.line([(margin, y), (page_width - margin, y)], fill='lightgray', width=1)
        y += 15

        # Reference image (small)
        ref_size = 150
        ref_img = Image.fromarray(self.analyzer.original_rgb)
        ref_img.thumbnail((ref_size, ref_size))
        img.paste(ref_img, (page_width - margin - ref_size, y))

        # Analysis summary
        draw.text((margin, y), "ANALYSIS:", fill='darkblue', font=font_bold)
        y += 18
        if self.analysis.has_dark_background:
            draw.text((margin, y), "* Dark background detected - use GLAZING technique", fill='black', font=font)
            y += 16
        if self.analysis.has_glow_effects:
            draw.text((margin, y), f"* Glow effects detected - build brightness in layers", fill='black', font=font)
            y += 16
        if self.analysis.gradient_intensity > 0.15:
            draw.text((margin, y), f"* Soft gradients - blend colors while wet", fill='black', font=font)
            y += 16

        y += 20

        # Layer instructions
        draw.text((margin, y), "LAYER-BY-LAYER INSTRUCTIONS:", fill='darkblue', font=font_bold)
        y += 25

        for inst in instructions:
            if y > page_height - 100:
                break  # Don't overflow page

            # Layer header
            draw.text((margin, y), inst['name'], fill='darkgreen', font=font_bold)
            y += 18

            # Instructions (wrapped)
            instr = inst['instructions'].replace('\n', ' ').strip()
            words = instr.split()
            lines = []
            current = ""
            for word in words:
                if len(current + word) < 90:
                    current += word + " "
                else:
                    lines.append(current)
                    current = word + " "
            if current:
                lines.append(current)

            for line in lines[:6]:  # Max 6 lines per instruction
                draw.text((margin + 15, y), line.strip(), fill='black', font=font)
                y += 14

            y += 10

        if save_path:
            img.save(save_path)
            print(f"Printable guide saved to {save_path}")

        return img

    def generate_all_outputs(self, output_dir: str):
        """
        Generate all visual outputs to a directory.

        Args:
            output_dir: Directory to save outputs
        """
        os.makedirs(output_dir, exist_ok=True)

        print(f"\nGenerating technique visualizations to {output_dir}/")
        print("-" * 50)

        # Technique map
        self.create_technique_map(os.path.join(output_dir, "technique_map.png"))

        # Brush direction map
        self.create_brush_direction_map(os.path.join(output_dir, "brush_directions.png"))

        # Layer guide
        self.create_layer_guide(os.path.join(output_dir, "layer_guide.png"))

        # Printable guide
        self.create_printable_guide(os.path.join(output_dir, "printable_guide.png"))

        # Export JSON analysis
        self.analyzer.export_analysis(os.path.join(output_dir, "technique_analysis.json"))

        print("-" * 50)
        print("Generated files:")
        print("  technique_map.png      - Color-coded technique zones")
        print("  brush_directions.png   - Recommended brush stroke directions")
        print("  layer_guide.png        - Visual layer-by-layer guide")
        print("  printable_guide.png    - Print-ready instruction sheet")
        print("  technique_analysis.json - Complete analysis data")


def analyze_and_visualize(image_path: str, output_dir: str, n_colors: int = 15):
    """
    Complete pipeline: analyze image and generate all visualizations.

    Args:
        image_path: Path to input image
        output_dir: Output directory
        n_colors: Number of colors for palette
    """
    print("=" * 60)
    print("TECHNIQUE ANALYSIS & VISUALIZATION")
    print("=" * 60)
    print(f"Image: {image_path}")
    print(f"Output: {output_dir}/")

    # Analyze
    analyzer = TechniqueAnalyzer(image_path)
    analysis = analyzer.full_analysis(n_colors)

    # Print summary
    print(f"\nPattern: {analysis.dominant_pattern.upper()}")
    print(f"Difficulty: {'*' * analysis.overall_difficulty} ({analysis.overall_difficulty}/5)")
    print(f"Dark background: {analysis.has_dark_background}")
    print(f"Glow effects: {analysis.has_glow_effects}")
    print(f"Layers needed: {analysis.layer_count}")

    # Visualize
    visualizer = TechniqueVisualizer(analyzer)
    visualizer.generate_all_outputs(output_dir)

    return analyzer, visualizer


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python technique_visualizer.py <image_path> [output_dir]")
        sys.exit(1)

    image_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "technique_output"

    analyze_and_visualize(image_path, output_dir)
