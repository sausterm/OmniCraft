"""
Generate paint-by-numbers templates for multiple N color values.
Outputs both blank numbered templates and colored versions.
"""

import os
import sys
import cv2
import numpy as np
from PIL import Image

# Add grandparent directory to path so artisan is a proper package
_artisan_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_project_dir = os.path.dirname(_artisan_dir)
sys.path.insert(0, _project_dir)

from artisan.core.paint_by_numbers import PaintByNumbers


def generate_for_n_values(image_path, n_values, output_dir='output', min_region_size=50):
    """
    Generate paint-by-numbers templates for multiple N color values.

    Args:
        image_path: Path to input image
        n_values: List of N values (e.g., [5, 10, 15, 20])
        output_dir: Output directory
        min_region_size: Minimum region size in pixels
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"Processing: {image_path}")
    print(f"N values: {n_values}")
    print(f"Output: {output_dir}/")
    print("=" * 60)

    for n in n_values:
        print(f"\n>>> Generating N={n} colors...")

        # Create subdirectory for this N value
        n_dir = os.path.join(output_dir, f"n{n}")
        os.makedirs(n_dir, exist_ok=True)

        # Process image
        pbn = PaintByNumbers(image_path, n_colors=n, min_region_size=min_region_size)
        pbn.quantize_colors()
        pbn.create_regions()
        pbn.create_template()

        # Save blank template with numbers
        template_path = os.path.join(n_dir, f"template_n{n}.png")
        cv2.imwrite(template_path, cv2.cvtColor(pbn.template, cv2.COLOR_RGB2BGR))
        print(f"    Saved: {template_path}")

        # Save colored version (quantized image)
        colored_path = os.path.join(n_dir, f"colored_n{n}.png")
        cv2.imwrite(colored_path, cv2.cvtColor(pbn.quantized_image, cv2.COLOR_RGB2BGR))
        print(f"    Saved: {colored_path}")

        # Save palette swatch
        palette_path = os.path.join(n_dir, f"palette_n{n}.png")
        save_palette(pbn.palette, n, palette_path)
        print(f"    Saved: {palette_path}")

        # Save color guide
        matched = pbn.match_to_paint_colors()
        pbn.save_color_guide(matched, os.path.join(n_dir, f"color_guide_n{n}.json"))

    print("\n" + "=" * 60)
    print("Done! Generated files for each N value:")
    for n in n_values:
        print(f"  {output_dir}/n{n}/")
        print(f"    - template_n{n}.png     (blank with numbers)")
        print(f"    - colored_n{n}.png      (completed colored)")
        print(f"    - palette_n{n}.png      (color palette)")
        print(f"    - color_guide_n{n}.json (color data)")


def save_palette(palette, n, save_path):
    """Save a visual palette swatch image."""
    swatch_height = 60
    swatch_width = 60
    padding = 2

    cols = min(n, 10)
    rows = (n + cols - 1) // cols

    img_width = cols * (swatch_width + padding) + padding
    img_height = rows * (swatch_height + padding + 20) + padding

    img = Image.new('RGB', (img_width, img_height), 'white')
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except:
            font = ImageFont.load_default()

    for i, color in enumerate(palette):
        row = i // cols
        col = i % cols

        x = padding + col * (swatch_width + padding)
        y = padding + row * (swatch_height + padding + 20)

        # Draw color swatch
        draw.rectangle([x, y, x + swatch_width, y + swatch_height],
                      fill=tuple(color))
        draw.rectangle([x, y, x + swatch_width, y + swatch_height],
                      outline='black', width=1)

        # Draw number below
        text = str(i + 1)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_x = x + (swatch_width - text_width) // 2
        text_y = y + swatch_height + 2
        draw.text((text_x, text_y), text, fill='black', font=font)

    img.save(save_path)


def create_comparison_grid(image_path, n_values, output_path, min_region_size=50):
    """
    Create a side-by-side comparison grid showing all N values.
    """
    import matplotlib.pyplot as plt

    n_cols = len(n_values)
    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 8))

    if n_cols == 1:
        axes = axes.reshape(2, 1)

    for i, n in enumerate(n_values):
        pbn = PaintByNumbers(image_path, n_colors=n, min_region_size=min_region_size)
        pbn.quantize_colors()
        pbn.create_regions()
        pbn.create_template()

        # Top row: colored version
        axes[0, i].imshow(pbn.quantized_image)
        axes[0, i].set_title(f'N={n} Colors', fontsize=12, fontweight='bold')
        axes[0, i].axis('off')

        # Bottom row: template
        axes[1, i].imshow(pbn.template)
        axes[1, i].set_title(f'Template (N={n})', fontsize=10)
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison grid: {output_path}")


if __name__ == "__main__":
    import sys

    # Default image and N values
    image_path = "aurora_maria.png"
    n_values = [5, 10, 15, 20, 25]
    output_dir = "output"

    # Parse command line args
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    if len(sys.argv) > 2:
        n_values = [int(x) for x in sys.argv[2].split(',')]
    if len(sys.argv) > 3:
        output_dir = sys.argv[3]

    print("=" * 60)
    print("Paint by Numbers - Multi-N Generator")
    print("=" * 60)

    # Generate individual files
    generate_for_n_values(image_path, n_values, output_dir)

    # Generate comparison grid
    comparison_path = os.path.join(output_dir, "comparison_grid.png")
    create_comparison_grid(image_path, n_values, comparison_path)

    print("\nUsage:")
    print(f"  python generate_multi_n.py <image> <n_values> <output_dir>")
    print(f"  python generate_multi_n.py photo.jpg 5,10,15,20 my_output")
