#!/usr/bin/env python3
"""
Generate comprehensive PDF instruction booklets from painting step images and metadata.
Includes detailed painting techniques, tool guidance, and tutorial instructions.

Usage:
    python generate_instruction_booklets.py                    # Generate for all outputs
    python generate_instruction_booklets.py dogs fireweed      # Generate for specific outputs
"""

import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
import textwrap


def rgb_to_hex(rgb):
    """Convert RGB tuple to hex color string."""
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def wrap_text(text, width=80):
    """Wrap text to specified width."""
    return "\n".join(textwrap.wrap(text, width=width))


def get_technique_details(technique, brush, motion):
    """Generate detailed technique instructions based on technique type."""
    techniques = {
        "blocking": {
            "description": "Blocking establishes your foundational values. Work quickly and confidently.",
            "prep": "Load your brush generously with thinned paint (consistency of heavy cream).",
            "execution": [
                "Start from the largest shapes first",
                "Don't worry about edges - we'll refine later",
                "Keep your strokes following the natural form",
                "Cover the canvas section completely before moving on"
            ],
            "common_errors": "Avoid overworking - first impressions are often best."
        },
        "layering": {
            "description": "Building up transparent or semi-transparent layers creates depth and richness.",
            "prep": "Use medium-consistency paint. Each layer should be slightly thinner than the last.",
            "execution": [
                "Wait for previous layer to be touch-dry (tacky but not wet)",
                "Apply paint with a light touch to avoid lifting underlying layers",
                "Build color intensity gradually",
                "Work in the same direction as underlying strokes"
            ],
            "common_errors": "Don't rush drying time - patience creates luminosity."
        },
        "highlighting": {
            "description": "Highlights bring your painting to life. Less is always more.",
            "prep": "Use your brightest, most opaque paint. Very little on the brush.",
            "execution": [
                "Identify where light would naturally hit",
                "Touch lightly - don't drag",
                "Build up gradually rather than one heavy application",
                "Step back frequently to assess"
            ],
            "common_errors": "Too many highlights flatten the image. Be selective."
        },
        "detailing": {
            "description": "Details draw the eye and create focal points. Save for last.",
            "prep": "Use a script liner or small round brush. Thin paint to ink consistency.",
            "execution": [
                "Steady your hand by bracing against the canvas edge",
                "Pull strokes toward you when possible",
                "Work from dark to light",
                "Add texture sparingly"
            ],
            "common_errors": "Over-detailing everywhere loses impact. Focus on key areas."
        },
        "shadow": {
            "description": "Shadows create depth and form. They're the foundation of realism.",
            "prep": "Mix a darker value of your local color. Shadows have color, never pure black.",
            "execution": [
                "Observe where light is blocked",
                "Shadows are softer further from the object",
                "Use cool colors for distant shadows, warmer for close",
                "Blend edges where appropriate"
            ],
            "common_errors": "Shadows too dark or too uniform look flat. Vary your values."
        },
        "blending": {
            "description": "Blending creates smooth transitions between values and colors.",
            "prep": "Clean brush, minimal paint. The canvas should still be workable.",
            "execution": [
                "Use a dry brush or fan brush",
                "Light pressure, circular or crisscross motions",
                "Work from light to dark areas",
                "Don't overblend - some brush marks add life"
            ],
            "common_errors": "Overblending makes paintings look photographic and lifeless."
        }
    }

    base = techniques.get(technique, {
        "description": f"Apply {technique} technique to this area.",
        "prep": "Prepare your brush with appropriate paint load.",
        "execution": [f"Use {motion} with your {brush}"],
        "common_errors": "Work confidently and don't overthink."
    })

    return base


def get_brush_instructions(brush):
    """Generate detailed brush handling instructions."""
    # Normalize brush name for matching
    brush_lower = brush.lower() if brush else ""

    if "2-inch" in brush_lower or "2 inch" in brush_lower or "background" in brush_lower:
        return {
            "hold": "Hold at the ferrule for control, or mid-handle for looser strokes.",
            "loading": "Tap both sides into paint, then blend on palette to distribute evenly.",
            "cleaning": "Beat against easel leg or swirl in odorless thinner. Dry on paper towel.",
            "pressure": "Light pressure for glazes, firm for coverage."
        }
    elif "1-inch" in brush_lower or "1 inch" in brush_lower or "landscape" in brush_lower:
        return {
            "hold": "Hold closer to bristles for detail work.",
            "loading": "Pull through paint pile, working both edges.",
            "cleaning": "Swirl in thinner, wipe, repeat until clean.",
            "pressure": "Moderate pressure, adjusting for stroke width needed."
        }
    elif "fan" in brush_lower:
        return {
            "hold": "Hold at the end of the handle for natural movement.",
            "loading": "Touch just the tips into paint - a little goes a long way.",
            "cleaning": "Gentle wipe on paper towel between colors.",
            "pressure": "Very light - let the brush do the work."
        }
    elif "liner" in brush_lower or "script" in brush_lower:
        return {
            "hold": "Hold like a pencil, close to the ferrule.",
            "loading": "Roll brush in thinned paint to form a point.",
            "cleaning": "Wipe gently, reshape point.",
            "pressure": "Minimal pressure, pull don't push."
        }
    elif "filbert" in brush_lower:
        return {
            "hold": "Hold like a pencil for detail, mid-handle for broader strokes.",
            "loading": "Load on one side, blend on palette.",
            "cleaning": "Wipe on paper towel, reshape the rounded tip.",
            "pressure": "Vary pressure for thick-to-thin strokes."
        }
    else:
        return {
            "hold": "Hold comfortably with good control.",
            "loading": "Load paint evenly across bristles.",
            "cleaning": "Clean thoroughly between colors.",
            "pressure": "Adjust pressure as needed for desired effect."
        }


def get_paint_mixing_guide(color_rgb):
    """Generate paint mixing and thinning instructions based on color."""
    # Determine consistency based on luminosity
    luminosity = (0.299 * color_rgb[0] + 0.587 * color_rgb[1] + 0.114 * color_rgb[2]) / 255

    if luminosity > 0.7:
        consistency = "creamy consistency - add small amount of medium"
        thinning = "For acrylics: add 1-2 drops water. For oils: touch of medium."
    elif luminosity > 0.4:
        consistency = "medium consistency - standard tube paint"
        thinning = "Use as-is or thin slightly for better flow."
    else:
        consistency = "thin consistency - good for underlayers"
        thinning = "Thin with medium/water for transparent application."

    return {
        "rgb": color_rgb,
        "consistency": consistency,
        "thinning": thinning
    }


def create_booklet(output_name: str, base_path: Path):
    """
    Create a comprehensive instruction booklet PDF for a processed output.

    Args:
        output_name: Name of the output folder (e.g., "dogs")
        base_path: Base path containing the output folder
    """
    output_dir = base_path / "output" / output_name
    steps_dir = output_dir / "steps" / "cumulative"

    if not output_dir.exists():
        print(f"[ERROR] Output folder not found: {output_dir}")
        return None

    # Load painting guide
    guide_path = output_dir / "painting_guide.json"
    if not guide_path.exists():
        print(f"[ERROR] Painting guide not found: {guide_path}")
        return None

    with open(guide_path, "r") as f:
        guide_data = json.load(f)

    # Load scene analysis for additional context
    analysis_path = output_dir / "scene_analysis.json"
    scene_data = {}
    if analysis_path.exists():
        with open(analysis_path, "r") as f:
            scene_data = json.load(f)

    # Filter to paint steps only
    paint_steps = [s for s in guide_data if s.get('type') == 'paint_substep']
    layer_intros = {s.get('region'): s for s in guide_data if s.get('type') == 'layer_intro'}
    setup = next((s for s in guide_data if s.get('type') == 'setup'), None)
    finish = next((s for s in guide_data if s.get('type') == 'finishing'), None)

    # Create PDF
    pdf_path = output_dir / f"{output_name}_instruction_booklet.pdf"

    # Get title from output name
    title = output_name.replace('_', ' ').title()

    # Get scene context for subtitle
    scene_ctx = scene_data.get('scene_context', {})
    time_of_day = scene_ctx.get('time_of_day', 'day').title()
    weather = scene_ctx.get('weather', 'clear').title()
    setting = scene_ctx.get('setting', 'nature').replace('_', ' ').title()

    print(f"Generating booklet for: {output_name}")

    with PdfPages(pdf_path) as pdf:
        # ===== TITLE PAGE =====
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        ax.text(0.5, 0.75, title, fontsize=42, fontweight='bold',
                ha='center', va='center', color='#2c3e50')
        ax.text(0.5, 0.62, "A Complete Painting Tutorial", fontsize=20,
                ha='center', va='center', color='#666666')
        ax.text(0.5, 0.52, f"{len(paint_steps)} Steps | {time_of_day} | {weather}",
                fontsize=14, ha='center', va='center', color='#888888')

        ax.axhline(y=0.45, xmin=0.25, xmax=0.75, color='#3498db', linewidth=2)

        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # ===== MATERIALS & SETUP PAGE =====
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        ax.text(0.5, 0.92, "Materials & Preparation", fontsize=24, fontweight='bold',
                ha='center', va='top', color='#2c3e50')

        y = 0.82
        if setup:
            ax.text(0.08, y, "MATERIALS NEEDED", fontsize=12, fontweight='bold', color='#444444')
            for material in setup.get('materials', []):
                y -= 0.035
                ax.text(0.1, y, f"- {material}", fontsize=10, color='#333333')

            y -= 0.06
            ax.text(0.08, y, "TIPS BEFORE STARTING", fontsize=12, fontweight='bold', color='#444444')
            for tip in setup.get('tips', []):
                y -= 0.035
                ax.text(0.1, y, f"- {tip}", fontsize=10, color='#333333')

            if setup.get('bob_ross_intro'):
                y -= 0.06
                ax.add_patch(plt.Rectangle((0.08, y - 0.08), 0.84, 0.1,
                                          facecolor='#ecf0f1', edgecolor='#3498db'))
                ax.text(0.5, y - 0.03, f'"{setup["bob_ross_intro"]}"',
                       fontsize=11, fontstyle='italic', ha='center', color='#555555')

        # Brush legend
        y -= 0.14
        ax.text(0.08, y, "BRUSH GUIDE", fontsize=12, fontweight='bold', color='#2c3e50')
        brush_legend = {
            "2-inch brush": "Large coverage areas, skies, backgrounds",
            "1-inch brush": "Medium areas, transitions",
            "Fan brush": "Grass, foliage, texture effects, blending",
            "Script liner": "Details, signatures, fine lines",
            "Filbert brush": "Organic shapes, leaves, petals"
        }
        for brush, use in brush_legend.items():
            y -= 0.035
            ax.text(0.1, y, f"- {brush}: {use}", fontsize=9, color='#555555')

        # Scene info
        y -= 0.08
        ax.text(0.08, y, "SCENE ANALYSIS", fontsize=12, fontweight='bold', color='#2c3e50')
        y -= 0.035
        ax.text(0.1, y, f"Setting: {setting}", fontsize=10, color='#555555')
        y -= 0.03
        ax.text(0.1, y, f"Time: {time_of_day} | Weather: {weather}", fontsize=10, color='#555555')
        y -= 0.03
        ax.text(0.1, y, f"Lighting: {scene_ctx.get('lighting', 'natural').replace('_', ' ').title()}", fontsize=10, color='#555555')
        y -= 0.03
        ax.text(0.1, y, f"Light direction: {scene_ctx.get('light_direction', 'top-left')}", fontsize=10, color='#555555')

        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # ===== REFERENCE IMAGE PAGE =====
        ref_path = output_dir / "colored_reference.png"
        if ref_path.exists():
            fig = plt.figure(figsize=(11, 8.5))
            fig.text(0.5, 0.96, "Reference Image - Your Goal", fontsize=18,
                    fontweight='bold', ha='center', color='#2c3e50')

            ref_img = Image.open(ref_path)
            ax_img = fig.add_axes([0.1, 0.15, 0.8, 0.75])
            ax_img.imshow(ref_img)
            ax_img.axis('off')

            fig.text(0.5, 0.08, "Study this image before beginning. Notice the values, colors, and composition.",
                    fontsize=10, ha='center', color='#666666')

            pdf.savefig(fig, dpi=150)
            plt.close(fig)

        # ===== STEP PAGES =====
        current_region = None
        for step_data in paint_steps:
            step_num = int(step_data['step'])
            parent_region = step_data.get('parent_region', '')

            # Add region intro page if new region
            if parent_region != current_region and parent_region in layer_intros:
                current_region = parent_region
                intro = layer_intros[parent_region]

                fig, ax = plt.subplots(figsize=(11, 8.5))
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')

                ax.text(0.5, 0.65, f"Beginning: {parent_region}", fontsize=28, fontweight='bold',
                        ha='center', va='center', color='#2c3e50')

                if intro.get('bob_ross_intro'):
                    ax.text(0.5, 0.5, f'"{intro["bob_ross_intro"]}"',
                           fontsize=14, fontstyle='italic', ha='center', color='#555555')

                tips = intro.get('tips', [])
                y = 0.38
                for tip in tips[:3]:
                    ax.text(0.5, y, f"- {tip}", fontsize=11, ha='center', color='#666666')
                    y -= 0.04

                pdf.savefig(fig, dpi=150)
                plt.close(fig)

            # Find step image
            step_name = step_data['name'].replace(' ', '_').replace('-', '_')
            step_files = list(steps_dir.glob(f"step_{step_num:02d}_*.png"))
            if not step_files:
                continue
            step_img_path = step_files[0]
            img = Image.open(step_img_path)

            # ----- PAGE 1: Visual Reference -----
            fig = plt.figure(figsize=(11, 8.5))

            fig.text(0.5, 0.96, f"STEP {step_num}: {step_data['name'].upper()}",
                    fontsize=14, fontweight='bold', ha='center', color='#2c3e50')

            ax_img = fig.add_axes([0.1, 0.2, 0.8, 0.7])
            ax_img.imshow(img)
            ax_img.axis('off')

            # Info bar
            fig.text(0.12, 0.12, f"Brush: {step_data.get('brush', 'N/A')}", fontsize=10, color='#444444')
            fig.text(0.12, 0.08, f"Strokes: {step_data.get('strokes', 'N/A')}", fontsize=10, color='#444444')
            fig.text(0.12, 0.04, f"Technique: {step_data.get('technique', 'N/A')}", fontsize=10, color='#666666')

            if 'dominant_color' in step_data:
                color_rgb = step_data['dominant_color']
                color_hex = rgb_to_hex(color_rgb)
                swatch = plt.Rectangle((0.7, 0.105), 0.05, 0.03,
                                       facecolor=color_hex, edgecolor='#666666',
                                       transform=fig.transFigure, clip_on=False)
                fig.add_artist(swatch)
                fig.text(0.76, 0.115, f"RGB({color_rgb[0]}, {color_rgb[1]}, {color_rgb[2]})",
                        fontsize=8, color='#666666')

            pdf.savefig(fig, dpi=150)
            plt.close(fig)

            # ----- PAGE 2: Detailed Instructions -----
            fig = plt.figure(figsize=(11, 8.5))

            fig.text(0.5, 0.96, f"STEP {step_num}: {step_data['name']} - Instructions",
                    fontsize=14, fontweight='bold', ha='center', color='#2c3e50')

            ax_text = fig.add_axes([0.08, 0.08, 0.84, 0.85])
            ax_text.set_xlim(0, 1)
            ax_text.set_ylim(0, 1)
            ax_text.axis('off')

            y = 0.90

            # Technique details
            technique = step_data.get('technique', 'general')
            brush = step_data.get('brush', 'brush')
            motion = step_data.get('strokes', 'standard strokes')
            tech_details = get_technique_details(technique, brush, motion)

            ax_text.text(0, y, f"Technique: {technique.title()}", fontsize=12, fontweight='bold', color='#2c3e50')
            y -= 0.035
            ax_text.text(0.02, y, tech_details['description'], fontsize=10, color='#333333')
            y -= 0.04

            ax_text.text(0, y, "Preparation:", fontsize=11, fontweight='bold', color='#444444')
            y -= 0.03
            ax_text.text(0.02, y, tech_details['prep'], fontsize=9, color='#555555')
            y -= 0.04

            ax_text.text(0, y, "Execution Steps:", fontsize=11, fontweight='bold', color='#444444')
            for exec_step in tech_details['execution']:
                y -= 0.03
                ax_text.text(0.02, y, f"- {exec_step}", fontsize=9, color='#555555')
            y -= 0.04

            ax_text.text(0, y, "Common Error to Avoid:", fontsize=10, fontweight='bold', color='#c0392b')
            y -= 0.03
            ax_text.text(0.02, y, tech_details['common_errors'], fontsize=9, fontstyle='italic', color='#c0392b')
            y -= 0.05

            # Brush info
            brush_info = get_brush_instructions(brush)
            ax_text.text(0, y, f"Using the {brush}:", fontsize=11, fontweight='bold', color='#444444')
            y -= 0.035
            ax_text.text(0.02, y, f"Hold: {brush_info['hold']}", fontsize=9, color='#555555')
            y -= 0.028
            ax_text.text(0.02, y, f"Loading: {brush_info['loading']}", fontsize=9, color='#555555')
            y -= 0.028
            ax_text.text(0.02, y, f"Pressure: {brush_info['pressure']}", fontsize=9, color='#555555')
            y -= 0.05

            # Paint mixing guide
            if 'dominant_color' in step_data:
                paint_guide = get_paint_mixing_guide(step_data['dominant_color'])
                ax_text.text(0, y, "Paint Preparation:", fontsize=11, fontweight='bold', color='#444444')
                y -= 0.03
                ax_text.text(0.02, y, f"Consistency: {paint_guide['consistency']}", fontsize=9, color='#555555')
                y -= 0.028
                ax_text.text(0.02, y, f"Thinning: {paint_guide['thinning']}", fontsize=9, color='#555555')
                y -= 0.05

            # Main instruction
            instruction = step_data.get('instruction', '')
            if instruction:
                ax_text.text(0, y, "What to Do:", fontsize=11, fontweight='bold', color='#444444')
                y -= 0.03
                for line in wrap_text(instruction, width=95).split('\n')[:4]:
                    ax_text.text(0.02, y, line, fontsize=9, color='#333333')
                    y -= 0.028

            # Coverage info
            y -= 0.04
            coverage = step_data.get('coverage', 'N/A')
            lum_range = step_data.get('luminosity_range', [0, 1])
            ax_text.text(0, y, f"Coverage: {coverage} | Value Range: {lum_range[0]:.2f} - {lum_range[1]:.2f}",
                        fontsize=9, color='#666666')

            # Tips
            tips = step_data.get('tips', [])
            if tips:
                y -= 0.05
                ax_text.text(0, y, "Tips:", fontsize=10, fontweight='bold', color='#666666')
                for tip in tips[:2]:
                    y -= 0.028
                    ax_text.text(0.02, y, f"- {tip}", fontsize=9, fontstyle='italic', color='#666666')

            pdf.savefig(fig, dpi=150)
            plt.close(fig)

        # ===== COMPLETION PAGE =====
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        ax.text(0.5, 0.7, "Congratulations!", fontsize=36, fontweight='bold',
                ha='center', va='center', color='#27ae60')

        if finish:
            ax.text(0.5, 0.55, wrap_text(finish.get('bob_ross_outro', ''), width=70),
                   fontsize=12, ha='center', va='center', color='#444444')

            tips = finish.get('tips', [])
            y = 0.38
            for tip in tips:
                ax.text(0.5, y, f"- {tip}", fontsize=10, ha='center', color='#666666')
                y -= 0.04

        ax.text(0.5, 0.1, "Happy Painting!", fontsize=18, fontweight='bold',
               ha='center', color='#27ae60')

        pdf.savefig(fig, dpi=150)
        plt.close(fig)

    print(f"  Created: {pdf_path}")
    return pdf_path


def main():
    """Main entry point."""
    base_path = Path(__file__).parent

    # Get list of output folders
    output_path = base_path / "output"
    if not output_path.exists():
        print("No output folder found!")
        return

    # Default: all output folders with painting_guide.json
    if len(sys.argv) > 1:
        folders = sys.argv[1:]
    else:
        folders = [
            d.name for d in output_path.iterdir()
            if d.is_dir() and (d / "painting_guide.json").exists()
        ]

    if not folders:
        print("No valid output folders found!")
        return

    print()
    print("=" * 70)
    print("GENERATING INSTRUCTION BOOKLETS")
    print("=" * 70)
    print(f"Folders: {folders}")
    print()

    results = {}
    for folder in folders:
        pdf_path = create_booklet(folder, base_path)
        results[folder] = pdf_path is not None

    # Summary
    print()
    print("=" * 70)
    print("BOOKLET GENERATION SUMMARY")
    print("=" * 70)

    for folder, success in results.items():
        status = "OK" if success else "FAILED"
        print(f"  {folder}: {status}")

    print()
    print("Booklets are in each output folder:")
    for folder in folders:
        pdf_path = base_path / "output" / folder / f"{folder}_instruction_booklet.pdf"
        if pdf_path.exists():
            size_mb = pdf_path.stat().st_size / (1024 * 1024)
            print(f"  output/{folder}/{folder}_instruction_booklet.pdf ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
