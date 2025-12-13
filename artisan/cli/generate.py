#!/usr/bin/env python3
"""
PaintX - Main Generation Script
================================

Command-line interface for generating paint-by-numbers kits.

This script is the main entry point for the PaintX system. It handles:
- Finding input images in the expected folder structure
- Generating instruction sets at different skill levels
- Creating paint kits with shopping lists and mixing guides
- Budget comparison and optimization

Folder Structure:
    input/<image_name>/<image_name>.png   # Source images
    output/<image_name>/<level>/...        # Generated outputs

Usage:
    python generate.py <image_name> [level] [n_colors] [canvas_size] [budget]

Examples:
    # Generate all instruction levels + paint kit
    python generate.py aurora_maria all

    # Generate only advanced (Bob Ross style) instructions
    python generate.py aurora_maria advanced

    # Generate paint kit with $75 budget
    python generate.py aurora_maria kit 15 16x20 75

    # Compare what different budgets provide
    python generate.py aurora_maria compare

    # Generate kits for all budget tiers
    python generate.py aurora_maria compare-all

Instruction Levels:
    1 / beginner     - Simple paint-by-numbers (fill numbered regions)
    2 / easy         - Guided with suggested layer order
    3 / intermediate - Technique-aware with brush recommendations
    4 / advanced     - Bob Ross style step-by-step guidance
    kit              - Paint shopping list and mixing guide only
    compare          - Compare budget tiers (no file generation)
    compare-all      - Generate kits for all budget tiers
    all              - Generate all instruction levels + paint kit

Budget Tiers:
    $40-55   Minimal      - 5 paints, heavy mixing required
    $55-80   Budget       - 6-7 paints, moderate mixing
    $80-120  Standard     - 8-10 paints, light mixing
    $120-180 Premium      - 12-14 paints, minimal mixing
    $180+    Professional - 20+ paints, near-perfect matching

Canvas Sizes:
    8x10, 9x12, 11x14, 12x16, 16x20, 18x24, 24x30, 24x36 (inches)
"""

import os
import sys
import glob
import shutil

# Add grandparent directory to path so artisan is a proper package
_artisan_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_project_dir = os.path.dirname(_artisan_dir)
sys.path.insert(0, _project_dir)

from artisan.generators.instruction_generator import UnifiedInstructionGenerator, InstructionLevel
from artisan.generators.bob_ross import BobRossGenerator
from artisan.generators.paint_kit_generator import PaintKitGenerator, CANVAS_SIZES
from artisan.generators.smart_paint_by_numbers import SmartPaintByNumbers
from artisan.optimization.budget_optimizer import BudgetOptimizer, BUDGET_TIERS


# Level name mappings
LEVEL_MAP = {
    '1': InstructionLevel.BEGINNER,
    '2': InstructionLevel.EASY,
    '3': InstructionLevel.INTERMEDIATE,
    '4': InstructionLevel.ADVANCED,
    'beginner': InstructionLevel.BEGINNER,
    'easy': InstructionLevel.EASY,
    'intermediate': InstructionLevel.INTERMEDIATE,
    'advanced': InstructionLevel.ADVANCED,
}

LEVEL_NAMES = {
    InstructionLevel.BEGINNER: 'beginner',
    InstructionLevel.EASY: 'easy',
    InstructionLevel.INTERMEDIATE: 'intermediate',
    InstructionLevel.ADVANCED: 'advanced',
}


def find_input_image(image_name: str) -> str:
    """
    Find the input image in the expected folder structure.

    Looks for:
      input/<image_name>/<image_name>.png
      input/<image_name>/<image_name>.jpg
      input/<image_name>/*.png (first found)
      input/<image_name>/*.jpg (first found)
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # artisan root
    input_dir = os.path.join(base_dir, 'input', image_name)

    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    # Try exact match first
    for ext in ['.png', '.jpg', '.jpeg']:
        exact_path = os.path.join(input_dir, f"{image_name}{ext}")
        if os.path.isfile(exact_path):
            return exact_path

    # Try any image in the folder
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        matches = glob.glob(os.path.join(input_dir, ext))
        if matches:
            return matches[0]

    raise FileNotFoundError(f"No image found in: {input_dir}")


def get_output_dir(image_name: str, level: InstructionLevel) -> str:
    """Get the output directory for a given image and level."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # artisan root
    level_name = LEVEL_NAMES[level]
    return os.path.join(base_dir, 'output', image_name, level_name)


def generate_level(image_path: str, image_name: str, level: InstructionLevel,
                   n_colors: int = 15, canvas_size: str = "16x20"):
    """Generate instructions for a single level."""
    output_dir = get_output_dir(image_name, level)

    # Clean output directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    level_name = LEVEL_NAMES[level]

    print(f"\n{'='*60}")
    print(f"Generating: {image_name} / {level_name.upper()}")
    print(f"{'='*60}")
    print(f"Input:  {image_path}")
    print(f"Output: {output_dir}")

    if level == InstructionLevel.ADVANCED:
        # Use Bob Ross generator for advanced level
        generator = BobRossGenerator(image_path, n_colors)
        generator.generate_steps()
        generator.create_step_by_step_guide(output_dir, include_substeps=False)
    else:
        # Use unified generator for other levels
        generator = UnifiedInstructionGenerator(image_path, n_colors)
        generator.generate(level, output_dir)

    print(f"\nGenerated {level_name} instructions to: {output_dir}")
    return output_dir


def compare_budgets(image_path: str, image_name: str, n_colors: int = 15,
                   generate_kits: bool = False, canvas_size: str = "16x20"):
    """Compare different budget tiers for a given image palette."""
    from paint_by_numbers import PaintByNumbers

    print(f"\n{'='*70}")
    print(f"BUDGET COMPARISON: {image_name}")
    print(f"{'='*70}")

    # Process image to get palette
    pbn = PaintByNumbers(image_path, n_colors)
    pbn.quantize_colors()

    palette = [tuple(int(c) for c in color) for color in pbn.palette]
    pixel_counts = [pbn.color_counts.get(i, 0) for i in range(n_colors)]

    optimizer = BudgetOptimizer(palette, pixel_counts)

    # Compare different budgets
    budgets = [50, 75, 100, 150, 200]

    print(f"\n{'Budget':<10} {'Tier':<12} {'Paints':<8} {'Accuracy':<10} {'Cost':<10} {'Mixing':<12}")
    print("-" * 70)

    for budget in budgets:
        analysis = optimizer.analyze_budget(budget)
        paint_keys, cost = optimizer.get_optimal_paint_set(budget)

        print(f"${budget:<9} {analysis.tier.name:<12} {len(paint_keys):<8} "
              f"{analysis.average_accuracy:.0f}%{'':<6} ${cost:.0f}{'':<6} "
              f"{analysis.max_mixing_complexity.name:<12}")

    print("-" * 70)

    # Find recommended budget
    min_budget_85 = optimizer.find_minimum_budget(85)
    min_budget_95 = optimizer.find_minimum_budget(95)

    print(f"\nRecommendations:")
    print(f"  Minimum for 85% accuracy: ${min_budget_85}")
    print(f"  Minimum for 95% accuracy: ${min_budget_95}")
    print(f"\nTip: Run with specific budget:")
    print(f"  python generate.py {image_name} kit {n_colors} 16x20 {min_budget_85}")

    # Generate kits for each budget tier if requested
    if generate_kits:
        print(f"\n{'='*70}")
        print(f"GENERATING KITS FOR ALL BUDGET TIERS")
        print(f"{'='*70}")
        for budget in budgets:
            generate_paint_kit(image_path, image_name, n_colors, canvas_size, budget)

    return optimizer


def generate_budget_comparison_report(image_path: str, image_name: str,
                                      n_colors: int = 15, canvas_size: str = "16x20"):
    """Generate kits at multiple budget tiers for side-by-side comparison."""
    from artisan.core.paint_by_numbers import PaintByNumbers
    import json

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # artisan root
    output_dir = os.path.join(base_dir, 'output', image_name, 'budget_comparison')

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"GENERATING BUDGET COMPARISON REPORT: {image_name}")
    print(f"{'='*70}")

    # Process image
    pbn = PaintByNumbers(image_path, n_colors)
    pbn.quantize_colors()

    palette = [tuple(int(c) for c in color) for color in pbn.palette]
    pixel_counts = [pbn.color_counts.get(i, 0) for i in range(n_colors)]

    optimizer = BudgetOptimizer(palette, pixel_counts)

    budgets = [50, 75, 100, 150]
    report = {
        "image": image_name,
        "n_colors": n_colors,
        "canvas_size": canvas_size,
        "budget_tiers": []
    }

    for budget in budgets:
        analysis = optimizer.analyze_budget(budget)
        paint_keys, cost = optimizer.get_optimal_paint_set(budget)

        tier_info = {
            "budget": budget,
            "tier_name": analysis.tier.name,
            "paint_count": len(paint_keys),
            "paint_cost": cost,
            "accuracy_percent": round(analysis.average_accuracy, 1),
            "mixing_complexity": analysis.max_mixing_complexity.name,
            "colors_achievable": analysis.colors_achievable,
        }
        report["budget_tiers"].append(tier_info)

        # Generate kit for this budget
        generate_paint_kit(image_path, image_name, n_colors, canvas_size, budget)

    # Save comparison report
    report_path = os.path.join(output_dir, "comparison_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*70}")
    print(f"COMPARISON REPORT COMPLETE")
    print(f"{'='*70}")
    print(f"Report saved to: {report_path}")
    print(f"\nKits generated for budgets: {budgets}")
    print(f"Check output/{image_name}/paint_kit_$XX/ folders")

    return report


def generate_paint_kit(image_path: str, image_name: str, n_colors: int = 15,
                       canvas_size: str = "16x20", budget: float = None):
    """Generate paint kit (shopping list, mixing guide, etc.)."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # artisan root

    # Use different folder name for budget kits
    folder_name = f'paint_kit_${int(budget)}' if budget else 'paint_kit'
    output_dir = os.path.join(base_dir, 'output', image_name, folder_name)

    # Clean output directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Generating Paint Kit: {image_name}")
    print(f"{'='*60}")
    print(f"Canvas size: {canvas_size}")
    if budget:
        print(f"Budget: ${budget:.0f}")
    print(f"Output: {output_dir}")

    generator = PaintKitGenerator(image_path, n_colors, canvas_size, budget)
    kit = generator.generate_kit()
    generator.generate_all_outputs(output_dir)

    # Print budget analysis summary if available
    if kit.budget_info:
        bi = kit.budget_info
        print(f"\nBudget Analysis:")
        print(f"  Tier: {bi['tier']} - {bi['tier_description']}")
        print(f"  Accuracy: {bi['average_accuracy']:.0f}%")
        print(f"  Paints: {bi['paints_in_set']}")
        print(f"  Mixing complexity: {bi['max_mixing_complexity']}")

    return output_dir, kit


def generate_organic(image_path: str, image_name: str, n_colors: int = 15,
                     target_layers: int = 12, full: bool = False):
    """Generate organic layer-based painting guide using SmartPaintByNumbers.

    Args:
        full: If True, generate all outputs. If False (default), generate minimal outputs.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # artisan root
    output_dir = os.path.join(base_dir, 'output', image_name, 'organic')

    # Clean output directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Generating: {image_name} / ORGANIC {'(full)' if full else '(minimal)'}")
    print(f"{'='*60}")
    print(f"Input:  {image_path}")
    print(f"Output: {output_dir}")

    spbn = SmartPaintByNumbers(
        image_path,
        n_colors=n_colors,
        use_organic_layers=True,
        target_layers=target_layers
    )
    spbn.process()
    spbn.save_all(output_dir, minimal=not full)

    print(f"\nGenerated organic layer guide to: {output_dir}")
    return output_dir


def generate_all_levels(image_path: str, image_name: str, n_colors: int = 15,
                        canvas_size: str = "16x20"):
    """Generate all instruction levels for an image."""
    print(f"\n{'='*60}")
    print(f"GENERATING ALL LEVELS FOR: {image_name}")
    print(f"{'='*60}")

    for level in InstructionLevel:
        generate_level(image_path, image_name, level, n_colors, canvas_size)

    # Also generate paint kit
    generate_paint_kit(image_path, image_name, n_colors, canvas_size)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # artisan root
    output_base = os.path.join(base_dir, 'output', image_name)

    print(f"\n{'='*60}")
    print(f"ALL LEVELS COMPLETE")
    print(f"{'='*60}")
    print(f"Output folder: {output_base}/")
    print(f"  - beginner/")
    print(f"  - easy/")
    print(f"  - intermediate/")
    print(f"  - advanced/")
    print(f"  - paint_kit/")

    return output_base


def list_available_inputs():
    """List all available input images."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # artisan root
    input_dir = os.path.join(base_dir, 'input')

    if not os.path.isdir(input_dir):
        print("No input folder found. Create input/<image_name>/<image>.png")
        return []

    inputs = []
    for item in os.listdir(input_dir):
        item_path = os.path.join(input_dir, item)
        if os.path.isdir(item_path):
            # Check if it has images
            images = glob.glob(os.path.join(item_path, '*.png')) + \
                    glob.glob(os.path.join(item_path, '*.jpg'))
            if images:
                inputs.append(item)

    return inputs


def main():
    if len(sys.argv) < 2:
        print("PaintX - Painting Instruction Generator")
        print("=" * 50)
        print("\nUsage:")
        print("  python generate.py <image_name> [level] [n_colors] [canvas_size] [budget]")
        print("\nExamples:")
        print("  python generate.py aurora_maria advanced")
        print("  python generate.py aurora_maria all")
        print("  python generate.py aurora_maria kit")
        print("  python generate.py aurora_maria kit 15 16x20 75   # $75 budget")
        print("  python generate.py aurora_maria kit 15 16x20 50   # $50 budget (more mixing)")
        print("\nLevels:")
        print("  1 / beginner     - Simple paint-by-numbers")
        print("  2 / easy         - Guided with layer order")
        print("  3 / intermediate - Technique-aware")
        print("  4 / advanced     - Bob Ross style")
        print("  organic          - Smart organic layer detection (minimal output)")
        print("  organic-full     - Organic with all outputs")
        print("  kit              - Paint kit only (shopping list, mixing guide)")
        print("  compare          - Compare budget tiers for this image")
        print("  compare-all      - Compare AND generate kits for all budget tiers")
        print("  all              - Generate all levels + paint kit")
        print("\nBudget Tiers (for kit mode):")
        print("  $40-55   Minimal   - 5 paints, heavy mixing")
        print("  $55-80   Budget    - 7 paints, moderate mixing")
        print("  $80-120  Standard  - 10 paints, light mixing")
        print("  $120-180 Premium   - 14 paints, minimal mixing")
        print("  $180+    Professional - 20+ paints")
        print("\nCanvas sizes:", ", ".join(CANVAS_SIZES.keys()))
        print("\nFolder Structure:")
        print("  input/<image_name>/<image>.png")
        print("  output/<image_name>/<level>/...")

        # List available inputs
        available = list_available_inputs()
        if available:
            print(f"\nAvailable inputs: {', '.join(available)}")
        else:
            print("\nNo inputs found. Create: input/<name>/<name>.png")

        sys.exit(0)

    image_name = sys.argv[1]
    level_arg = sys.argv[2] if len(sys.argv) > 2 else 'all'
    n_colors = int(sys.argv[3]) if len(sys.argv) > 3 else 15
    canvas_size = sys.argv[4] if len(sys.argv) > 4 else "16x20"
    budget = float(sys.argv[5]) if len(sys.argv) > 5 else None

    # Find the input image
    try:
        image_path = find_input_image(image_name)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        available = list_available_inputs()
        if available:
            print(f"Available inputs: {', '.join(available)}")
        sys.exit(1)

    # Generate
    if level_arg.lower() == 'all':
        generate_all_levels(image_path, image_name, n_colors, canvas_size)
    elif level_arg.lower() == 'organic':
        generate_organic(image_path, image_name, n_colors, full=False)
    elif level_arg.lower() == 'organic-full':
        generate_organic(image_path, image_name, n_colors, full=True)
    elif level_arg.lower() == 'kit':
        generate_paint_kit(image_path, image_name, n_colors, canvas_size, budget)
    elif level_arg.lower() == 'compare':
        compare_budgets(image_path, image_name, n_colors)
    elif level_arg.lower() == 'compare-all':
        generate_budget_comparison_report(image_path, image_name, n_colors, canvas_size)
    else:
        level_key = level_arg.lower()
        if level_key not in LEVEL_MAP:
            print(f"Unknown level: {level_arg}")
            print("Valid levels: beginner, easy, intermediate, advanced, organic, organic-full, kit, all")
            sys.exit(1)

        level = LEVEL_MAP[level_key]
        generate_level(image_path, image_name, level, n_colors, canvas_size)


if __name__ == "__main__":
    main()
