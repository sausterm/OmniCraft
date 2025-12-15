"""
Paint Kit Generator
===================

Complete paint buying and mixing guide generator.

This module creates everything needed to purchase paints and create
a painting from the generated template:

Generated Outputs:
    - Shopping list (PNG): Visual list of paints to buy with quantities/costs
    - Color chart (PNG): All colors with RGB values and paint matches
    - Mixing guide (PNG): Step-by-step instructions for mixing colors
    - Paint kit (JSON): Machine-readable export of all kit data

Features:
    - Canvas size-based quantity calculations
    - Commercial paint database matching
    - Budget-aware paint selection (when budget specified)
    - Mixing recipe generation for colors requiring mixing
    - Coverage calculations with safety margins

Example Usage:
    >>> from paint_kit_generator import PaintKitGenerator
    >>>
    >>> # Standard kit (no budget constraint)
    >>> kit_gen = PaintKitGenerator('image.jpg', n_colors=15, canvas_size='16x20')
    >>> kit = kit_gen.generate_kit()
    >>> kit_gen.generate_all_outputs('output/')
    >>>
    >>> # Budget-constrained kit ($75 budget)
    >>> kit_gen = PaintKitGenerator('image.jpg', n_colors=15, budget=75)
    >>> kit = kit_gen.generate_kit()
    >>> print(f"Total cost: ${kit.total_cost}")
    >>> print(f"Paints to buy: {len(kit.shopping_list)}")
    >>> print(f"Colors needing mixing: {len(kit.mixed_colors)}")

Canvas Sizes Supported:
    8x10, 9x12, 11x14, 12x16, 16x20, 18x24, 24x30, 24x36 inches

Paint Quantity Calculation:
    - Based on canvas area and color coverage percentage
    - Assumes ~3.5 sq ft coverage per oz of paint
    - Includes 30% safety margin for mixing and corrections
    - Rounds up to whole 2oz tubes

Classes:
    - PaintKitGenerator: Main kit generation class
    - PaintKit: Complete kit data container
    - PaintQuantity: Single paint item in shopping list
    - MixedColor: Color requiring mixing with recipe
    - CanvasSize: Standard canvas dimensions
"""

import os
import json
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict

from artisan.core.paint_by_numbers import PaintByNumbers
from artisan.core.paint_database import Paint, PAINT_DATABASE, get_paint, PaintBrand
from artisan.core.color_matcher import ColorMatcher, ColorSolution, match_palette
from artisan.paint.optimization.budget_optimizer import BudgetOptimizer, BudgetAnalysis, PAINT_SETS, BUDGET_TIERS


@dataclass
class CanvasSize:
    """Standard canvas sizes."""
    name: str
    width_inches: float
    height_inches: float
    area_sqft: float


# Common canvas sizes
CANVAS_SIZES = {
    "8x10": CanvasSize("8x10 inches", 8, 10, 8*10/144),
    "9x12": CanvasSize("9x12 inches", 9, 12, 9*12/144),
    "11x14": CanvasSize("11x14 inches", 11, 14, 11*14/144),
    "12x16": CanvasSize("12x16 inches", 12, 16, 12*16/144),
    "16x20": CanvasSize("16x20 inches", 16, 20, 16*20/144),
    "18x24": CanvasSize("18x24 inches", 18, 24, 18*24/144),
    "24x30": CanvasSize("24x30 inches", 24, 30, 24*30/144),
    "24x36": CanvasSize("24x36 inches", 24, 36, 24*36/144),
}


@dataclass
class PaintQuantity:
    """Paint quantity needed for a color."""
    paint_key: str
    paint_name: str
    brand: str
    oz_needed: float
    tubes_2oz: int
    price_per_tube: float
    total_price: float
    color_numbers: List[int]  # Which color numbers use this paint


@dataclass
class MixedColor:
    """A color that needs mixing."""
    color_number: int
    target_rgb: Tuple[int, int, int]
    recipe_paints: List[Tuple[str, float]]  # (paint_name, ratio)
    mixing_instructions: str
    result_rgb: Tuple[int, int, int]


@dataclass
class PaintKit:
    """Complete paint kit for a painting."""
    image_name: str
    canvas_size: CanvasSize
    n_colors: int
    shopping_list: List[PaintQuantity]
    mixed_colors: List[MixedColor]
    total_cost: float
    colors_info: List[Dict]
    budget_info: Optional[Dict] = None  # Budget analysis info if budget mode used


class PaintKitGenerator:
    """Generates complete paint kits for paintings."""

    # Paint coverage assumptions
    COVERAGE_SQFT_PER_OZ = 3.5  # Average
    SAFETY_MARGIN = 1.3  # Order 30% extra

    def __init__(self, image_path: str, n_colors: int = 15,
                 canvas_size: str = "16x20", budget: float = None):
        """
        Initialize paint kit generator.

        Args:
            image_path: Path to source image
            n_colors: Number of colors in palette
            canvas_size: Canvas size key (e.g., "16x20")
            budget: Optional budget constraint in USD (paint cost only)
        """
        self.image_path = image_path
        self.n_colors = n_colors
        self.canvas = CANVAS_SIZES.get(canvas_size, CANVAS_SIZES["16x20"])
        self.budget = budget

        # Process image
        self.pbn = PaintByNumbers(image_path, n_colors)
        self.pbn.quantize_colors()
        self.pbn.create_regions()

        # Get image dimensions for coverage calculations
        self.img_height, self.img_width = self.pbn.original_image.shape[:2]
        self.total_pixels = self.img_height * self.img_width

        # Color matcher
        self.matcher = ColorMatcher()

        # Budget optimizer
        self.budget_analysis: Optional[BudgetAnalysis] = None

        # Results
        self.kit: Optional[PaintKit] = None

    def generate_kit(self) -> PaintKit:
        """Generate the complete paint kit."""
        # Get palette and pixel counts
        palette = [tuple(int(c) for c in color) for color in self.pbn.palette]
        pixel_counts = [self.pbn.color_counts.get(i, 0) for i in range(self.n_colors)]

        # Use budget optimizer if budget is specified
        if self.budget is not None:
            return self._generate_budget_kit(palette, pixel_counts)

        # Match colors using standard method
        solutions = []
        for i, rgb in enumerate(palette):
            solution = self.matcher.solve_color(rgb, f"Color {i+1}")
            solutions.append(solution)

        # Calculate quantities
        paint_usage = defaultdict(lambda: {"oz": 0, "colors": []})
        mixed_colors = []
        colors_info = []

        for i, (solution, pixel_count) in enumerate(zip(solutions, pixel_counts)):
            coverage_fraction = pixel_count / self.total_pixels
            area_sqft = self.canvas.area_sqft * coverage_fraction

            color_info = {
                "number": i + 1,
                "rgb": solution.target_rgb,
                "hex": "#{:02x}{:02x}{:02x}".format(*solution.target_rgb),
                "coverage_percent": round(coverage_fraction * 100, 1),
                "recommended": solution.recommended,
            }

            if solution.recommended == "single" and solution.best_single:
                # Direct paint match
                paint_key = solution.best_single.paint_key
                paint = solution.best_single.paint
                oz_needed = self._calculate_oz_needed(area_sqft, paint)

                paint_usage[paint_key]["oz"] += oz_needed
                paint_usage[paint_key]["colors"].append(i + 1)
                paint_usage[paint_key]["paint"] = paint

                color_info["paint"] = paint.name
                color_info["match_confidence"] = round(solution.best_single.confidence * 100)

            elif solution.best_mix:
                # Needs mixing
                recipe = solution.best_mix
                total_oz = self._calculate_oz_needed(area_sqft)

                for paint_key, paint, ratio in recipe.paints:
                    oz_for_color = total_oz * ratio
                    paint_usage[paint_key]["oz"] += oz_for_color
                    paint_usage[paint_key]["colors"].append(i + 1)
                    paint_usage[paint_key]["paint"] = paint

                mixed_colors.append(MixedColor(
                    color_number=i + 1,
                    target_rgb=solution.target_rgb,
                    recipe_paints=[(p[1].name, round(p[2] * 100)) for p in recipe.paints],
                    mixing_instructions=recipe.mixing_instructions,
                    result_rgb=recipe.result_rgb
                ))

                color_info["mixing_required"] = True
                color_info["recipe"] = [
                    {"paint": p[1].name, "percent": round(p[2] * 100)}
                    for p in recipe.paints
                ]

            colors_info.append(color_info)

        # Create shopping list
        shopping_list = []
        for paint_key, usage in paint_usage.items():
            paint = usage["paint"]
            oz_needed = usage["oz"] * self.SAFETY_MARGIN
            tubes_needed = max(1, int(np.ceil(oz_needed / 2)))  # 2oz tubes

            shopping_list.append(PaintQuantity(
                paint_key=paint_key,
                paint_name=paint.name,
                brand=paint.brand.value,
                oz_needed=round(oz_needed, 2),
                tubes_2oz=tubes_needed,
                price_per_tube=paint.price_2oz,
                total_price=round(tubes_needed * paint.price_2oz, 2),
                color_numbers=usage["colors"]
            ))

        # Sort by total price (most expensive first)
        shopping_list.sort(key=lambda x: x.total_price, reverse=True)

        total_cost = sum(item.total_price for item in shopping_list)

        self.kit = PaintKit(
            image_name=os.path.basename(self.image_path),
            canvas_size=self.canvas,
            n_colors=self.n_colors,
            shopping_list=shopping_list,
            mixed_colors=mixed_colors,
            total_cost=round(total_cost, 2),
            colors_info=colors_info
        )

        return self.kit

    def _calculate_oz_needed(self, area_sqft: float,
                            paint: Optional[Paint] = None) -> float:
        """Calculate ounces of paint needed for an area."""
        coverage = paint.coverage_sqft_per_oz if paint else self.COVERAGE_SQFT_PER_OZ
        return area_sqft / coverage

    def _generate_budget_kit(self, palette: List[Tuple[int, int, int]],
                            pixel_counts: List[int]) -> PaintKit:
        """
        Generate paint kit with budget constraints.
        Uses limited paint set and provides mixing recipes.
        """
        # Run budget analysis
        optimizer = BudgetOptimizer(palette, pixel_counts)
        self.budget_analysis = optimizer.analyze_budget(self.budget)

        # Get the optimal paint set for this budget
        paint_keys = [k for k in PAINT_SETS.get(
            len(self.budget_analysis.paint_set),
            PAINT_SETS[5]
        )]

        # Find optimal paint set that fits budget
        for n in sorted(PAINT_SETS.keys(), reverse=True):
            test_cost = sum(get_paint(k).price_2oz for k in PAINT_SETS[n] if get_paint(k))
            if test_cost <= self.budget:
                paint_keys = PAINT_SETS[n]
                break

        paint_usage = defaultdict(lambda: {"oz": 0, "colors": []})
        mixed_colors = []
        colors_info = []

        for i, (rgb, pixel_count, analysis) in enumerate(
            zip(palette, pixel_counts, self.budget_analysis.color_details)
        ):
            coverage_fraction = pixel_count / self.total_pixels
            area_sqft = self.canvas.area_sqft * coverage_fraction
            total_oz = self._calculate_oz_needed(area_sqft)

            color_info = {
                "number": i + 1,
                "rgb": rgb,
                "hex": "#{:02x}{:02x}{:02x}".format(*rgb),
                "coverage_percent": round(coverage_fraction * 100, 1),
                "accuracy_percent": round(analysis.accuracy_percent, 0),
            }

            # Use the recipe from budget analysis
            for paint_name, ratio in analysis.recipe:
                # Find paint key from name
                paint_key = None
                for k, p in PAINT_DATABASE.items():
                    if p.name == paint_name:
                        paint_key = k
                        break

                if paint_key:
                    paint = get_paint(paint_key)
                    oz_for_color = total_oz * ratio
                    paint_usage[paint_key]["oz"] += oz_for_color
                    paint_usage[paint_key]["colors"].append(i + 1)
                    paint_usage[paint_key]["paint"] = paint

            # Determine if mixing is required
            if len(analysis.recipe) == 1:
                color_info["paint"] = analysis.recipe[0][0]
                color_info["match_confidence"] = round(analysis.accuracy_percent)
            else:
                color_info["mixing_required"] = True
                color_info["recipe"] = [
                    {"paint": name, "percent": round(ratio * 100)}
                    for name, ratio in analysis.recipe
                ]

                # Create mixing instructions
                recipe_str = " + ".join([
                    f"{name} ({round(ratio * 100)}%)"
                    for name, ratio in analysis.recipe
                ])
                mixing_instructions = f"""Mixing Recipe for Color #{i+1}
Target: RGB{rgb}

Mix: {recipe_str}

Steps:
1. Start with the dominant color (highest %)
2. Gradually add smaller portions
3. Mix thoroughly between additions
4. Test on paper before applying to canvas
"""

                mixed_colors.append(MixedColor(
                    color_number=i + 1,
                    target_rgb=rgb,
                    recipe_paints=[(name, round(ratio * 100)) for name, ratio in analysis.recipe],
                    mixing_instructions=mixing_instructions,
                    result_rgb=analysis.best_match_rgb
                ))

            if analysis.warning:
                color_info["warning"] = analysis.warning

            colors_info.append(color_info)

        # Create shopping list with minimum display threshold
        shopping_list = []
        for paint_key, usage in paint_usage.items():
            if "paint" not in usage:
                continue
            paint = usage["paint"]
            oz_needed = usage["oz"] * self.SAFETY_MARGIN

            # Fix: ensure minimum display of 0.1 oz if paint is used
            oz_display = max(0.1, round(oz_needed, 1))
            tubes_needed = max(1, int(np.ceil(oz_needed / 2)))

            shopping_list.append(PaintQuantity(
                paint_key=paint_key,
                paint_name=paint.name,
                brand=paint.brand.value,
                oz_needed=oz_display,
                tubes_2oz=tubes_needed,
                price_per_tube=paint.price_2oz,
                total_price=round(tubes_needed * paint.price_2oz, 2),
                color_numbers=usage["colors"]
            ))

        shopping_list.sort(key=lambda x: x.total_price, reverse=True)
        total_cost = sum(item.total_price for item in shopping_list)

        # Budget analysis info
        budget_info = {
            "budget_requested": self.budget,
            "tier": self.budget_analysis.tier.name,
            "tier_description": self.budget_analysis.tier.description,
            "paints_in_set": len(paint_keys),
            "average_accuracy": round(self.budget_analysis.average_accuracy, 0),
            "colors_achievable": self.budget_analysis.colors_achievable,
            "colors_total": self.budget_analysis.colors_analyzed,
            "max_mixing_complexity": self.budget_analysis.max_mixing_complexity.name,
            "warnings": self.budget_analysis.warnings[:5],
            "recommendations": self.budget_analysis.recommendations,
        }

        self.kit = PaintKit(
            image_name=os.path.basename(self.image_path),
            canvas_size=self.canvas,
            n_colors=self.n_colors,
            shopping_list=shopping_list,
            mixed_colors=mixed_colors,
            total_cost=round(total_cost, 2),
            colors_info=colors_info,
            budget_info=budget_info
        )

        return self.kit

    def export_kit_json(self, output_path: str):
        """Export kit to JSON file."""
        if not self.kit:
            self.generate_kit()

        export_data = {
            "image": self.kit.image_name,
            "canvas_size": {
                "name": self.kit.canvas_size.name,
                "width_inches": self.kit.canvas_size.width_inches,
                "height_inches": self.kit.canvas_size.height_inches,
            },
            "n_colors": self.kit.n_colors,
            "total_cost_usd": self.kit.total_cost,
            "shopping_list": [
                {
                    "paint": item.paint_name,
                    "brand": item.brand,
                    "quantity_oz": item.oz_needed,
                    "tubes_2oz": item.tubes_2oz,
                    "price_per_tube": item.price_per_tube,
                    "total_price": item.total_price,
                    "used_for_colors": item.color_numbers
                }
                for item in self.kit.shopping_list
            ],
            "colors": self.kit.colors_info,
            "mixing_required": [
                {
                    "color_number": mc.color_number,
                    "target_rgb": mc.target_rgb,
                    "recipe": mc.recipe_paints,
                    "instructions": mc.mixing_instructions
                }
                for mc in self.kit.mixed_colors
            ]
        }

        # Add budget info if available
        if self.kit.budget_info:
            export_data["budget_analysis"] = self.kit.budget_info

        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)

        print(f"Kit exported to {output_path}")

    def create_shopping_list_image(self, output_path: str):
        """Create a visual shopping list."""
        if not self.kit:
            self.generate_kit()

        width = 850
        row_height = 35
        header_height = 150

        # Add extra height if budget info present
        budget_height = 80 if self.kit.budget_info else 0
        height = header_height + len(self.kit.shopping_list) * row_height + 200 + budget_height

        img = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(img)

        try:
            font_title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
            font_header = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
            font_body = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
        except:
            font_title = font_header = font_body = ImageFont.load_default()

        y = 20

        # Title
        draw.text((20, y), "PAINT SHOPPING LIST", fill='black', font=font_title)
        y += 35

        # Info
        draw.text((20, y), f"Image: {self.kit.image_name}", fill='gray', font=font_body)
        y += 20
        draw.text((20, y), f"Canvas: {self.kit.canvas_size.name}", fill='gray', font=font_body)
        y += 20
        draw.text((20, y), f"Colors: {self.kit.n_colors}", fill='gray', font=font_body)
        y += 30

        # Budget info box if available
        if self.kit.budget_info:
            bi = self.kit.budget_info
            draw.rectangle([20, y, width - 20, y + 65], fill='#e3f2fd', outline='#1976d2')
            draw.text((30, y + 5), f"BUDGET MODE: {bi['tier']} Tier", fill='#1565c0', font=font_header)
            draw.text((30, y + 25), f"Budget: ${bi['budget_requested']:.0f} | "
                     f"Accuracy: {bi['average_accuracy']:.0f}% | "
                     f"Paints: {bi['paints_in_set']} | "
                     f"Mixing: {bi['max_mixing_complexity']}",
                     fill='#424242', font=font_body)
            draw.text((30, y + 45), bi['tier_description'][:80], fill='#666666', font=font_body)
            y += 75

        # Total cost highlight
        draw.rectangle([20, y, 250, y + 40], fill='#e8f5e9', outline='#4caf50')
        draw.text((30, y + 10), f"TOTAL ESTIMATED COST: ${self.kit.total_cost:.2f}",
                 fill='#2e7d32', font=font_header)
        y += 60

        # Table header
        draw.rectangle([20, y, width - 20, y + 30], fill='#f5f5f5')
        cols = [20, 200, 350, 450, 550, 650, 750]
        headers = ["Paint Name", "Brand", "Qty (oz)", "Tubes", "$/Tube", "Total", "Colors"]

        for col, header in zip(cols, headers):
            draw.text((col + 5, y + 8), header, fill='black', font=font_header)
        y += 35

        # Table rows
        for i, item in enumerate(self.kit.shopping_list):
            bg_color = '#ffffff' if i % 2 == 0 else '#fafafa'
            draw.rectangle([20, y, width - 20, y + row_height], fill=bg_color)

            draw.text((cols[0] + 5, y + 8), item.paint_name[:25], fill='black', font=font_body)
            draw.text((cols[1] + 5, y + 8), item.brand[:15], fill='gray', font=font_body)
            draw.text((cols[2] + 5, y + 8), f"{item.oz_needed:.1f}", fill='black', font=font_body)
            draw.text((cols[3] + 5, y + 8), str(item.tubes_2oz), fill='black', font=font_body)
            draw.text((cols[4] + 5, y + 8), f"${item.price_per_tube:.2f}", fill='gray', font=font_body)
            draw.text((cols[5] + 5, y + 8), f"${item.total_price:.2f}", fill='#2e7d32', font=font_body)
            draw.text((cols[6] + 5, y + 8), ",".join(map(str, item.color_numbers[:5])), fill='gray', font=font_body)

            y += row_height

        # Notes
        y += 20
        draw.text((20, y), "NOTES:", fill='black', font=font_header)
        y += 22
        notes = [
            "• Quantities include 30% safety margin for mixing and corrections",
            "• Prices are estimates and may vary by retailer",
            "• Consider buying a starter set if you need many colors",
            f"• {len(self.kit.mixed_colors)} colors require mixing (see mixing guide)"
        ]
        for note in notes:
            draw.text((25, y), note, fill='gray', font=font_body)
            y += 18

        img.save(output_path)
        print(f"Shopping list saved to {output_path}")

    def create_mixing_guide_image(self, output_path: str):
        """Create a visual mixing guide for colors that need mixing."""
        if not self.kit:
            self.generate_kit()

        if not self.kit.mixed_colors:
            # No mixing required
            return

        width = 800
        color_height = 180
        height = 100 + len(self.kit.mixed_colors) * color_height

        img = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(img)

        try:
            font_title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
            font_header = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
            font_body = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 11)
        except:
            font_title = font_header = font_body = ImageFont.load_default()

        y = 20

        # Title
        draw.text((20, y), "COLOR MIXING GUIDE", fill='black', font=font_title)
        y += 35
        draw.text((20, y), f"{len(self.kit.mixed_colors)} colors require mixing",
                 fill='gray', font=font_body)
        y += 40

        for mc in self.kit.mixed_colors:
            # Color box
            draw.rectangle([20, y, width - 20, y + color_height - 20],
                          outline='#ddd', fill='#fafafa')

            # Target color swatch
            draw.rectangle([30, y + 10, 80, y + 60], fill=mc.target_rgb, outline='black')
            draw.text((30, y + 65), f"Color #{mc.color_number}", fill='black', font=font_body)
            draw.text((30, y + 80), f"RGB{mc.target_rgb}", fill='gray', font=font_body)

            # Arrow
            draw.text((95, y + 30), "=", fill='black', font=font_title)

            # Recipe components
            x = 130
            for paint_name, percent in mc.recipe_paints:
                # Paint swatch (we'd need to look up the color)
                draw.rectangle([x, y + 10, x + 50, y + 40], outline='gray', fill='#eee')
                draw.text((x, y + 45), f"{percent}%", fill='black', font=font_body)
                paint_short = paint_name[:12] + "..." if len(paint_name) > 12 else paint_name
                draw.text((x, y + 60), paint_short, fill='black', font=font_body)
                x += 70

                if x < 400:
                    draw.text((x - 15, y + 20), "+", fill='black', font=font_header)

            # Result swatch
            draw.text((x + 20, y + 20), "→", fill='black', font=font_title)
            draw.rectangle([x + 50, y + 10, x + 100, y + 60], fill=mc.result_rgb, outline='black')
            draw.text((x + 50, y + 65), "Result", fill='gray', font=font_body)

            # Mixing instructions (abbreviated)
            instructions_short = mc.mixing_instructions.split('\n')[2:5]  # Skip header
            inst_y = y + 95
            for inst in instructions_short:
                if inst.strip():
                    draw.text((30, inst_y), inst.strip()[:80], fill='#666', font=font_body)
                    inst_y += 14

            y += color_height

        img.save(output_path)
        print(f"Mixing guide saved to {output_path}")

    def create_color_chart_image(self, output_path: str):
        """Create a color chart showing all colors with their paint matches."""
        if not self.kit:
            self.generate_kit()

        n_colors = self.kit.n_colors
        cols = 4
        rows = (n_colors + cols - 1) // cols

        cell_width = 200
        cell_height = 100
        width = cols * cell_width + 40
        height = rows * cell_height + 100

        img = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(img)

        try:
            font_title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
            font_body = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 11)
        except:
            font_title = font_body = ImageFont.load_default()

        # Title
        draw.text((20, 15), "COLOR CHART & PAINT GUIDE", fill='black', font=font_title)
        draw.text((20, 40), f"Canvas: {self.kit.canvas_size.name} | {n_colors} colors",
                 fill='gray', font=font_body)

        y_start = 70

        for i, color_info in enumerate(self.kit.colors_info):
            col = i % cols
            row = i // cols

            x = 20 + col * cell_width
            y = y_start + row * cell_height

            # Color swatch
            rgb = tuple(color_info['rgb'])
            draw.rectangle([x, y, x + 50, y + 50], fill=rgb, outline='black')

            # Color number
            draw.text((x + 60, y), f"#{color_info['number']}", fill='black', font=font_title)

            # Coverage
            draw.text((x + 60, y + 22), f"{color_info['coverage_percent']}% coverage",
                     fill='gray', font=font_body)

            # Paint info
            if color_info.get('paint'):
                draw.text((x + 60, y + 38), color_info['paint'][:20],
                         fill='#1976d2', font=font_body)
            elif color_info.get('mixing_required'):
                draw.text((x + 60, y + 38), "MIX REQUIRED",
                         fill='#f57c00', font=font_body)

            # Hex code
            draw.text((x, y + 55), color_info['hex'], fill='gray', font=font_body)

        img.save(output_path)
        print(f"Color chart saved to {output_path}")

    def generate_all_outputs(self, output_dir: str):
        """Generate all kit outputs to a directory."""
        os.makedirs(output_dir, exist_ok=True)

        if not self.kit:
            self.generate_kit()

        print(f"\nGenerating paint kit to {output_dir}/")
        print("-" * 50)

        # JSON export
        self.export_kit_json(os.path.join(output_dir, "paint_kit.json"))

        # Shopping list
        self.create_shopping_list_image(os.path.join(output_dir, "shopping_list.png"))

        # Color chart
        self.create_color_chart_image(os.path.join(output_dir, "color_chart.png"))

        # Mixing guide (if needed)
        if self.kit.mixed_colors:
            self.create_mixing_guide_image(os.path.join(output_dir, "mixing_guide.png"))

        print("-" * 50)
        print(f"Total paints to buy: {len(self.kit.shopping_list)}")
        print(f"Colors requiring mixing: {len(self.kit.mixed_colors)}")
        print(f"Estimated total cost: ${self.kit.total_cost:.2f}")

        return self.kit


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Paint Kit Generator")
        print("Usage: python paint_kit_generator.py <image_path> [canvas_size] [n_colors]")
        print("\nCanvas sizes: 8x10, 9x12, 11x14, 12x16, 16x20, 18x24, 24x30, 24x36")
        print("\nExample:")
        print("  python paint_kit_generator.py input/aurora/aurora.png 16x20 15")
        sys.exit(0)

    image_path = sys.argv[1]
    canvas_size = sys.argv[2] if len(sys.argv) > 2 else "16x20"
    n_colors = int(sys.argv[3]) if len(sys.argv) > 3 else 15

    generator = PaintKitGenerator(image_path, n_colors, canvas_size)
    kit = generator.generate_kit()

    output_dir = "paint_kit_output"
    generator.generate_all_outputs(output_dir)
