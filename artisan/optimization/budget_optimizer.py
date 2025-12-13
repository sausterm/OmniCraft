"""
Budget Optimizer
================

Dynamic paint kit generation based on budget constraints.

This module optimizes paint selection to balance cost, color accuracy,
and mixing complexity. Given a target budget, it determines:

1. Optimal paint selection from predefined sets
2. Color achievability (what % of palette colors are reachable)
3. Required mixing complexity for each color
4. Quality trade-offs at different price points

Budget Tiers:
    - Minimal ($40-55): 5 paints, heavy mixing, ~85% accuracy
    - Budget ($55-80): 6-7 paints, moderate mixing, ~90% accuracy
    - Standard ($80-120): 8-10 paints, light mixing, ~95% accuracy
    - Premium ($120-180): 12-14 paints, minimal mixing, ~98% accuracy
    - Professional ($180+): 20+ paints, near-perfect matching

Example Usage:
    >>> from budget_optimizer import BudgetOptimizer
    >>>
    >>> # Analyze palette at different budgets
    >>> palette = [(255, 0, 0), (0, 255, 0), (0, 0, 255), ...]
    >>> optimizer = BudgetOptimizer(palette)
    >>>
    >>> # Get analysis for $75 budget
    >>> analysis = optimizer.analyze_budget(75)
    >>> print(f"Accuracy: {analysis.average_accuracy}%")
    >>> print(f"Paints needed: {len(analysis.paint_set)}")
    >>>
    >>> # Find minimum budget for 90% accuracy
    >>> min_budget = optimizer.find_minimum_budget(90)
    >>> print(f"Minimum budget: ${min_budget}")

Key Classes:
    - BudgetOptimizer: Main optimization engine
    - BudgetAnalysis: Complete analysis for a budget level
    - ColorAchievability: How achievable a specific color is
    - BudgetTier: Predefined budget tier definitions

Paint Sets:
    The module defines optimal paint sets for different sizes:
    - 5 paints: CMY + White + Black (basic primaries)
    - 6 paints: + Burnt Sienna (earth tone)
    - 7 paints: + Phthalo Green (vibrant green)
    - 8 paints: + Cadmium Red (true red)
    - 10 paints: + Ultramarine + Yellow Ochre
    - 12 paints: + More variety for nuanced colors
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict
import json

from ..core.paint_database import (
    Paint, PAINT_DATABASE, ESSENTIAL_MIXING_SET,
    get_paint, color_distance_lab, rgb_to_lab
)
from ..core.color_matcher import ColorMatcher, ColorSolution


class MixingComplexity(Enum):
    """Mixing complexity levels."""
    NONE = 0        # Direct paint match
    SIMPLE = 1      # 2 paints, easy ratio (50/50, 70/30)
    MODERATE = 2    # 2-3 paints, precise ratios
    COMPLEX = 3     # 3+ paints, very precise ratios
    EXTREME = 4     # 4+ paints, expert level


@dataclass
class BudgetTier:
    """Predefined budget tiers with recommendations."""
    name: str
    paint_budget_min: float
    paint_budget_max: float
    recommended_paints: int
    expected_quality: str  # "fair", "good", "very_good", "excellent"
    mixing_complexity: MixingComplexity
    description: str


# Budget tiers
BUDGET_TIERS = [
    BudgetTier(
        name="Minimal",
        paint_budget_min=40,
        paint_budget_max=55,
        recommended_paints=5,
        expected_quality="fair",
        mixing_complexity=MixingComplexity.EXTREME,
        description="Basic primaries only. Heavy mixing required. Some colors unachievable."
    ),
    BudgetTier(
        name="Budget",
        paint_budget_min=55,
        paint_budget_max=80,
        recommended_paints=7,
        expected_quality="good",
        mixing_complexity=MixingComplexity.COMPLEX,
        description="Extended primaries. Most colors achievable with mixing."
    ),
    BudgetTier(
        name="Standard",
        paint_budget_min=80,
        paint_budget_max=120,
        recommended_paints=10,
        expected_quality="very_good",
        mixing_complexity=MixingComplexity.MODERATE,
        description="Good color range. Moderate mixing needed."
    ),
    BudgetTier(
        name="Premium",
        paint_budget_min=120,
        paint_budget_max=180,
        recommended_paints=14,
        expected_quality="excellent",
        mixing_complexity=MixingComplexity.SIMPLE,
        description="Wide color range. Minimal mixing."
    ),
    BudgetTier(
        name="Professional",
        paint_budget_min=180,
        paint_budget_max=999,
        recommended_paints=20,
        expected_quality="professional",
        mixing_complexity=MixingComplexity.NONE,
        description="Full palette. Near-perfect color matching."
    ),
]


# Minimum paint sets by tier (ordered by importance)
PAINT_SETS = {
    5: [  # Minimal - CMY + W + B
        "titanium_white",
        "mars_black",
        "cadmium_yellow_medium",
        "quinacridone_magenta",  # Better than cadmium red for mixing
        "phthalo_blue_gs",
    ],
    6: [  # + Earth tone
        "titanium_white",
        "mars_black",
        "cadmium_yellow_medium",
        "quinacridone_magenta",
        "phthalo_blue_gs",
        "burnt_sienna",
    ],
    7: [  # + Green (can't mix vibrant greens easily)
        "titanium_white",
        "mars_black",
        "cadmium_yellow_medium",
        "quinacridone_magenta",
        "phthalo_blue_gs",
        "phthalo_green_bs",
        "burnt_sienna",
    ],
    8: [  # + True red
        "titanium_white",
        "mars_black",
        "cadmium_yellow_medium",
        "cadmium_red_medium",
        "quinacridone_magenta",
        "phthalo_blue_gs",
        "phthalo_green_bs",
        "burnt_sienna",
    ],
    10: [  # + Violet + Yellow ochre
        "titanium_white",
        "mars_black",
        "cadmium_yellow_medium",
        "cadmium_red_medium",
        "quinacridone_magenta",
        "phthalo_blue_gs",
        "ultramarine_blue",
        "phthalo_green_bs",
        "burnt_sienna",
        "yellow_ochre",
    ],
    12: [  # + More variety
        "titanium_white",
        "mars_black",
        "cadmium_yellow_medium",
        "cadmium_yellow_light",
        "cadmium_red_medium",
        "alizarin_crimson",
        "quinacridone_magenta",
        "phthalo_blue_gs",
        "ultramarine_blue",
        "phthalo_green_bs",
        "sap_green",
        "burnt_sienna",
    ],
}


@dataclass
class ColorAchievability:
    """How achievable a target color is with a given paint set."""
    target_rgb: Tuple[int, int, int]
    achievable: bool
    best_match_rgb: Tuple[int, int, int]
    color_distance: float
    accuracy_percent: float  # 100 = perfect, 0 = impossible
    mixing_complexity: MixingComplexity
    recipe: List[Tuple[str, float]]  # (paint_name, ratio)
    warning: Optional[str]


@dataclass
class BudgetAnalysis:
    """Complete analysis for a budget level."""
    budget: float
    tier: BudgetTier
    paint_set: List[str]
    paint_cost: float
    colors_analyzed: int
    colors_achievable: int
    achievability_percent: float
    average_accuracy: float
    max_mixing_complexity: MixingComplexity
    color_details: List[ColorAchievability]
    warnings: List[str]
    recommendations: List[str]


class BudgetOptimizer:
    """Optimizes paint selection based on budget constraints."""

    # Color accuracy thresholds
    PERFECT_MATCH = 10      # Color distance for "perfect"
    GOOD_MATCH = 25         # "Good" match
    ACCEPTABLE_MATCH = 40   # "Acceptable" match
    POOR_MATCH = 60         # "Poor" but usable
    UNUSABLE = 100          # Can't reasonably achieve

    def __init__(self, palette: List[Tuple[int, int, int]],
                 pixel_counts: List[int] = None):
        """
        Initialize optimizer with target palette.

        Args:
            palette: List of target RGB colors
            pixel_counts: Optional pixel counts for coverage weighting
        """
        self.palette = palette
        self.pixel_counts = pixel_counts or [1] * len(palette)
        self.total_pixels = sum(self.pixel_counts)

    def get_tier_for_budget(self, budget: float) -> BudgetTier:
        """Get the appropriate tier for a budget."""
        for tier in BUDGET_TIERS:
            if tier.paint_budget_min <= budget <= tier.paint_budget_max:
                return tier
        # Default to highest or lowest
        if budget < BUDGET_TIERS[0].paint_budget_min:
            return BUDGET_TIERS[0]
        return BUDGET_TIERS[-1]

    def get_optimal_paint_set(self, budget: float) -> Tuple[List[str], float]:
        """
        Get the optimal paint set for a budget.

        Returns:
            (list of paint keys, total cost)
        """
        # Find largest set that fits budget
        best_set = PAINT_SETS[5]  # Minimum
        best_cost = self._calculate_set_cost(best_set)

        for n_paints in sorted(PAINT_SETS.keys()):
            paint_set = PAINT_SETS[n_paints]
            cost = self._calculate_set_cost(paint_set)
            if cost <= budget:
                best_set = paint_set
                best_cost = cost
            else:
                break

        return best_set, best_cost

    def _calculate_set_cost(self, paint_keys: List[str]) -> float:
        """Calculate total cost of a paint set."""
        total = 0
        for key in paint_keys:
            paint = get_paint(key)
            if paint:
                total += paint.price_2oz
        return total

    def analyze_achievability(self, paint_keys: List[str],
                             target_rgb: Tuple[int, int, int]) -> ColorAchievability:
        """
        Analyze how achievable a target color is with given paints.
        """
        available_paints = {k: get_paint(k) for k in paint_keys if get_paint(k)}

        # Find best single paint match
        best_single_distance = float('inf')
        best_single_key = None

        for key, paint in available_paints.items():
            dist = color_distance_lab(target_rgb, paint.rgb)
            if dist < best_single_distance:
                best_single_distance = dist
                best_single_key = key

        # Find best 2-paint mix
        best_mix_distance = float('inf')
        best_mix_rgb = None
        best_mix_recipe = []

        paint_items = list(available_paints.items())
        for i, (key1, paint1) in enumerate(paint_items):
            for key2, paint2 in paint_items[i+1:]:
                for ratio in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
                    mixed = self._mix_colors(paint1.rgb, paint2.rgb, ratio)
                    dist = color_distance_lab(target_rgb, mixed)
                    if dist < best_mix_distance:
                        best_mix_distance = dist
                        best_mix_rgb = mixed
                        best_mix_recipe = [(paint1.name, ratio), (paint2.name, 1-ratio)]

        # Find best 3-paint mix (with white or black modifier)
        for key1, paint1 in paint_items:
            for modifier_key in ['titanium_white', 'mars_black']:
                if modifier_key not in available_paints:
                    continue
                modifier = available_paints[modifier_key]
                for ratio1 in [0.5, 0.6, 0.7]:
                    for ratio_mod in [0.1, 0.2, 0.3]:
                        ratio2 = 1 - ratio1 - ratio_mod
                        if ratio2 <= 0:
                            continue
                        # Mix paint1 with modifier, then imagine base
                        mixed = self._mix_three(paint1.rgb, modifier.rgb,
                                               paint1.rgb, ratio1, ratio_mod, ratio2)
                        dist = color_distance_lab(target_rgb, mixed)
                        if dist < best_mix_distance:
                            best_mix_distance = dist
                            best_mix_rgb = mixed
                            best_mix_recipe = [
                                (paint1.name, ratio1 + ratio2),
                                (modifier.name, ratio_mod)
                            ]

        # Determine best approach
        if best_single_distance <= self.PERFECT_MATCH:
            # Single paint is perfect
            paint = available_paints[best_single_key]
            return ColorAchievability(
                target_rgb=target_rgb,
                achievable=True,
                best_match_rgb=paint.rgb,
                color_distance=best_single_distance,
                accuracy_percent=self._distance_to_accuracy(best_single_distance),
                mixing_complexity=MixingComplexity.NONE,
                recipe=[(paint.name, 1.0)],
                warning=None
            )

        if best_mix_distance < best_single_distance:
            # Mix is better
            complexity = MixingComplexity.SIMPLE if len(best_mix_recipe) == 2 else MixingComplexity.MODERATE
            achievable = best_mix_distance < self.UNUSABLE

            warning = None
            if best_mix_distance > self.ACCEPTABLE_MATCH:
                warning = f"Color will be approximate (accuracy: {self._distance_to_accuracy(best_mix_distance):.0f}%)"

            return ColorAchievability(
                target_rgb=target_rgb,
                achievable=achievable,
                best_match_rgb=best_mix_rgb,
                color_distance=best_mix_distance,
                accuracy_percent=self._distance_to_accuracy(best_mix_distance),
                mixing_complexity=complexity,
                recipe=best_mix_recipe,
                warning=warning
            )

        # Single paint is best (but may not be great)
        paint = available_paints[best_single_key]
        achievable = best_single_distance < self.UNUSABLE

        warning = None
        if best_single_distance > self.ACCEPTABLE_MATCH:
            warning = f"Best available match. Color will differ from original."

        return ColorAchievability(
            target_rgb=target_rgb,
            achievable=achievable,
            best_match_rgb=paint.rgb,
            color_distance=best_single_distance,
            accuracy_percent=self._distance_to_accuracy(best_single_distance),
            mixing_complexity=MixingComplexity.NONE,
            recipe=[(paint.name, 1.0)],
            warning=warning
        )

    def _mix_colors(self, rgb1: Tuple[int, int, int],
                   rgb2: Tuple[int, int, int], ratio: float) -> Tuple[int, int, int]:
        """Mix two colors."""
        return (
            int(rgb1[0] * ratio + rgb2[0] * (1-ratio)),
            int(rgb1[1] * ratio + rgb2[1] * (1-ratio)),
            int(rgb1[2] * ratio + rgb2[2] * (1-ratio))
        )

    def _mix_three(self, rgb1, rgb2, rgb3, r1, r2, r3) -> Tuple[int, int, int]:
        """Mix three colors."""
        return (
            int(rgb1[0] * r1 + rgb2[0] * r2 + rgb3[0] * r3),
            int(rgb1[1] * r1 + rgb2[1] * r2 + rgb3[1] * r3),
            int(rgb1[2] * r1 + rgb2[2] * r2 + rgb3[2] * r3)
        )

    def _distance_to_accuracy(self, distance: float) -> float:
        """Convert color distance to accuracy percentage."""
        if distance < self.PERFECT_MATCH:
            return 100.0
        elif distance < self.GOOD_MATCH:
            return 95 - (distance - self.PERFECT_MATCH) / (self.GOOD_MATCH - self.PERFECT_MATCH) * 10
        elif distance < self.ACCEPTABLE_MATCH:
            return 85 - (distance - self.GOOD_MATCH) / (self.ACCEPTABLE_MATCH - self.GOOD_MATCH) * 15
        elif distance < self.POOR_MATCH:
            return 70 - (distance - self.ACCEPTABLE_MATCH) / (self.POOR_MATCH - self.ACCEPTABLE_MATCH) * 20
        elif distance < self.UNUSABLE:
            return 50 - (distance - self.POOR_MATCH) / (self.UNUSABLE - self.POOR_MATCH) * 30
        else:
            return max(0, 20 - (distance - self.UNUSABLE) / 50 * 20)

    def analyze_budget(self, budget: float) -> BudgetAnalysis:
        """
        Complete analysis for a given budget.
        """
        tier = self.get_tier_for_budget(budget)
        paint_keys, paint_cost = self.get_optimal_paint_set(budget)

        # Analyze each color
        color_details = []
        achievable_count = 0
        total_accuracy = 0
        max_complexity = MixingComplexity.NONE
        warnings = []

        for i, (rgb, pixel_count) in enumerate(zip(self.palette, self.pixel_counts)):
            analysis = self.analyze_achievability(paint_keys, rgb)
            color_details.append(analysis)

            if analysis.achievable:
                achievable_count += 1

            # Weight accuracy by coverage
            weight = pixel_count / self.total_pixels
            total_accuracy += analysis.accuracy_percent * weight

            if analysis.mixing_complexity.value > max_complexity.value:
                max_complexity = analysis.mixing_complexity

            if analysis.warning:
                warnings.append(f"Color {i+1}: {analysis.warning}")

        # Generate recommendations
        recommendations = []
        achievability_pct = achievable_count / len(self.palette) * 100

        if achievability_pct < 80:
            recommendations.append("Consider increasing budget for better color coverage")
        if max_complexity.value >= MixingComplexity.COMPLEX.value:
            recommendations.append("Many colors require complex mixing - allow extra time")
        if paint_cost < budget * 0.8:
            recommendations.append(f"Budget allows for {int((budget - paint_cost) / 10)} additional paint tubes")

        return BudgetAnalysis(
            budget=budget,
            tier=tier,
            paint_set=[get_paint(k).name for k in paint_keys],
            paint_cost=paint_cost,
            colors_analyzed=len(self.palette),
            colors_achievable=achievable_count,
            achievability_percent=achievability_pct,
            average_accuracy=total_accuracy,
            max_mixing_complexity=max_complexity,
            color_details=color_details,
            warnings=warnings,
            recommendations=recommendations
        )

    def find_minimum_budget(self, target_accuracy: float = 85.0) -> float:
        """
        Find the minimum budget to achieve target accuracy.
        """
        for budget in range(40, 250, 10):
            analysis = self.analyze_budget(budget)
            if analysis.average_accuracy >= target_accuracy:
                return budget
        return 250  # Max

    def compare_budgets(self, budgets: List[float] = None) -> List[BudgetAnalysis]:
        """
        Compare multiple budget levels.
        """
        if budgets is None:
            budgets = [50, 75, 100, 150]

        return [self.analyze_budget(b) for b in budgets]


def print_budget_comparison(palette: List[Tuple[int, int, int]],
                           budgets: List[float] = None):
    """Print a budget comparison table."""
    optimizer = BudgetOptimizer(palette)

    if budgets is None:
        budgets = [50, 75, 100, 150, 200]

    print("\n" + "=" * 70)
    print("BUDGET COMPARISON")
    print("=" * 70)
    print(f"{'Budget':<10} {'Paints':<8} {'Accuracy':<10} {'Achievable':<12} {'Complexity':<12}")
    print("-" * 70)

    for budget in budgets:
        analysis = optimizer.analyze_budget(budget)
        print(f"${budget:<9} {len(analysis.paint_set):<8} "
              f"{analysis.average_accuracy:.0f}%{'':<6} "
              f"{analysis.achievability_percent:.0f}%{'':<8} "
              f"{analysis.max_mixing_complexity.name:<12}")

    print("-" * 70)

    # Find minimum budget
    min_budget = optimizer.find_minimum_budget(85)
    print(f"\nMinimum budget for 85% accuracy: ${min_budget}")


if __name__ == "__main__":
    # Test with sample aurora palette
    test_palette = [
        (0, 2, 4),       # Near black
        (47, 44, 33),    # Dark gray
        (95, 8, 54),     # Dark magenta
        (58, 2, 30),     # Dark red
        (124, 183, 97),  # Bright green
        (78, 175, 88),   # Green
        (136, 140, 100), # Olive
        (144, 193, 119), # Light green
        (145, 64, 92),   # Rose
        (185, 160, 135), # Tan
        (217, 203, 182), # Cream
        (56, 88, 93),    # Teal
        (108, 142, 125), # Sage
        (174, 127, 133), # Dusty rose
        (190, 178, 163), # Warm gray
    ]

    print("Budget Optimizer Test")
    print_budget_comparison(test_palette)

    # Detailed analysis for $75
    print("\n" + "=" * 70)
    print("DETAILED ANALYSIS: $75 Budget")
    print("=" * 70)

    optimizer = BudgetOptimizer(test_palette)
    analysis = optimizer.analyze_budget(75)

    print(f"Tier: {analysis.tier.name}")
    print(f"Paints: {len(analysis.paint_set)}")
    print(f"Cost: ${analysis.paint_cost:.2f}")
    print(f"Average accuracy: {analysis.average_accuracy:.0f}%")
    print(f"Colors achievable: {analysis.colors_achievable}/{analysis.colors_analyzed}")
    print(f"Max complexity: {analysis.max_mixing_complexity.name}")

    print("\nPaint Set:")
    for paint in analysis.paint_set:
        print(f"  - {paint}")

    if analysis.warnings:
        print("\nWarnings:")
        for w in analysis.warnings[:5]:
            print(f"  ! {w}")

    if analysis.recommendations:
        print("\nRecommendations:")
        for r in analysis.recommendations:
            print(f"  * {r}")
