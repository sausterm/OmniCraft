"""
Color Matcher - Match colors to paints or mixing recipes

Features:
- Find closest single paint match
- Generate 2-3 paint mixing recipes
- Calculate mixing ratios
- Estimate quantities needed
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from .paint_database import (
    Paint, PAINT_DATABASE, ESSENTIAL_MIXING_SET,
    get_paint, color_distance_lab, rgb_to_lab
)


@dataclass
class PaintMatch:
    """A single paint that matches a target color."""
    paint_key: str
    paint: Paint
    distance: float  # Color distance (lower = better)
    confidence: float  # 0-1, how good the match is


@dataclass
class MixingRecipe:
    """A recipe for mixing paints to achieve a target color."""
    paints: List[Tuple[str, Paint, float]]  # (key, paint, ratio)
    result_rgb: Tuple[int, int, int]
    distance: float
    confidence: float
    mixing_instructions: str


@dataclass
class ColorSolution:
    """Complete solution for achieving a target color."""
    target_rgb: Tuple[int, int, int]
    target_name: str
    best_single: Optional[PaintMatch]
    best_mix: Optional[MixingRecipe]
    recommended: str  # "single" or "mix"
    recommendation_reason: str


class ColorMatcher:
    """Matches target colors to paints or mixing recipes."""

    # Maximum acceptable color distance for a "good" match
    GOOD_MATCH_THRESHOLD = 15.0
    ACCEPTABLE_MATCH_THRESHOLD = 30.0

    def __init__(self, use_essential_set: bool = False):
        """
        Initialize color matcher.

        Args:
            use_essential_set: If True, only use essential mixing colors
        """
        if use_essential_set:
            self.available_paints = {k: get_paint(k) for k in ESSENTIAL_MIXING_SET}
        else:
            self.available_paints = PAINT_DATABASE.copy()

    def find_closest_paint(self, target_rgb: Tuple[int, int, int]) -> PaintMatch:
        """Find the single closest paint to a target color."""
        best_match = None
        best_distance = float('inf')

        for key, paint in self.available_paints.items():
            distance = color_distance_lab(target_rgb, paint.rgb)
            if distance < best_distance:
                best_distance = distance
                best_match = PaintMatch(
                    paint_key=key,
                    paint=paint,
                    distance=distance,
                    confidence=self._distance_to_confidence(distance)
                )

        return best_match

    def find_mixing_recipe(self, target_rgb: Tuple[int, int, int],
                           max_paints: int = 3) -> Optional[MixingRecipe]:
        """
        Find a mixing recipe to achieve the target color.

        Uses optimization to find the best combination of 2-3 paints.
        """
        target_lab = rgb_to_lab(target_rgb)
        best_recipe = None
        best_distance = float('inf')

        # Try 2-paint combinations
        paint_keys = list(self.available_paints.keys())

        for i, key1 in enumerate(paint_keys):
            paint1 = self.available_paints[key1]

            for key2 in paint_keys[i+1:]:
                paint2 = self.available_paints[key2]

                # Try different ratios
                for ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                    mixed_rgb = self._mix_colors(
                        [(paint1.rgb, ratio), (paint2.rgb, 1-ratio)]
                    )
                    distance = color_distance_lab(target_rgb, mixed_rgb)

                    if distance < best_distance:
                        best_distance = distance
                        best_recipe = MixingRecipe(
                            paints=[
                                (key1, paint1, ratio),
                                (key2, paint2, 1-ratio)
                            ],
                            result_rgb=mixed_rgb,
                            distance=distance,
                            confidence=self._distance_to_confidence(distance),
                            mixing_instructions=""
                        )

        # Try 3-paint combinations if 2-paint isn't good enough
        if best_distance > self.GOOD_MATCH_THRESHOLD and max_paints >= 3:
            # Add white or black to improve the match
            for key1 in paint_keys:
                paint1 = self.available_paints[key1]

                for key2 in ['titanium_white', 'mars_black']:
                    if key2 not in self.available_paints:
                        continue
                    paint2 = self.available_paints[key2]

                    for ratio1 in [0.3, 0.5, 0.7]:
                        for ratio2 in [0.1, 0.2, 0.3]:
                            ratio3 = 1 - ratio1 - ratio2
                            if ratio3 <= 0:
                                continue

                            # Find a third color
                            for key3 in paint_keys:
                                if key3 in [key1, key2]:
                                    continue
                                paint3 = self.available_paints[key3]

                                mixed_rgb = self._mix_colors([
                                    (paint1.rgb, ratio1),
                                    (paint2.rgb, ratio2),
                                    (paint3.rgb, ratio3)
                                ])
                                distance = color_distance_lab(target_rgb, mixed_rgb)

                                if distance < best_distance:
                                    best_distance = distance
                                    best_recipe = MixingRecipe(
                                        paints=[
                                            (key1, paint1, ratio1),
                                            (key3, paint3, ratio3),
                                            (key2, paint2, ratio2)
                                        ],
                                        result_rgb=mixed_rgb,
                                        distance=distance,
                                        confidence=self._distance_to_confidence(distance),
                                        mixing_instructions=""
                                    )

        if best_recipe:
            best_recipe.mixing_instructions = self._generate_mixing_instructions(best_recipe)

        return best_recipe

    def _mix_colors(self, colors_with_ratios: List[Tuple[Tuple[int, int, int], float]]) -> Tuple[int, int, int]:
        """
        Mix colors in given ratios.

        Uses a simplified mixing model (weighted average in RGB).
        Real paint mixing is more complex, but this gives a reasonable approximation.
        """
        total_r, total_g, total_b = 0.0, 0.0, 0.0

        for rgb, ratio in colors_with_ratios:
            total_r += rgb[0] * ratio
            total_g += rgb[1] * ratio
            total_b += rgb[2] * ratio

        return (
            int(min(255, max(0, total_r))),
            int(min(255, max(0, total_g))),
            int(min(255, max(0, total_b)))
        )

    def _distance_to_confidence(self, distance: float) -> float:
        """Convert color distance to confidence score (0-1)."""
        if distance < self.GOOD_MATCH_THRESHOLD:
            return 1.0 - (distance / self.GOOD_MATCH_THRESHOLD) * 0.2
        elif distance < self.ACCEPTABLE_MATCH_THRESHOLD:
            return 0.8 - ((distance - self.GOOD_MATCH_THRESHOLD) /
                         (self.ACCEPTABLE_MATCH_THRESHOLD - self.GOOD_MATCH_THRESHOLD)) * 0.3
        else:
            return max(0.1, 0.5 - (distance - self.ACCEPTABLE_MATCH_THRESHOLD) / 100)

    def _generate_mixing_instructions(self, recipe: MixingRecipe) -> str:
        """Generate human-readable mixing instructions."""
        # Sort by ratio (highest first)
        sorted_paints = sorted(recipe.paints, key=lambda x: x[2], reverse=True)

        instructions = []
        instructions.append("MIXING RECIPE:")

        # Convert ratios to parts (easier to measure)
        total_parts = 10
        parts = [(p[1].name, round(p[2] * total_parts)) for p in sorted_paints]

        for paint_name, paint_parts in parts:
            if paint_parts > 0:
                instructions.append(f"  {paint_parts} parts {paint_name}")

        instructions.append("")
        instructions.append("STEPS:")
        instructions.append(f"  1. Start with {parts[0][1]} parts of {parts[0][0]}")

        for i, (paint_name, paint_parts) in enumerate(parts[1:], 2):
            if paint_parts > 0:
                instructions.append(f"  {i}. Add {paint_parts} parts of {paint_name}, mix thoroughly")

        instructions.append(f"  {len(parts)+1}. Mix until uniform color is achieved")
        instructions.append("  Tip: Test on scrap paper before applying to canvas")

        return "\n".join(instructions)

    def solve_color(self, target_rgb: Tuple[int, int, int],
                    color_name: str = "") -> ColorSolution:
        """
        Find the best solution (single paint or mix) for a target color.
        """
        single_match = self.find_closest_paint(target_rgb)
        mix_recipe = self.find_mixing_recipe(target_rgb)

        # Decide which is better
        if single_match.distance < self.GOOD_MATCH_THRESHOLD:
            # Single paint is good enough
            recommended = "single"
            reason = f"Excellent single-paint match ({single_match.paint.name})"
        elif mix_recipe and mix_recipe.distance < single_match.distance - 5:
            # Mix is noticeably better
            recommended = "mix"
            reason = "Mixing provides a closer color match"
        elif single_match.distance < self.ACCEPTABLE_MATCH_THRESHOLD:
            # Single paint is acceptable
            recommended = "single"
            reason = f"Good single-paint match ({single_match.paint.name})"
        elif mix_recipe:
            recommended = "mix"
            reason = "Mixing required for better accuracy"
        else:
            recommended = "single"
            reason = "Closest available option"

        return ColorSolution(
            target_rgb=target_rgb,
            target_name=color_name,
            best_single=single_match,
            best_mix=mix_recipe,
            recommended=recommended,
            recommendation_reason=reason
        )


def match_palette(palette: List[Tuple[int, int, int]],
                  pixel_counts: List[int] = None) -> List[ColorSolution]:
    """
    Match an entire palette of colors.

    Args:
        palette: List of RGB tuples
        pixel_counts: Optional pixel counts for each color (for quantity calculation)

    Returns:
        List of ColorSolution objects
    """
    matcher = ColorMatcher()
    solutions = []

    for i, rgb in enumerate(palette):
        name = f"Color {i+1}"
        solution = matcher.solve_color(rgb, name)
        solutions.append(solution)

    return solutions


if __name__ == "__main__":
    # Test with some colors
    test_colors = [
        ((0, 0, 0), "Black"),
        ((255, 255, 255), "White"),
        ((227, 38, 54), "Red"),
        ((0, 100, 80), "Teal"),
        ((142, 36, 99), "Magenta"),
        ((95, 8, 54), "Dark Magenta"),  # Aurora color
        ((78, 175, 88), "Bright Green"),  # Aurora color
    ]

    print("Color Matching Test")
    print("=" * 60)

    matcher = ColorMatcher()

    for rgb, name in test_colors:
        solution = matcher.solve_color(rgb, name)
        print(f"\n{name} RGB{rgb}:")
        print(f"  Recommended: {solution.recommended}")
        print(f"  Reason: {solution.recommendation_reason}")

        if solution.best_single:
            print(f"  Single match: {solution.best_single.paint.name}")
            print(f"    Distance: {solution.best_single.distance:.1f}")
            print(f"    Confidence: {solution.best_single.confidence:.0%}")

        if solution.best_mix and solution.recommended == "mix":
            print(f"  Mix recipe:")
            for key, paint, ratio in solution.best_mix.paints:
                print(f"    {ratio:.0%} {paint.name}")
            print(f"    Result RGB: {solution.best_mix.result_rgb}")
            print(f"    Distance: {solution.best_mix.distance:.1f}")
