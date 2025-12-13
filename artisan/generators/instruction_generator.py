"""
Unified Instruction Generator

A spectrum of painting instruction styles from simple to detailed:

Level 1 - BEGINNER (Simple Paint-by-Numbers)
  - Numbered regions only
  - Color guide
  - "Fill region 1 with blue"
  - Output: Clean, graphic look

Level 2 - EASY (Guided Paint-by-Numbers)
  - Numbered regions + layer order
  - Basic technique hints
  - "Start with background colors, then foreground"
  - Output: Better color harmony

Level 3 - INTERMEDIATE (Technique-Aware)
  - Technique zones + brush suggestions
  - Blending guidance
  - "Use wet-on-wet for the sky gradient"
  - Output: Smoother, more painterly

Level 4 - ADVANCED (Bob Ross Style)
  - Full step-by-step instructions
  - Specific motions, timing, encouragement
  - "Load your fan brush with titanium white..."
  - Output: Professional-looking painting
"""

import os
import json
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from enum import Enum

from ..core.paint_by_numbers import PaintByNumbers
from ..analysis.technique_analyzer import TechniqueAnalyzer, Technique
from .bob_ross import BobRossGenerator


class InstructionLevel(Enum):
    """Instruction detail levels"""
    BEGINNER = 1      # Simple paint-by-numbers
    EASY = 2          # Guided with layer order
    INTERMEDIATE = 3  # Technique-aware
    ADVANCED = 4      # Bob Ross style


@dataclass
class LevelConfig:
    """Configuration for each instruction level"""
    name: str
    description: str
    show_numbers: bool
    show_layer_order: bool
    show_techniques: bool
    show_brush_info: bool
    show_motions: bool
    show_timing: bool
    show_encouragement: bool
    granularity: str  # "regions", "layers", "steps", "detailed_steps"
    expected_skill: str
    expected_output: str
    estimated_time_multiplier: float  # vs base time


LEVEL_CONFIGS = {
    InstructionLevel.BEGINNER: LevelConfig(
        name="Beginner",
        description="Simple Paint-by-Numbers",
        show_numbers=True,
        show_layer_order=False,
        show_techniques=False,
        show_brush_info=False,
        show_motions=False,
        show_timing=False,
        show_encouragement=False,
        granularity="regions",
        expected_skill="No experience needed",
        expected_output="Clean, graphic style with distinct color regions",
        estimated_time_multiplier=1.0
    ),
    InstructionLevel.EASY: LevelConfig(
        name="Easy",
        description="Guided Paint-by-Numbers",
        show_numbers=True,
        show_layer_order=True,
        show_techniques=False,
        show_brush_info=True,
        show_motions=False,
        show_timing=False,
        show_encouragement=False,
        granularity="layers",
        expected_skill="Basic brush control",
        expected_output="Better color harmony with proper layering",
        estimated_time_multiplier=1.3
    ),
    InstructionLevel.INTERMEDIATE: LevelConfig(
        name="Intermediate",
        description="Technique-Aware Painting",
        show_numbers=True,
        show_layer_order=True,
        show_techniques=True,
        show_brush_info=True,
        show_motions=True,
        show_timing=False,
        show_encouragement=False,
        granularity="steps",
        expected_skill="Some painting experience",
        expected_output="Smoother blends and painterly effects",
        estimated_time_multiplier=1.8
    ),
    InstructionLevel.ADVANCED: LevelConfig(
        name="Advanced",
        description="Bob Ross Style Master Class",
        show_numbers=False,  # No numbers, full technique
        show_layer_order=True,
        show_techniques=True,
        show_brush_info=True,
        show_motions=True,
        show_timing=True,
        show_encouragement=True,
        granularity="detailed_steps",
        expected_skill="Willing to learn techniques",
        expected_output="Professional-looking painting with proper techniques",
        estimated_time_multiplier=2.5
    )
}


class UnifiedInstructionGenerator:
    """
    Generates painting instructions at any detail level.
    """

    def __init__(self, image_path: str, n_colors: int = 15):
        self.image_path = image_path
        self.n_colors = n_colors

        # Load image
        self.original = cv2.imread(image_path)
        self.original_rgb = cv2.cvtColor(self.original, cv2.COLOR_BGR2RGB)
        self.height, self.width = self.original.shape[:2]

        # Initialize components
        self.pbn = PaintByNumbers(image_path, n_colors)
        self.analyzer = TechniqueAnalyzer(image_path)
        self.analysis = self.analyzer.full_analysis(n_colors)

        # Process paint-by-numbers
        self.pbn.quantize_colors()
        self.pbn.create_regions()
        self.pbn.create_template()

    def generate(self, level: InstructionLevel, output_dir: str):
        """
        Generate instructions at the specified level.

        Args:
            level: InstructionLevel (BEGINNER, EASY, INTERMEDIATE, ADVANCED)
            output_dir: Directory for output files
        """
        os.makedirs(output_dir, exist_ok=True)
        config = LEVEL_CONFIGS[level]

        print("=" * 60)
        print(f"GENERATING {config.name.upper()} LEVEL INSTRUCTIONS")
        print(f"Style: {config.description}")
        print("=" * 60)

        if level == InstructionLevel.BEGINNER:
            self._generate_beginner(output_dir, config)
        elif level == InstructionLevel.EASY:
            self._generate_easy(output_dir, config)
        elif level == InstructionLevel.INTERMEDIATE:
            self._generate_intermediate(output_dir, config)
        elif level == InstructionLevel.ADVANCED:
            self._generate_advanced(output_dir, config)

        # Always save reference and palette
        self._save_common_outputs(output_dir, config)

        print(f"\nOutput saved to {output_dir}/")
        print(f"Expected skill: {config.expected_skill}")
        print(f"Expected result: {config.expected_output}")

    def _generate_beginner(self, output_dir: str, config: LevelConfig):
        """Generate simple paint-by-numbers instructions."""

        # Template with numbers
        cv2.imwrite(
            os.path.join(output_dir, "template.png"),
            cv2.cvtColor(self.pbn.template, cv2.COLOR_RGB2BGR)
        )

        # Simple color guide
        matched = self.pbn.match_to_paint_colors()

        instructions = {
            "level": "Beginner",
            "style": "Simple Paint-by-Numbers",
            "instructions": [
                "1. Print the template on canvas paper or transfer to canvas",
                "2. Match each number on the template to the color guide below",
                "3. Fill in each numbered region with its matching color",
                "4. Start with larger regions, finish with smaller details",
                "5. Let each region dry before painting adjacent regions"
            ],
            "tips": [
                "Use a small brush for small regions",
                "Don't worry about perfect edges - small overlaps are okay",
                "Work from top to bottom to avoid smudging"
            ],
            "colors": [
                {
                    "number": c['number'],
                    "color_name": c['paint_name'],
                    "hex": c['extracted_hex']
                }
                for c in matched
            ]
        }

        with open(os.path.join(output_dir, "instructions.json"), 'w') as f:
            json.dump(instructions, f, indent=2)

        # Create simple instruction sheet
        self._create_beginner_sheet(output_dir, matched)

    def _generate_easy(self, output_dir: str, config: LevelConfig):
        """Generate guided paint-by-numbers with layer order."""

        # Template
        cv2.imwrite(
            os.path.join(output_dir, "template.png"),
            cv2.cvtColor(self.pbn.template, cv2.COLOR_RGB2BGR)
        )

        matched = self.pbn.match_to_paint_colors()

        # Determine layer order based on region size and position
        layers = self._determine_layer_order(matched)

        instructions = {
            "level": "Easy",
            "style": "Guided Paint-by-Numbers",
            "overview": f"This painting has {len(layers)} painting layers.",
            "layers": layers,
            "general_tips": [
                "Paint background colors (sky, distant areas) first",
                "Work from back to front",
                "Let each layer dry before the next",
                "Use a flat brush for large areas, round brush for details"
            ],
            "brush_suggestions": {
                "large_regions": "1-inch flat brush",
                "medium_regions": "1/2-inch flat brush",
                "small_regions": "Round brush #4-6",
                "details": "Round brush #1-2"
            }
        }

        with open(os.path.join(output_dir, "instructions.json"), 'w') as f:
            json.dump(instructions, f, indent=2)

        self._create_easy_sheet(output_dir, layers, matched)

    def _generate_intermediate(self, output_dir: str, config: LevelConfig):
        """Generate technique-aware instructions."""

        # Template
        cv2.imwrite(
            os.path.join(output_dir, "template.png"),
            cv2.cvtColor(self.pbn.template, cv2.COLOR_RGB2BGR)
        )

        matched = self.pbn.match_to_paint_colors()
        technique_zones = self.analysis.technique_zones

        steps = []
        step_num = 1

        # Group instructions by technique
        if self.analysis.has_dark_background:
            steps.append({
                "step": step_num,
                "title": "Base Layer",
                "instruction": "Cover the entire canvas with a dark base color.",
                "technique": "Flat fill",
                "brush": "2-inch flat brush",
                "motion": "Horizontal strokes across the canvas",
                "colors": ["Dark base color (black or dark blue)"]
            })
            step_num += 1

        for zone in technique_zones:
            if zone.technique == Technique.GLAZING:
                steps.append({
                    "step": step_num,
                    "title": "Glazing Layer",
                    "instruction": "Apply thin, transparent layers of color over the dark base.",
                    "technique": "Glazing - thin transparent layers",
                    "brush": "Soft flat brush or fan brush",
                    "motion": "Light vertical strokes, barely touching the canvas",
                    "tip": "Less paint = more transparency. Build up slowly.",
                    "colors": [f"RGB{c}" for c in zone.overlay_colors[:3]]
                })
                step_num += 1

            elif zone.technique == Technique.WET_ON_WET:
                steps.append({
                    "step": step_num,
                    "title": "Blending",
                    "instruction": "Blend colors while the paint is still wet.",
                    "technique": "Wet-on-wet blending",
                    "brush": "Soft round brush",
                    "motion": "Gentle back-and-forth where colors meet",
                    "tip": "Work quickly before paint dries.",
                    "colors": []
                })
                step_num += 1

        # Add numbered regions step
        steps.append({
            "step": step_num,
            "title": "Fill Remaining Regions",
            "instruction": "Fill the numbered regions according to the color guide.",
            "technique": "Standard fill",
            "brush": "Various sizes based on region",
            "motion": "Fill within the lines",
            "colors": []
        })

        instructions = {
            "level": "Intermediate",
            "style": "Technique-Aware Painting",
            "pattern_detected": self.analysis.dominant_pattern,
            "techniques_used": [z.technique.value for z in technique_zones],
            "steps": steps,
            "color_guide": [
                {
                    "number": c['number'],
                    "color_name": c['paint_name'],
                    "hex": c['extracted_hex']
                }
                for c in matched
            ]
        }

        with open(os.path.join(output_dir, "instructions.json"), 'w') as f:
            json.dump(instructions, f, indent=2)

        self._create_intermediate_sheet(output_dir, steps, matched)

    def _generate_advanced(self, output_dir: str, config: LevelConfig):
        """Generate full Bob Ross style instructions."""

        # Use the Bob Ross generator
        bob = BobRossGenerator(self.image_path, self.n_colors)
        bob.generate_steps()
        bob.create_step_by_step_guide(output_dir)

        # Also save regular template for reference
        cv2.imwrite(
            os.path.join(output_dir, "template_reference.png"),
            cv2.cvtColor(self.pbn.template, cv2.COLOR_RGB2BGR)
        )

    def _determine_layer_order(self, matched_colors: List[Dict]) -> List[Dict]:
        """Determine painting layer order based on colors and regions."""
        # Simple heuristic: darker and more common colors first
        layers = []

        # Sort by brightness and frequency
        sorted_colors = sorted(matched_colors,
                               key=lambda c: (sum(c['extracted_rgb']), -c['pixel_count']))

        # Group into layers
        dark_colors = [c for c in sorted_colors if sum(c['extracted_rgb']) < 200]
        mid_colors = [c for c in sorted_colors if 200 <= sum(c['extracted_rgb']) < 500]
        light_colors = [c for c in sorted_colors if sum(c['extracted_rgb']) >= 500]

        if dark_colors:
            layers.append({
                "layer": 1,
                "name": "Dark Background Colors",
                "description": "Paint these first - they form the base of your painting",
                "color_numbers": [c['number'] for c in dark_colors],
                "tip": "Don't worry about being perfect - lighter colors will cover edges"
            })

        if mid_colors:
            layers.append({
                "layer": 2,
                "name": "Mid-Tone Colors",
                "description": "Add these next - they create depth and dimension",
                "color_numbers": [c['number'] for c in mid_colors],
                "tip": "Let the dark layer dry first, or blend at the edges"
            })

        if light_colors:
            layers.append({
                "layer": 3,
                "name": "Light and Highlight Colors",
                "description": "Add these last - they make everything pop",
                "color_numbers": [c['number'] for c in light_colors],
                "tip": "Use a lighter touch - highlights should be the finishing touches"
            })

        return layers

    def _save_common_outputs(self, output_dir: str, config: LevelConfig):
        """Save outputs common to all levels."""

        # Colored reference
        cv2.imwrite(
            os.path.join(output_dir, "reference.png"),
            cv2.cvtColor(self.pbn.quantized_image, cv2.COLOR_RGB2BGR)
        )

        # Palette image
        self._create_palette_image(output_dir)

        # Summary info
        summary = {
            "level": config.name,
            "style": config.description,
            "n_colors": self.n_colors,
            "pattern_detected": self.analysis.dominant_pattern,
            "difficulty": self.analysis.overall_difficulty,
            "expected_skill": config.expected_skill,
            "expected_output": config.expected_output,
            "image_size": f"{self.width} x {self.height}"
        }

        with open(os.path.join(output_dir, "summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)

    def _create_palette_image(self, output_dir: str):
        """Create a visual color palette."""
        swatch_size = 50
        cols = min(self.n_colors, 5)
        rows = (self.n_colors + cols - 1) // cols

        width = cols * (swatch_size + 10) + 20
        height = rows * (swatch_size + 30) + 20

        img = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
        except:
            font = ImageFont.load_default()

        for i, color in enumerate(self.pbn.palette):
            row = i // cols
            col = i % cols

            x = 10 + col * (swatch_size + 10)
            y = 10 + row * (swatch_size + 30)

            # Color swatch
            draw.rectangle([x, y, x + swatch_size, y + swatch_size],
                          fill=tuple(color), outline='black')

            # Number
            draw.text((x + swatch_size//2 - 5, y + swatch_size + 5),
                     str(i + 1), fill='black', font=font)

        img.save(os.path.join(output_dir, "palette.png"))

    def _create_beginner_sheet(self, output_dir: str, matched: List[Dict]):
        """Create beginner instruction sheet."""
        width, height = 800, 1000
        img = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(img)

        try:
            font_title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
            font_body = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        except:
            font_title = font_body = ImageFont.load_default()

        y = 30

        # Title
        draw.text((30, y), "PAINT BY NUMBERS", fill='black', font=font_title)
        y += 40
        draw.text((30, y), "Beginner Level - Just fill in the numbers!", fill='gray', font=font_body)
        y += 50

        # Instructions
        instructions = [
            "1. Find region #1 on your template",
            "2. Look up color #1 in the guide below",
            "3. Fill in that region with the color",
            "4. Repeat for all numbers",
            "5. Let dry and enjoy your painting!"
        ]

        for inst in instructions:
            draw.text((30, y), inst, fill='black', font=font_body)
            y += 25

        y += 30

        # Color guide
        draw.text((30, y), "COLOR GUIDE:", fill='black', font=font_title)
        y += 40

        for c in matched[:15]:  # Show first 15
            # Swatch
            draw.rectangle([30, y, 60, y + 25], fill=tuple(c['extracted_rgb']), outline='black')
            # Number and name
            draw.text((70, y + 3), f"#{c['number']} - {c['paint_name']}", fill='black', font=font_body)
            y += 35

        img.save(os.path.join(output_dir, "instruction_sheet.png"))

    def _create_easy_sheet(self, output_dir: str, layers: List[Dict], matched: List[Dict]):
        """Create easy level instruction sheet."""
        width, height = 800, 1200
        img = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(img)

        try:
            font_title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
            font_body = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
        except:
            font_title = font_body = ImageFont.load_default()

        y = 30

        # Title
        draw.text((30, y), "GUIDED PAINT BY NUMBERS", fill='black', font=font_title)
        y += 35
        draw.text((30, y), "Easy Level - Paint in layers for better results", fill='gray', font=font_body)
        y += 50

        # Layers
        for layer in layers:
            draw.rectangle([25, y-5, width-25, y + 80], outline='lightgray')
            draw.text((30, y), f"LAYER {layer['layer']}: {layer['name']}", fill='darkblue', font=font_title)
            y += 30
            draw.text((30, y), layer['description'], fill='black', font=font_body)
            y += 20
            draw.text((30, y), f"Colors: {', '.join(map(str, layer['color_numbers']))}", fill='black', font=font_body)
            y += 20
            draw.text((30, y), f"Tip: {layer['tip']}", fill='green', font=font_body)
            y += 40

        y += 20

        # Color guide
        draw.text((30, y), "COLOR GUIDE:", fill='black', font=font_title)
        y += 35

        cols = 3
        col_width = (width - 60) // cols

        for i, c in enumerate(matched[:15]):
            col = i % cols
            row = i // cols

            x = 30 + col * col_width
            cy = y + row * 30

            draw.rectangle([x, cy, x + 20, cy + 20], fill=tuple(c['extracted_rgb']), outline='black')
            draw.text((x + 25, cy + 2), f"#{c['number']} {c['paint_name'][:15]}", fill='black', font=font_body)

        img.save(os.path.join(output_dir, "instruction_sheet.png"))

    def _create_intermediate_sheet(self, output_dir: str, steps: List[Dict], matched: List[Dict]):
        """Create intermediate level instruction sheet."""
        width, height = 850, 1400
        img = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(img)

        try:
            font_title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
            font_body = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 11)
            font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 10)
        except:
            font_title = font_body = font_small = ImageFont.load_default()

        y = 25

        # Title
        draw.text((25, y), "TECHNIQUE-AWARE PAINTING GUIDE", fill='black', font=font_title)
        y += 30
        draw.text((25, y), f"Intermediate Level | Pattern: {self.analysis.dominant_pattern.upper()}", fill='gray', font=font_body)
        y += 40

        # Steps
        for step in steps:
            draw.rectangle([20, y-5, width-20, y + 100], outline='#ddd', fill='#fafafa')

            draw.text((25, y), f"STEP {step['step']}: {step['title']}", fill='darkblue', font=font_title)
            y += 25

            draw.text((25, y), step['instruction'], fill='black', font=font_body)
            y += 20

            draw.text((25, y), f"Technique: {step['technique']}", fill='#666', font=font_small)
            y += 16
            draw.text((25, y), f"Brush: {step['brush']}", fill='#666', font=font_small)
            y += 16
            draw.text((25, y), f"Motion: {step['motion']}", fill='#666', font=font_small)
            y += 16

            if 'tip' in step:
                draw.text((25, y), f"Tip: {step['tip']}", fill='green', font=font_small)
                y += 16

            y += 25

        # Color guide at bottom
        y += 10
        draw.text((25, y), "COLOR GUIDE:", fill='black', font=font_title)
        y += 30

        cols = 3
        col_width = (width - 50) // cols

        for i, c in enumerate(matched):
            col = i % cols
            row = i // cols

            x = 25 + col * col_width
            cy = y + row * 28

            if cy > height - 30:
                break

            draw.rectangle([x, cy, x + 20, cy + 20], fill=tuple(c['extracted_rgb']), outline='black')
            draw.text((x + 25, cy + 2), f"#{c['number']} {c['paint_name'][:18]}", fill='black', font=font_small)

        img.save(os.path.join(output_dir, "instruction_sheet.png"))


def generate_all_levels(image_path: str, n_colors: int = 15, base_output_dir: str = "output"):
    """
    Generate instructions at ALL levels for comparison.

    Args:
        image_path: Path to image
        n_colors: Number of colors
        base_output_dir: Base directory for outputs
    """
    print("=" * 60)
    print("GENERATING ALL INSTRUCTION LEVELS")
    print("=" * 60)

    generator = UnifiedInstructionGenerator(image_path, n_colors)

    for level in InstructionLevel:
        output_dir = os.path.join(base_output_dir, f"level_{level.value}_{level.name.lower()}")
        generator.generate(level, output_dir)
        print()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Unified Instruction Generator")
        print("=" * 40)
        print("Usage:")
        print("  python instruction_generator.py <image> [level] [output_dir]")
        print()
        print("Levels:")
        print("  1 = Beginner (Simple Paint-by-Numbers)")
        print("  2 = Easy (Guided with layer order)")
        print("  3 = Intermediate (Technique-aware)")
        print("  4 = Advanced (Bob Ross style)")
        print("  all = Generate all levels")
        print()
        print("Examples:")
        print("  python instruction_generator.py aurora.png 1")
        print("  python instruction_generator.py aurora.png 4 bob_ross_output")
        print("  python instruction_generator.py aurora.png all")
        sys.exit(0)

    image_path = sys.argv[1]
    level_arg = sys.argv[2] if len(sys.argv) > 2 else "2"
    output_dir = sys.argv[3] if len(sys.argv) > 3 else None

    if level_arg.lower() == "all":
        generate_all_levels(image_path, 15, output_dir or "output_all_levels")
    else:
        level_num = int(level_arg)
        level = InstructionLevel(level_num)

        if output_dir is None:
            output_dir = f"output_{level.name.lower()}"

        generator = UnifiedInstructionGenerator(image_path, 15)
        generator.generate(level, output_dir)
