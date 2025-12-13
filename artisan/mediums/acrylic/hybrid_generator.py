"""
Hybrid Acrylic Generator - Combines Bob Ross checkpoints with granular substeps.

Structure:
- Keep original Bob Ross high-level steps as checkpoints
- Break each checkpoint into granular substeps by region
- Generate cumulative progress images (build up from blank canvas)
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple
import os

from ...generators.bob_ross import BobRossGenerator
from ..base import Material, MaterialType, CanvasArea, Substep, Layer
from ...core import RegionDetector, RegionDetectionConfig, find_colors_in_layer, PaintByNumbers


class HybridAcrylicGenerator:
    """
    Combines Bob Ross checkpoints with granular substep breakdown.

    Generates:
    - Original high-level Bob Ross steps (checkpoints)
    - Granular substeps for each checkpoint
    - Cumulative progress images
    """

    def __init__(self, image_path: str, n_colors: int = 15):
        """
        Initialize hybrid generator.

        Args:
            image_path: Path to source image
            n_colors: Number of colors
        """
        self.image_path = image_path
        self.n_colors = n_colors

        # Use original Bob Ross generator for checkpoints
        self.bob_ross = BobRossGenerator(image_path, n_colors)
        self.bob_ross.generate_steps()  # Generate high-level steps

        # Load image
        self.original = cv2.imread(image_path)
        self.original_rgb = cv2.cvtColor(self.original, cv2.COLOR_BGR2RGB)
        self.height, self.width = self.original.shape[:2]

        # Color quantization for substep breakdown
        self.pbn = PaintByNumbers(image_path, n_colors)
        self.pbn.quantize_colors()
        self.pbn.create_regions()

        # Region detector for granular breakdown
        self.region_detector = RegionDetector(
            self.height,
            self.width,
            RegionDetectionConfig(use_contours=True, min_region_pixels=200)
        )

        self.checkpoints = []
        self.cumulative_canvas = None

    def generate_hybrid_guide(self) -> List[Dict]:
        """
        Generate complete guide with checkpoints and substeps.

        Returns:
            List of checkpoint dicts, each containing substeps
        """
        self.checkpoints = []

        # Initialize blank canvas for cumulative building
        self.cumulative_canvas = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255

        # For each Bob Ross checkpoint step
        for bob_step in self.bob_ross.steps:
            checkpoint = {
                'checkpoint_number': bob_step.step_number,
                'title': bob_step.title,
                'overview': bob_step.instruction,
                'technique_tip': bob_step.technique_tip,
                'encouragement': bob_step.encouragement,
                'substeps': []
            }

            # Break down this checkpoint into granular substeps
            substeps = self._generate_substeps_for_checkpoint(bob_step)
            checkpoint['substeps'] = substeps

            # Update cumulative canvas
            self._apply_checkpoint_to_canvas(bob_step)
            checkpoint['progress_canvas'] = self.cumulative_canvas.copy()

            self.checkpoints.append(checkpoint)

        return self.checkpoints

    def _generate_substeps_for_checkpoint(self, bob_step) -> List[Dict]:
        """
        Break down a Bob Ross checkpoint into granular substeps.

        Args:
            bob_step: PaintingStep from Bob Ross generator

        Returns:
            List of substep dicts with region breakdowns
        """
        substeps = []

        # Get the region mask for this step
        step_mask = bob_step.region_mask
        if step_mask is None:
            # Whole canvas
            step_mask = np.ones((self.height, self.width), dtype=bool)

        # For each color in this step
        for i, (color_name, color_rgb) in enumerate(zip(bob_step.colors, bob_step.color_rgbs)):
            # Find where this color appears in the step region
            # Match color from palette
            color_idx = self._find_closest_palette_color(color_rgb)

            if color_idx is not None:
                # Get mask for this color within the step region
                color_mask = (self.pbn.label_map == color_idx) & step_mask

                if np.sum(color_mask) < 100:  # Too small
                    # Create single substep for entire region
                    substeps.append({
                        'substep_id': f"{bob_step.step_number}.{len(substeps)+1}",
                        'color_name': color_name,
                        'color_rgb': color_rgb,
                        'area_name': bob_step.canvas_region,
                        'coverage_percent': np.sum(step_mask) / (self.height * self.width) * 100,
                        'brush': bob_step.brush.value,
                        'motion': bob_step.motion.value,
                        'instruction': f"Apply {color_name} to {bob_step.canvas_region} using {bob_step.motion.value}."
                    })
                else:
                    # Find distinct regions
                    regions = self.region_detector.find_regions(color_mask)

                    # Create substep for each region
                    for region in regions:
                        substeps.append({
                            'substep_id': f"{bob_step.step_number}.{len(substeps)+1}",
                            'color_name': color_name,
                            'color_rgb': color_rgb,
                            'area_name': region['name'],
                            'bounds': region['bounds'],
                            'coverage_percent': region['coverage_percent'],
                            'brush': bob_step.brush.value,
                            'motion': bob_step.motion.value,
                            'instruction': f"Apply {color_name} to the {region['name']} using {bob_step.motion.value}."
                        })
            else:
                # Color not in quantized palette - use whole region
                substeps.append({
                    'substep_id': f"{bob_step.step_number}.{len(substeps)+1}",
                    'color_name': color_name,
                    'color_rgb': color_rgb,
                    'area_name': bob_step.canvas_region,
                    'coverage_percent': np.sum(step_mask) / (self.height * self.width) * 100,
                    'brush': bob_step.brush.value,
                    'motion': bob_step.motion.value,
                    'instruction': f"Apply {color_name} to {bob_step.canvas_region} using {bob_step.motion.value}."
                })

        return substeps

    def _find_closest_palette_color(self, target_rgb: Tuple[int, int, int]) -> int:
        """Find closest color in quantized palette."""
        target = np.array(target_rgb)
        min_dist = float('inf')
        closest_idx = None

        for idx, palette_color in enumerate(self.pbn.palette):
            dist = np.linalg.norm(target - palette_color)
            if dist < min_dist:
                min_dist = dist
                closest_idx = idx

        # Only return if reasonably close
        if min_dist < 80:  # Threshold
            return closest_idx
        return None

    def _apply_checkpoint_to_canvas(self, bob_step):
        """
        Apply checkpoint colors to cumulative canvas.

        Simulates painting the step on the canvas.
        """
        if bob_step.region_mask is None:
            # Whole canvas
            mask = np.ones((self.height, self.width), dtype=bool)
        else:
            mask = bob_step.region_mask

        # Apply average color to masked region
        if bob_step.color_rgbs:
            avg_color = np.mean(bob_step.color_rgbs, axis=0).astype(int)
            for c in range(3):
                self.cumulative_canvas[:, :, c] = np.where(
                    mask,
                    avg_color[c],
                    self.cumulative_canvas[:, :, c]
                )
