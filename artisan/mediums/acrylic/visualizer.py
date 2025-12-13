"""
Acrylic Medium Visualizer - Generate progress images for painting steps.

Creates visual guides showing:
- Individual substep images highlighting the target region
- Layer overview images (filmstrip of substeps)
- Cumulative progress images after each substep
- Complete painting guide with annotations
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
from typing import List, Optional, Tuple

from ..base import Layer, Substep


class AcrylicVisualizer:
    """Generate visual outputs for acrylic painting instructions."""

    def __init__(self, image_rgb: np.ndarray, height: int, width: int):
        """
        Initialize visualizer.

        Args:
            image_rgb: Original image as RGB array
            height: Image height
            width: Image width
        """
        self.original = image_rgb
        self.height = height
        self.width = width

    def create_granular_guide(
        self,
        layers: List[Layer],
        output_dir: str,
        create_filmstrips: bool = True
    ):
        """
        Create complete granular guide with substep images.

        Output structure:
        output_dir/
        ├── layer_01_far_background/
        │   ├── substep_1.1.png
        │   ├── substep_1.2.png
        │   ├── progress.png
        │   └── filmstrip.png (optional)
        ├── layer_02_middle_ground/
        │   └── ...
        └── complete_progress.png

        Args:
            layers: List of Layer objects with substeps
            output_dir: Output directory path
            create_filmstrips: Whether to create filmstrip overview images
        """
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nGenerating granular visual guide to {output_dir}/")
        print("=" * 70)

        # Initialize cumulative canvas
        cumulative_canvas = np.ones((self.height, self.width, 3), dtype=np.uint8) * 240

        for layer in layers:
            # Create layer directory
            layer_name_safe = layer.name.lower().replace(' ', '_').replace('-', '_')
            layer_dir = os.path.join(output_dir, f"layer_{layer.layer_number:02d}_{layer_name_safe}")
            os.makedirs(layer_dir, exist_ok=True)

            print(f"\nLayer {layer.layer_number}: {layer.name}")
            print(f"  Substeps: {len(layer.substeps)}")

            # Generate substep images
            for substep in layer.substeps:
                self._create_substep_highlight_image(
                    substep,
                    layer_dir,
                    self.original
                )

                # Update cumulative canvas
                self._apply_substep_to_canvas(substep, cumulative_canvas)

            # Save layer progress
            progress_path = os.path.join(layer_dir, 'progress.png')
            cv2.imwrite(progress_path, cv2.cvtColor(cumulative_canvas, cv2.COLOR_RGB2BGR))

            # Optionally create filmstrip
            if create_filmstrips and layer.substeps:
                self._create_layer_filmstrip(layer, layer_dir, self.original)

        # Save final cumulative progress
        final_path = os.path.join(output_dir, 'complete_progress.png')
        cv2.imwrite(final_path, cv2.cvtColor(cumulative_canvas, cv2.COLOR_RGB2BGR))

        print("\n" + "=" * 70)
        total_substeps = sum(len(layer.substeps) for layer in layers)
        print(f"✓ Granular guide complete! {total_substeps} substeps generated.")
        print(f"✓ Output saved to: {output_dir}")

    def _create_substep_highlight_image(
        self,
        substep: Substep,
        output_dir: str,
        reference_image: np.ndarray
    ):
        """
        Create image highlighting the substep's target region.

        Args:
            substep: Substep to visualize
            output_dir: Directory to save image
            reference_image: Reference image to overlay on
        """
        # Create a dimmed version of the reference
        highlight_img = (reference_image * 0.4).astype(np.uint8)

        # Get full mask for this substep's area
        bounds = substep.area.bounds
        y1, y2, x1, x2 = bounds
        py1, py2 = int(y1 * self.height), int(y2 * self.height)
        px1, px2 = int(x1 * self.width), int(x2 * self.width)

        # Highlight the target region
        if substep.area.mask is not None:
            # Use the actual mask
            full_mask = np.zeros((self.height, self.width), dtype=bool)

            # Ensure mask fits in bounds (handle rounding errors)
            mask_h, mask_w = substep.area.mask.shape
            target_h = py2 - py1
            target_w = px2 - px1

            # Resize mask if there's a mismatch
            if mask_h != target_h or mask_w != target_w:
                mask_h = min(mask_h, target_h)
                mask_w = min(mask_w, target_w)

            full_mask[py1:py1+mask_h, px1:px1+mask_w] = substep.area.mask[:mask_h, :mask_w]

            # Brighten the masked region
            highlight_img[full_mask] = reference_image[full_mask]

            # Draw colored outline
            contours, _ = cv2.findContours(
                full_mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            for contour in contours:
                cv2.drawContours(highlight_img, [contour], -1, substep.material.color_rgb, 3)
        else:
            # Just highlight the bounding box
            highlight_img[py1:py2, px1:px2] = reference_image[py1:py2, px1:px2]
            cv2.rectangle(highlight_img, (px1, py1), (px2, py2), substep.material.color_rgb, 3)

        # Add text annotation
        pil_img = Image.fromarray(highlight_img)
        draw = ImageDraw.Draw(pil_img)

        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
            font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        except:
            font = ImageFont.load_default()
            font_small = font

        # Draw color swatch and label
        swatch_size = 30
        swatch_x = 10
        swatch_y = self.height - 80

        draw.rectangle(
            [(swatch_x, swatch_y), (swatch_x + swatch_size, swatch_y + swatch_size)],
            fill=substep.material.color_rgb,
            outline=(255, 255, 255),
            width=2
        )

        # Draw text with background
        text = f"{substep.substep_id}: {substep.material.name}"
        text_bbox = draw.textbbox((swatch_x + swatch_size + 10, swatch_y), text, font=font)
        draw.rectangle(text_bbox, fill=(0, 0, 0, 180))
        draw.text((swatch_x + swatch_size + 10, swatch_y), text, fill=(255, 255, 255), font=font)

        # Draw area name
        area_text = f"Area: {substep.area.name}"
        draw.text((swatch_x + swatch_size + 10, swatch_y + 25), area_text, fill=(255, 255, 255), font=font_small)

        # Save
        output_path = os.path.join(output_dir, f"substep_{substep.substep_id}.png")
        pil_img.save(output_path)

    def _apply_substep_to_canvas(self, substep: Substep, canvas: np.ndarray):
        """
        Apply substep's color to the cumulative canvas.

        Args:
            substep: Substep to apply
            canvas: Cumulative canvas (modified in place)
        """
        bounds = substep.area.bounds
        y1, y2, x1, x2 = bounds
        py1, py2 = int(y1 * self.height), int(y2 * self.height)
        px1, px2 = int(x1 * self.width), int(x2 * self.width)

        if substep.area.mask is not None:
            # Apply using mask
            full_mask = np.zeros((self.height, self.width), dtype=bool)

            # Ensure mask fits in bounds (handle rounding errors)
            mask_h, mask_w = substep.area.mask.shape
            target_h = py2 - py1
            target_w = px2 - px1

            # Resize mask if there's a mismatch
            if mask_h != target_h or mask_w != target_w:
                mask_h = min(mask_h, target_h)
                mask_w = min(mask_w, target_w)

            full_mask[py1:py1+mask_h, px1:px1+mask_w] = substep.area.mask[:mask_h, :mask_w]

            for c in range(3):
                canvas[:, :, c] = np.where(
                    full_mask,
                    substep.material.color_rgb[c],
                    canvas[:, :, c]
                )
        else:
            # Apply to bounding box
            canvas[py1:py2, px1:px2] = substep.material.color_rgb

    def _create_layer_filmstrip(
        self,
        layer: Layer,
        output_dir: str,
        reference_image: np.ndarray
    ):
        """
        Create filmstrip showing all substeps in a layer.

        Args:
            layer: Layer to visualize
            output_dir: Directory to save filmstrip
            reference_image: Reference image
        """
        if not layer.substeps:
            return

        # Determine filmstrip dimensions
        n_substeps = len(layer.substeps)
        thumb_height = 150
        thumb_width = int(thumb_height * self.width / self.height)

        # Create filmstrip image
        filmstrip = Image.new('RGB', (thumb_width * n_substeps, thumb_height), (40, 40, 40))

        for i, substep in enumerate(layer.substeps):
            # Create thumbnail highlighting this region
            highlight_img = (reference_image * 0.5).astype(np.uint8)

            bounds = substep.area.bounds
            y1, y2, x1, x2 = bounds
            py1, py2 = int(y1 * self.height), int(y2 * self.height)
            px1, px2 = int(x1 * self.width), int(x2 * self.width)

            # Highlight region
            if substep.area.mask is not None:
                full_mask = np.zeros((self.height, self.width), dtype=bool)

                # Handle potential size mismatches
                mask_h, mask_w = substep.area.mask.shape
                target_h = py2 - py1
                target_w = px2 - px1
                if mask_h != target_h or mask_w != target_w:
                    mask_h = min(mask_h, target_h)
                    mask_w = min(mask_w, target_w)

                full_mask[py1:py1+mask_h, px1:px1+mask_w] = substep.area.mask[:mask_h, :mask_w]
                highlight_img[full_mask] = reference_image[full_mask]
            else:
                highlight_img[py1:py2, px1:px2] = reference_image[py1:py2, px1:px2]

            # Convert to PIL and resize
            thumb = Image.fromarray(highlight_img)
            thumb.thumbnail((thumb_width, thumb_height), Image.Resampling.LANCZOS)

            # Paste into filmstrip
            filmstrip.paste(thumb, (i * thumb_width, 0))

        # Save filmstrip
        filmstrip_path = os.path.join(output_dir, 'filmstrip.png')
        filmstrip.save(filmstrip_path)
