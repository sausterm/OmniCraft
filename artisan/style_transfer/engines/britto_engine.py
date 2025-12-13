"""
Romero Britto style transfer engine
Wraps the existing Britto implementation into the modular framework
"""

import sys
from pathlib import Path
from typing import List
from PIL import Image
import time

# Add parent directory to path to import britto_style_transfer
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from britto_style_transfer import (
    simplify_image,
    quantize_to_britto_palette,
    create_bold_outlines,
    add_britto_patterns,
    remove_background
)
from ..base import LocalStyleEngine, StyleResult


class BrittoEngine(LocalStyleEngine):
    """
    Romero Britto pop art style transfer engine.

    Converts images to Britto's signature style:
    - Bright, vibrant colors
    - Bold black outlines
    - Geometric patterns
    - Flat color areas
    """

    def __init__(self):
        super().__init__(
            name="britto",
            description="Romero Britto pop art style with vibrant colors and bold outlines"
        )

        # Register available styles/presets
        self.register_style(
            "britto_classic",
            description="Classic Britto style with all features",
            outline_thickness=8,
            add_patterns=True,
            pattern_intensity=0.7,
            remove_bg=True
        )

        self.register_style(
            "britto_minimal",
            description="Minimal Britto style without patterns",
            outline_thickness=6,
            add_patterns=False,
            pattern_intensity=0.0,
            remove_bg=True
        )

        self.register_style(
            "britto_bold",
            description="Extra bold outlines with intense patterns",
            outline_thickness=12,
            add_patterns=True,
            pattern_intensity=0.9,
            remove_bg=True
        )

    def get_available_styles(self) -> List[str]:
        """Return available Britto style variations"""
        return list(self._styles.keys())

    def apply_style(
        self,
        image: Image.Image,
        style: str = "britto_classic",
        outline_thickness: int = None,
        add_patterns: bool = None,
        pattern_intensity: float = None,
        remove_bg: bool = None,
        **kwargs
    ) -> StyleResult:
        """
        Apply Britto style to an image.

        Args:
            image: Input PIL Image
            style: Style preset to use
            outline_thickness: Override thickness of black outlines
            add_patterns: Override whether to add patterns
            pattern_intensity: Override pattern intensity (0-1)
            remove_bg: Override whether to remove background

        Returns:
            StyleResult with the styled image
        """
        start_time = time.time()

        # Get style preset parameters
        if style not in self._styles:
            raise ValueError(f"Unknown style: {style}. Available: {self.get_available_styles()}")

        preset = self._styles[style]

        # Use preset values unless overridden
        outline_thickness = outline_thickness if outline_thickness is not None else preset['outline_thickness']
        add_patterns = add_patterns if add_patterns is not None else preset['add_patterns']
        pattern_intensity = pattern_intensity if pattern_intensity is not None else preset['pattern_intensity']
        remove_bg = remove_bg if remove_bg is not None else preset['remove_bg']

        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Apply Britto transformation pipeline
        result_image = image.copy()

        # Step 1: Remove background if requested
        if remove_bg:
            result_image = remove_background(result_image, threshold=50)

        # Step 2: Simplify image
        result_image = simplify_image(result_image, bilateral_d=20, sigma_color=100, sigma_space=100)
        result_image = simplify_image(result_image, bilateral_d=15, sigma_color=80, sigma_space=80)

        # Step 3: Apply color palette
        result_image = quantize_to_britto_palette(result_image)

        # Step 4: Add bold outlines
        result_image = create_bold_outlines(result_image, thickness=outline_thickness)

        # Step 5: Add patterns if requested
        if add_patterns:
            result_image = add_britto_patterns(result_image, intensity=pattern_intensity)

        # Create result
        metadata = {
            'style': style,
            'outline_thickness': outline_thickness,
            'patterns_enabled': add_patterns,
            'pattern_intensity': pattern_intensity if add_patterns else 0,
            'background_removed': remove_bg,
            'original_size': image.size,
            'final_size': result_image.size
        }

        return self._create_result(result_image, style, metadata, start_time)
