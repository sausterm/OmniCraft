"""
Replicate API style transfer engine
Provides access to 100+ style transfer models via Replicate API
"""

import os
from typing import List, Optional, Dict, Any
from PIL import Image
import time
import io
import base64

try:
    import replicate
    REPLICATE_AVAILABLE = True
except ImportError:
    REPLICATE_AVAILABLE = False

from ..base import APIStyleEngine, StyleResult


class ReplicateEngine(APIStyleEngine):
    """
    Style transfer engine using Replicate API.

    Provides access to various style transfer models:
    - Stable Diffusion with ControlNet
    - Neural Style Transfer models
    - Artist-specific models

    Requires: pip install replicate
    Set REPLICATE_API_TOKEN environment variable
    """

    # Model mappings for different artistic styles
    STYLE_MODELS = {
        "van_gogh": {
            "model": "rossjillian/controlnet:795433b19458d0f4fa172a7ccf93178d2adb1cb8ab2ad6c8faecc48a9c45ca93",
            "prompt": "painting in the style of Van Gogh, starry night style, impressionist, swirling brushstrokes",
            "description": "Van Gogh impressionist style with swirling brushstrokes"
        },
        "picasso_cubist": {
            "model": "rossjillian/controlnet:795433b19458d0f4fa172a7ccf93178d2adb1cb8ab2ad6c8faecc48a9c45ca93",
            "prompt": "cubist painting in the style of Picasso, geometric shapes, multiple perspectives",
            "description": "Picasso-style cubist interpretation"
        },
        "monet": {
            "model": "rossjillian/controlnet:795433b19458d0f4fa172a7ccf93178d2adb1cb8ab2ad6c8faecc48a9c45ca93",
            "prompt": "impressionist painting in the style of Monet, soft brushstrokes, water lilies style",
            "description": "Monet impressionist style"
        },
        "pop_art": {
            "model": "rossjillian/controlnet:795433b19458d0f4fa172a7ccf93178d2adb1cb8ab2ad6c8faecc48a9c45ca93",
            "prompt": "pop art style, bold colors, high contrast, Andy Warhol style",
            "description": "Pop art with bold colors and high contrast"
        },
        "watercolor": {
            "model": "rossjillian/controlnet:795433b19458d0f4fa172a7ccf93178d2adb1cb8ab2ad6c8faecc48a9c45ca93",
            "prompt": "watercolor painting, soft edges, translucent colors, artistic",
            "description": "Watercolor painting style"
        },
        "anime": {
            "model": "cjwbw/anything-v3.0:f410ed4c6a0c3bf8b76747860b3a3c9e4c8b5a827a16eac9dd5ad9642edce9a2",
            "prompt": "anime style illustration, vibrant colors, clean lines",
            "description": "Anime/manga illustration style"
        },
    }

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Replicate engine.

        Args:
            api_key: Replicate API token (or set REPLICATE_API_TOKEN env var)
        """
        if not REPLICATE_AVAILABLE:
            raise ImportError(
                "replicate package not installed. "
                "Install with: pip install replicate"
            )

        api_key = api_key or os.environ.get("REPLICATE_API_TOKEN")

        super().__init__(
            name="replicate",
            description="Multi-model style transfer via Replicate API",
            api_key=api_key,
            api_base_url="https://api.replicate.com"
        )

        # Register all available styles
        for style_name, style_info in self.STYLE_MODELS.items():
            self.register_style(
                style_name,
                description=style_info["description"],
                model=style_info["model"],
                base_prompt=style_info["prompt"]
            )

    def _validate_api_credentials(self) -> bool:
        """Check if API credentials are valid"""
        if not self.api_key:
            raise ValueError(
                "Replicate API key not provided. "
                "Set REPLICATE_API_TOKEN environment variable or pass api_key parameter"
            )
        return True

    def get_available_styles(self) -> List[str]:
        """Get list of available styles"""
        return list(self.STYLE_MODELS.keys())

    def _image_to_data_uri(self, image: Image.Image) -> str:
        """Convert PIL Image to data URI for API"""
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"

    def apply_style(
        self,
        image: Image.Image,
        style: str,
        prompt_override: Optional[str] = None,
        strength: float = 0.8,
        guidance_scale: float = 7.5,
        **kwargs
    ) -> StyleResult:
        """
        Apply style using Replicate API.

        Args:
            image: Input PIL Image
            style: Style name from STYLE_MODELS
            prompt_override: Custom prompt to override default
            strength: How strongly to apply the style (0-1)
            guidance_scale: How closely to follow the prompt (1-20)
            **kwargs: Additional model-specific parameters

        Returns:
            StyleResult with styled image
        """
        start_time = time.time()
        self._validate_api_credentials()

        if style not in self.STYLE_MODELS:
            raise ValueError(f"Unknown style: {style}. Available: {self.get_available_styles()}")

        style_config = self.STYLE_MODELS[style]

        # Prepare input
        image_uri = self._image_to_data_uri(image)
        prompt = prompt_override or style_config["prompt"]

        # Run model
        model_id = style_config["model"]

        input_params = {
            "image": image_uri,
            "prompt": prompt,
            "strength": strength,
            "guidance_scale": guidance_scale,
            **kwargs
        }

        # Call Replicate API
        output = replicate.run(model_id, input=input_params)

        # Handle output (usually a URL or list of URLs)
        if isinstance(output, list):
            output_url = output[0]
        else:
            output_url = output

        # Download result image
        import requests
        response = requests.get(output_url)
        result_image = Image.open(io.BytesIO(response.content))

        # Create metadata
        metadata = {
            'style': style,
            'model': model_id,
            'prompt': prompt,
            'strength': strength,
            'guidance_scale': guidance_scale,
            'original_size': image.size,
            'final_size': result_image.size,
            'api': 'replicate'
        }

        return self._create_result(result_image, style, metadata, start_time)

    def get_cost_estimate(self, style: str, image_size: tuple) -> Optional[float]:
        """
        Estimate cost for Replicate API call.

        Note: Actual costs vary by model. This is a rough estimate.
        """
        # Rough estimate: $0.002-0.01 per image depending on model
        return 0.005
