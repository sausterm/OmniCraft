"""
Replicate API style transfer engine
Uses direct HTTP calls to avoid Python 3.14 compatibility issues with replicate SDK
"""

import os
import requests
import time
import io
import base64
from typing import List, Optional, Dict, Any
from PIL import Image

from ..base import APIStyleEngine, StyleResult

REPLICATE_AVAILABLE = True  # Using HTTP, no SDK needed


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

    # img2img model for style transfer (preserves structure while applying style)
    DEFAULT_MODEL = "stability-ai/sdxl:7762fd07cf82c948538e41f63f77d685e02b063e37e496e96eefd46c929f9bdc"

    # Model mappings for different artistic styles
    STYLE_MODELS = {
        "van_gogh": {
            "prompt": "A painting in the style of Vincent Van Gogh, post-impressionist masterpiece, swirling brushstrokes like Starry Night, thick impasto texture, vivid expressive colors, emotional and dynamic",
            "description": "Van Gogh impressionist style with swirling brushstrokes"
        },
        "picasso_cubist": {
            "prompt": "A cubist painting in the style of Pablo Picasso, geometric abstract shapes, multiple perspectives shown simultaneously, fragmented forms, bold angular lines, analytical cubism masterpiece",
            "description": "Picasso-style cubist interpretation"
        },
        "monet": {
            "prompt": "An impressionist painting in the style of Claude Monet, soft delicate brushstrokes, dappled natural light, pastel colors, dreamy atmospheric quality, water lilies style",
            "description": "Monet impressionist style"
        },
        "pop_art": {
            "prompt": "Pop art style painting, bold primary colors, thick black outlines, high contrast, flat graphic style, Andy Warhol and Roy Lichtenstein inspired, comic book aesthetic",
            "description": "Pop art with bold colors and high contrast"
        },
        "britto_style": {
            "prompt": "Romero Britto neo-pop art style, vibrant bold colors, thick black outlines, geometric patterns, cubist influences, hearts and decorative patterns, playful joyful optimistic mood, flat colorful aesthetic",
            "description": "Romero Britto geometric pop art with vibrant colors"
        },
        "watercolor": {
            "prompt": "A beautiful watercolor painting, soft wet edges, translucent layered colors, artistic color bleeding, delicate brushwork, wet on wet technique, fine art watercolor",
            "description": "Watercolor painting style"
        },
        "oil_painting": {
            "prompt": "A classical oil painting, rich deep colors, visible textured brushstrokes, Renaissance technique, dramatic chiaroscuro lighting, museum quality fine art",
            "description": "Classical oil painting style"
        },
        "sketch": {
            "prompt": "A detailed pencil sketch drawing, hand-drawn linework, artistic cross-hatching shading, graphite on paper texture, fine art illustration, professional sketch",
            "description": "Pencil sketch with detailed linework"
        },
        "anime": {
            "prompt": "Anime style illustration, vibrant saturated colors, clean crisp lines, cel shading, manga aesthetic, Japanese animation style, detailed expressive eyes, studio quality",
            "description": "Anime/manga illustration style"
        },
    }

    # Negative prompt to avoid common issues
    DEFAULT_NEGATIVE_PROMPT = "blurry, low quality, distorted, deformed, ugly, disfigured, bad anatomy, watermark, signature, text, worst quality, jpeg artifacts"

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

    def _call_replicate_http(self, model_version: str, input_params: Dict[str, Any]) -> Optional[str]:
        """
        Call Replicate API directly via HTTP.
        Avoids the replicate SDK which has Python 3.14 compatibility issues.
        """
        headers = {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "application/json",
        }

        # Create prediction
        create_url = "https://api.replicate.com/v1/predictions"
        payload = {
            "version": model_version.split(":")[-1] if ":" in model_version else model_version,
            "input": input_params,
        }

        print(f"Creating prediction...")
        response = requests.post(create_url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        prediction = response.json()

        prediction_id = prediction.get("id")
        if not prediction_id:
            raise RuntimeError(f"Failed to create prediction: {prediction}")

        # Poll for completion
        get_url = f"https://api.replicate.com/v1/predictions/{prediction_id}"
        max_wait = 300  # 5 minutes
        start_time = time.time()

        while time.time() - start_time < max_wait:
            response = requests.get(get_url, headers=headers, timeout=30)
            response.raise_for_status()
            prediction = response.json()

            status = prediction.get("status")
            print(f"Status: {status}")

            if status == "succeeded":
                output = prediction.get("output")
                if isinstance(output, list):
                    return output[0] if output else None
                return output

            if status == "failed":
                error = prediction.get("error", "Unknown error")
                raise RuntimeError(f"Prediction failed: {error}")

            if status == "canceled":
                raise RuntimeError("Prediction was canceled")

            time.sleep(2)  # Poll every 2 seconds

        raise RuntimeError("Prediction timed out")

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
        custom_prompt: Optional[str] = None,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 20,
        controlnet_conditioning_scale: float = 1.0,
        negative_prompt: Optional[str] = None,
        **kwargs
    ) -> StyleResult:
        """
        Apply style using Replicate API with ControlNet.

        Args:
            image: Input PIL Image
            style: Style name from STYLE_MODELS or "custom"
            custom_prompt: Custom prompt (required if style="custom")
            guidance_scale: How closely to follow the prompt (1-20)
            num_inference_steps: Number of denoising steps (10-50)
            controlnet_conditioning_scale: How much to preserve structure (0.5-2.0)
            negative_prompt: What to avoid in the output
            **kwargs: Additional model-specific parameters

        Returns:
            StyleResult with styled image
        """
        import requests

        start_time = time.time()
        self._validate_api_credentials()

        # Handle custom style
        if style == "custom":
            if not custom_prompt:
                raise ValueError("custom_prompt is required when style is 'custom'")
            prompt = custom_prompt
        elif style in self.STYLE_MODELS:
            style_config = self.STYLE_MODELS[style]
            prompt = custom_prompt or style_config["prompt"]
        else:
            raise ValueError(f"Unknown style: {style}. Available: {self.get_available_styles() + ['custom']}")

        model_id = self.DEFAULT_MODEL

        # Prepare image - resize if too large (max 1024px on longest side)
        max_dim = 1024
        if max(image.size) > max_dim:
            ratio = max_dim / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)

        image_uri = self._image_to_data_uri(image)

        # 5 discrete levels for style strength based on control_strength slider
        # Slider range: 0.5 (Loose) to 1.5 (Exact)
        # Level 1: Most style change | Level 5: Preserve original most
        if controlnet_conditioning_scale <= 0.7:
            prompt_strength = 0.55  # Level 1 - Heavy style
        elif controlnet_conditioning_scale <= 0.9:
            prompt_strength = 0.47  # Level 2
        elif controlnet_conditioning_scale <= 1.1:
            prompt_strength = 0.40  # Level 3 - Balanced (default)
        elif controlnet_conditioning_scale <= 1.3:
            prompt_strength = 0.32  # Level 4
        else:
            prompt_strength = 0.25  # Level 5 - Preserve original

        print(f"Style level: control={controlnet_conditioning_scale:.1f} -> prompt_strength={prompt_strength}")

        # Build input params for SDXL img2img
        input_params = {
            "image": image_uri,
            "prompt": prompt,
            "negative_prompt": negative_prompt or self.DEFAULT_NEGATIVE_PROMPT,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "prompt_strength": prompt_strength,  # How much to change from original (0.0-1.0)
            "num_outputs": 1,
            "scheduler": "K_EULER",
        }

        # Call Replicate API via HTTP (avoiding SDK compatibility issues)
        print(f"Calling Replicate API with model: {model_id}")
        print(f"Prompt: {prompt[:100]}...")

        output_url = self._call_replicate_http(model_id, input_params)

        if not output_url:
            raise RuntimeError("Replicate API returned no output")

        # Download result image
        response = requests.get(output_url, timeout=30)
        response.raise_for_status()
        result_image = Image.open(io.BytesIO(response.content))

        # Create metadata
        metadata = {
            'style': style,
            'model': model_id,
            'prompt': prompt,
            'guidance_scale': guidance_scale,
            'num_inference_steps': num_inference_steps,
            'controlnet_conditioning_scale': controlnet_conditioning_scale,
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
