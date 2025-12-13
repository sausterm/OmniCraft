"""
Stability AI style transfer engine
Uses Stable Diffusion with ControlNet for style transfer
"""

import os
from typing import List, Optional
from PIL import Image
import time
import io
import base64

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

from ..base import APIStyleEngine, StyleResult


class StabilityEngine(APIStyleEngine):
    """
    Style transfer using Stability AI API.

    Uses Stable Diffusion with image-to-image or ControlNet
    for high-quality style transfer.

    Requires: Stability AI API key
    Set STABILITY_API_KEY environment variable
    """

    STYLES = {
        "realistic": "photorealistic, detailed, high quality",
        "anime": "anime style, vibrant colors, manga art",
        "fantasy": "fantasy art, magical, ethereal, highly detailed",
        "oil_painting": "oil painting, classical art, detailed brushstrokes",
        "sketch": "pencil sketch, line art, artistic drawing",
        "digital_art": "digital art, concept art, detailed illustration",
    }

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Stability AI engine.

        Args:
            api_key: Stability AI API key
        """
        api_key = api_key or os.environ.get("STABILITY_API_KEY")

        super().__init__(
            name="stability",
            description="Stability AI Stable Diffusion style transfer",
            api_key=api_key,
            api_base_url="https://api.stability.ai/v1"
        )

        # Register styles
        for style_name, style_prompt in self.STYLES.items():
            self.register_style(
                style_name,
                description=f"Transform to {style_name.replace('_', ' ')} style",
                base_prompt=style_prompt
            )

    def _validate_api_credentials(self) -> bool:
        """Validate API credentials"""
        if not self.api_key:
            raise ValueError(
                "Stability AI API key not provided. "
                "Set STABILITY_API_KEY environment variable"
            )
        return True

    def get_available_styles(self) -> List[str]:
        """Get available styles"""
        return list(self.STYLES.keys())

    def apply_style(
        self,
        image: Image.Image,
        style: str,
        prompt_override: Optional[str] = None,
        strength: float = 0.5,
        cfg_scale: float = 7.0,
        **kwargs
    ) -> StyleResult:
        """
        Apply style using Stability AI.

        Args:
            image: Input image
            style: Style to apply
            prompt_override: Custom style prompt
            strength: Transformation strength (0-1)
            cfg_scale: Classifier-free guidance scale

        Returns:
            StyleResult with styled image
        """
        start_time = time.time()
        self._validate_api_credentials()

        if style not in self.STYLES:
            raise ValueError(f"Unknown style: {style}")

        # Prepare image
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        buffered.seek(0)

        # Prepare request
        prompt = prompt_override or self.STYLES[style]

        url = f"{self.api_base_url}/generation/stable-diffusion-xl-1024-v1-0/image-to-image"

        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        files = {
            "init_image": buffered
        }

        data = {
            "text_prompts[0][text]": prompt,
            "text_prompts[0][weight]": 1,
            "cfg_scale": cfg_scale,
            "image_strength": strength,
            "samples": 1,
            "steps": 30,
        }

        # Make request
        response = requests.post(url, headers=headers, files=files, data=data)

        if response.status_code != 200:
            raise Exception(f"Stability AI API error: {response.text}")

        # Parse response
        result_json = response.json()
        image_data = base64.b64decode(result_json["artifacts"][0]["base64"])
        result_image = Image.open(io.BytesIO(image_data))

        metadata = {
            'style': style,
            'prompt': prompt,
            'strength': strength,
            'cfg_scale': cfg_scale,
            'original_size': image.size,
            'final_size': result_image.size,
            'api': 'stability'
        }

        return self._create_result(result_image, style, metadata, start_time)

    def get_cost_estimate(self, style: str, image_size: tuple) -> Optional[float]:
        """Estimate cost - roughly $0.01-0.03 per image"""
        return 0.02
