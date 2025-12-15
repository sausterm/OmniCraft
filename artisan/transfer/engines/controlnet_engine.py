"""
ControlNet style transfer engine for local processing
Optimized for Apple Silicon (Mac) with MPS support
Supports custom text prompts for unlimited style possibilities
"""

import torch
from typing import List, Optional, Dict, Any, Literal
from PIL import Image
import numpy as np
import time
import cv2

try:
    from diffusers import (
        StableDiffusionControlNetPipeline,
        ControlNetModel,
        UniPCMultistepScheduler
    )
    from controlnet_aux import (
        CannyDetector,
        HEDdetector,
        OpenposeDetector
    )
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False

from ..base import LocalStyleEngine, StyleResult


ControlMode = Literal["canny", "hed", "depth", "openpose", "mlsd"]


class ControlNetEngine(LocalStyleEngine):
    """
    Local ControlNet-based style transfer engine.

    Uses Stable Diffusion with ControlNet to apply any text-described style
    while preserving the structure and composition of the input image.

    Features:
    - Custom text prompts (unlimited styles)
    - Multiple control modes (edges, depth, pose, etc.)
    - Optimized for Apple Silicon (MPS) and CUDA
    - Maintains subject identity and composition

    Requires:
        pip install diffusers transformers accelerate controlnet_aux
    """

    # Default ControlNet model paths (Hugging Face)
    CONTROLNET_MODELS = {
        "canny": "lllyasviel/control_v11p_sd15_canny",
        "hed": "lllyasviel/control_v11p_sd15_softedge",
        "depth": "lllyasviel/control_v11f1p_sd15_depth",
        "openpose": "lllyasviel/control_v11p_sd15_openpose",
        "mlsd": "lllyasviel/control_v11p_sd15_mlsd",
    }

    # Base Stable Diffusion model
    SD_MODEL = "runwayml/stable-diffusion-v1-5"

    def __init__(
        self,
        device: Optional[str] = None,
        control_mode: ControlMode = "canny",
        model_precision: Literal["fp32", "fp16"] = "fp16"
    ):
        """
        Initialize ControlNet engine.

        Args:
            device: Device to run on ("mps", "cuda", "cpu"). Auto-detects if None.
            control_mode: Type of control signal to use (default: "canny" for edges)
            model_precision: Model precision ("fp16" for faster, "fp32" for quality)
        """
        if not DIFFUSERS_AVAILABLE:
            raise ImportError(
                "Required packages not installed.\n"
                "Install with: pip install diffusers transformers accelerate controlnet_aux opencv-python"
            )

        super().__init__(
            name="controlnet",
            description="Local ControlNet-based style transfer with custom text prompts"
        )

        # Auto-detect device
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        self.device = device
        self.control_mode = control_mode
        self.model_precision = model_precision
        self.dtype = torch.float16 if model_precision == "fp16" and device != "mps" else torch.float32

        # Lazy loading - models loaded on first use
        self._pipeline = None
        self._preprocessors = {}

        print(f"ControlNetEngine initialized on {device} ({model_precision})")

    def _load_pipeline(self):
        """Lazy load the ControlNet pipeline"""
        if self._pipeline is not None:
            return

        print(f"Loading ControlNet pipeline ({self.control_mode})...")

        # Load ControlNet model
        controlnet = ControlNetModel.from_pretrained(
            self.CONTROLNET_MODELS[self.control_mode],
            torch_dtype=self.dtype
        )

        # Load Stable Diffusion pipeline with ControlNet
        self._pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            self.SD_MODEL,
            controlnet=controlnet,
            torch_dtype=self.dtype,
            safety_checker=None,  # Disable for speed
        )

        # Optimize for device
        if self.device == "mps":
            # Apple Silicon optimizations
            self._pipeline = self._pipeline.to(self.device)
            # Enable attention slicing for memory efficiency
            self._pipeline.enable_attention_slicing()
        elif self.device == "cuda":
            # NVIDIA GPU optimizations
            self._pipeline = self._pipeline.to(self.device)
            self._pipeline.enable_attention_slicing()
            # Use faster scheduler
            self._pipeline.scheduler = UniPCMultistepScheduler.from_config(
                self._pipeline.scheduler.config
            )
        else:
            # CPU
            self._pipeline = self._pipeline.to(self.device)

        print(f"Pipeline loaded on {self.device}")

    def _get_preprocessor(self, mode: ControlMode):
        """Get or create preprocessor for control mode"""
        if mode not in self._preprocessors:
            if mode == "canny":
                self._preprocessors[mode] = CannyDetector()
            elif mode == "hed":
                self._preprocessors[mode] = HEDdetector.from_pretrained('lllyasviel/Annotators')
            elif mode == "openpose":
                self._preprocessors[mode] = OpenposeDetector.from_pretrained('lllyasviel/Annotators')
            # depth and mlsd don't need separate preprocessors in this implementation

        return self._preprocessors.get(mode)

    def _preprocess_canny(self, image: Image.Image, low_threshold: int = 100, high_threshold: int = 200) -> Image.Image:
        """Extract Canny edges from image"""
        # Convert to numpy
        img_np = np.array(image)

        # Convert to grayscale
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        # Apply Canny edge detection
        edges = cv2.Canny(gray, low_threshold, high_threshold)

        # Convert back to RGB (3 channels)
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

        return Image.fromarray(edges_rgb)

    def _preprocess_hed(self, image: Image.Image) -> Image.Image:
        """Extract HED (soft edges) from image"""
        preprocessor = self._get_preprocessor("hed")
        return preprocessor(image)

    def _preprocess_image(self, image: Image.Image, mode: ControlMode = None) -> Image.Image:
        """
        Preprocess image to create control signal.

        Args:
            image: Input image
            mode: Control mode (uses self.control_mode if None)

        Returns:
            Preprocessed control image
        """
        mode = mode or self.control_mode

        if mode == "canny":
            return self._preprocess_canny(image)
        elif mode == "hed":
            return self._preprocess_hed(image)
        elif mode == "openpose":
            preprocessor = self._get_preprocessor("openpose")
            return preprocessor(image)
        else:
            # For depth and mlsd, use canny as fallback for now
            return self._preprocess_canny(image)

    def get_available_styles(self) -> List[str]:
        """
        ControlNet supports unlimited styles via text prompts.
        This returns some common presets.
        """
        return [
            "pop_art",
            "britto_style",
            "van_gogh",
            "picasso_cubist",
            "anime",
            "watercolor",
            "oil_painting",
            "sketch",
            "custom"  # User provides their own prompt
        ]

    def apply_style(
        self,
        image: Image.Image,
        style: str,
        custom_prompt: Optional[str] = None,
        negative_prompt: str = "blurry, low quality, distorted, deformed, ugly",
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        controlnet_conditioning_scale: float = 1.0,
        control_mode: Optional[ControlMode] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> StyleResult:
        """
        Apply style to image using custom text prompt.

        Args:
            image: Input PIL Image
            style: Style name (or "custom" for custom_prompt)
            custom_prompt: Custom style description (overrides style presets)
            negative_prompt: What to avoid in the output
            num_inference_steps: Number of denoising steps (20-50, higher=better quality)
            guidance_scale: How closely to follow prompt (7-15)
            controlnet_conditioning_scale: How much to preserve structure (0.5-1.5)
            control_mode: Control type to use (overrides default)
            seed: Random seed for reproducibility
            **kwargs: Additional pipeline parameters

        Returns:
            StyleResult with styled image
        """
        start_time = time.time()

        # Load pipeline if not already loaded
        self._load_pipeline()

        # Use custom control mode if specified
        if control_mode and control_mode != self.control_mode:
            # Would need to reload pipeline with different ControlNet
            print(f"Warning: Requested control mode {control_mode} differs from loaded {self.control_mode}")
            print("Using loaded control mode. Reload engine to change.")
            control_mode = self.control_mode
        else:
            control_mode = self.control_mode

        # Build prompt
        if custom_prompt:
            prompt = custom_prompt
            style_used = "custom"
        else:
            # Use preset style prompts
            prompt = self._get_style_prompt(style)
            style_used = style

        # Preprocess image for control signal
        print(f"Preprocessing image ({control_mode})...")
        control_image = self._preprocess_image(image, control_mode)

        # Resize if needed (SD works best at 512x512 or multiples)
        target_size = self._get_optimal_size(image.size)
        if image.size != target_size:
            print(f"Resizing from {image.size} to {target_size}")
            image = image.resize(target_size, Image.Resampling.LANCZOS)
            control_image = control_image.resize(target_size, Image.Resampling.LANCZOS)

        # Set seed for reproducibility
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        # Run inference
        print(f"Generating styled image with prompt: '{prompt[:100]}...'")

        with torch.inference_mode():
            output = self._pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=control_image,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                generator=generator,
                **kwargs
            )

        result_image = output.images[0]

        # Create metadata
        metadata = {
            'style': style_used,
            'prompt': prompt,
            'negative_prompt': negative_prompt,
            'control_mode': control_mode,
            'num_inference_steps': num_inference_steps,
            'guidance_scale': guidance_scale,
            'controlnet_conditioning_scale': controlnet_conditioning_scale,
            'seed': seed,
            'device': self.device,
            'original_size': image.size,
            'final_size': result_image.size,
        }

        return self._create_result(result_image, style_used, metadata, start_time)

    def _get_style_prompt(self, style: str) -> str:
        """Get preset prompt for common styles"""
        prompts = {
            "pop_art": (
                "vibrant pop art style, bold thick black outlines, geometric patterns, "
                "extremely bright saturated colors, flat 2D aesthetic, playful optimistic mood, "
                "high contrast, decorative patterns with hearts dots stripes, "
                "contemporary Brazilian pop art with neo-cubist geometric patterns"
            ),
            "britto_style": (
                "Romero Britto art style, vibrant pop art, bold thick black outlines defining all shapes, "
                "geometric patterns and cubist influences, extremely bright saturated primary colors, "
                "decorative pattern fills with hearts dots stripes diamonds flowers swirls checkered patterns, "
                "flat 2D aesthetic no shading, playful optimistic mood, high contrast, "
                "modern pop art meets Cuban folk art meets children's book illustration, "
                "joyful energetic highly decorative"
            ),
            "van_gogh": (
                "painting in the style of Vincent van Gogh, post-impressionist, "
                "swirling brushstrokes, impasto texture, vivid colors, "
                "starry night style, expressive emotional"
            ),
            "picasso_cubist": (
                "cubist painting in the style of Pablo Picasso, geometric shapes, "
                "multiple perspectives, fragmented forms, bold lines, analytical cubism"
            ),
            "anime": (
                "anime style illustration, vibrant colors, clean lines, "
                "cel shading, manga style, Japanese animation"
            ),
            "watercolor": (
                "watercolor painting, soft edges, translucent colors, "
                "bleeding colors, artistic, delicate brushstrokes"
            ),
            "oil_painting": (
                "oil painting, rich colors, visible brushstrokes, "
                "classical painting technique, textured canvas"
            ),
            "sketch": (
                "pencil sketch, hand-drawn, detailed linework, "
                "cross-hatching, artistic drawing"
            ),
        }

        return prompts.get(style, prompts["pop_art"])

    def _get_optimal_size(self, current_size: tuple) -> tuple:
        """
        Get optimal size for Stable Diffusion (multiple of 64).

        Args:
            current_size: (width, height)

        Returns:
            Optimal (width, height) maintaining aspect ratio
        """
        width, height = current_size

        # Find the closest multiple of 64
        def round_to_64(x):
            return ((x + 31) // 64) * 64

        # Calculate aspect ratio
        aspect_ratio = width / height

        # Target around 512 on the shorter side
        if width < height:
            new_width = 512
            new_height = round_to_64(int(512 / aspect_ratio))
        else:
            new_height = 512
            new_width = round_to_64(int(512 * aspect_ratio))

        # Clamp to reasonable sizes (256-1024)
        new_width = max(256, min(1024, new_width))
        new_height = max(256, min(1024, new_height))

        return (new_width, new_height)

    def save_control_image(self, image: Image.Image, output_path: str):
        """
        Save the preprocessed control image (useful for debugging).

        Args:
            image: Input image
            output_path: Where to save control image
        """
        control_image = self._preprocess_image(image)
        control_image.save(output_path)
        print(f"Control image saved to {output_path}")
