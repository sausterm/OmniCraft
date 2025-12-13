#!/usr/bin/env python3
"""
IP-Adapter + ControlNet Britto Style Transfer
Uses actual Britto examples as style reference while preserving subject structure
"""

import torch
from PIL import Image
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler
)
from diffusers.utils import load_image
import cv2
import numpy as np

# Check for IP-Adapter
try:
    from ip_adapter import IPAdapter
    IP_ADAPTER_AVAILABLE = True
except ImportError:
    IP_ADAPTER_AVAILABLE = False

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def get_canny_image(image, low_threshold=100, high_threshold=200):
    """Extract canny edges for ControlNet"""
    image = np.array(image)
    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    return Image.fromarray(image)

def apply_ip_adapter_style(
    content_image_path,
    style_image_path,
    output_path,
    prompt="",
    negative_prompt="blurry, low quality, distorted, photorealistic, photograph",
    num_steps=40,
    guidance_scale=7.5,
    controlnet_scale=1.2,
    ip_adapter_scale=0.8
):
    """
    Apply Britto style using IP-Adapter + ControlNet

    Args:
        content_image_path: Your dogs photo
        style_image_path: Britto example image
        output_path: Where to save result
        prompt: Additional text prompt (optional)
        ip_adapter_scale: How strongly to apply style (0-1, default 0.8)
        controlnet_scale: How strongly to preserve structure (default 1.2)
    """

    print("IP-Adapter + ControlNet Britto Style Transfer")
    print("="*60)

    # Check if IP-Adapter is available
    if not IP_ADAPTER_AVAILABLE:
        print("\nInstalling IP-Adapter...")
        import subprocess
        subprocess.run(["pip", "install", "-q", "ip-adapter"])
        from ip_adapter import IPAdapter

    # Load images
    print("\nLoading images...")
    content_image = load_image(content_image_path).resize((512, 512))
    style_image = load_image(style_image_path).resize((512, 512))

    # Get canny edges for ControlNet
    print("Extracting structure (canny edges)...")
    canny_image = get_canny_image(content_image)

    # Load ControlNet
    print("Loading ControlNet...")
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_canny",
        torch_dtype=torch.float32  # Use float32 for MPS
    )

    # Load Stable Diffusion pipeline
    print("Loading Stable Diffusion pipeline...")
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float32,
        safety_checker=None
    )

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()

    # Initialize IP-Adapter
    print("Loading IP-Adapter...")
    ip_model = IPAdapter(pipe, "h94/IP-Adapter", "ip-adapter_sd15.bin", device=device)

    # Build prompt
    full_prompt = "Romero Britto pop art style, thick black outlines, geometric shapes, bright vibrant colors, flat 2D, cheerful"
    if prompt:
        full_prompt = f"{prompt}, {full_prompt}"

    print(f"\nGenerating with IP-Adapter style transfer...")
    print(f"Style reference: {style_image_path}")
    print(f"Structure guidance: {controlnet_scale}")
    print(f"Style strength: {ip_adapter_scale}")

    # Generate
    images = ip_model.generate(
        pil_image=style_image,  # Style reference
        prompt=full_prompt,
        negative_prompt=negative_prompt,
        num_samples=1,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        controlnet_conditioning_scale=controlnet_scale,
        image=canny_image,  # Structure reference
        scale=ip_adapter_scale  # IP-Adapter strength
    )

    result = images[0]

    # Save
    result.save(output_path)

    print(f"\n{'='*60}")
    print("Complete!")
    print(f"Saved to: {output_path}")
    print(f"{'='*60}\n")

    return result

if __name__ == '__main__':
    # Use your Britto dog example as style reference
    apply_ip_adapter_style(
        content_image_path='input/wilburderby1/input.png',
        style_image_path='input/britto_examples/Britto_Dog.png',
        output_path='output/wilburderby1_ip_adapter_britto.png',
        num_steps=40,
        controlnet_scale=1.2,  # Structure preservation
        ip_adapter_scale=0.8    # Style strength
    )
