#!/usr/bin/env python3
"""
Replicate API Britto Style Transfer
Uses IP-Adapter + ControlNet on Replicate for perfect style transfer
"""

import os
import replicate
from PIL import Image
import requests
from io import BytesIO

def replicate_britto_style(
    content_image_path,
    style_image_path,
    output_path,
    api_token=None
):
    """
    Apply Britto style using Replicate's IP-Adapter + ControlNet

    Args:
        content_image_path: Your dogs photo
        style_image_path: Britto example image
        output_path: Where to save result
        api_token: Replicate API token (or set REPLICATE_API_TOKEN env var)
    """

    print("Replicate IP-Adapter + ControlNet Britto Style Transfer")
    print("="*60)

    # Set API token
    api_token = api_token or os.environ.get("REPLICATE_API_TOKEN")
    if not api_token:
        raise ValueError(
            "Replicate API token required.\n"
            "Get one at: https://replicate.com/account/api-tokens\n"
            "Then: export REPLICATE_API_TOKEN='your-token'"
        )

    os.environ["REPLICATE_API_TOKEN"] = api_token

    # Open and upload images
    print("\nPreparing images...")

    # Detailed prompt for Britto style
    prompt = """Romero Britto pop art style. Transform this image with:
- VERY thick black outlines dividing the image into LARGE geometric sections like stained glass
- Each large section filled with ONE extremely bright, vibrant, saturated color
- Neon colors: electric red, bright blue, sunshine yellow, hot pink, vivid orange, lime green, purple, turquoise
- Mostly SOLID flat colors in each section (75%)
- Only SIMPLE decorative patterns (hearts, dots, circles, stripes, flowers) in SOME sections (25%)
- Completely flat 2D with zero shading
- High saturation, maximum brightness
- Cheerful, joyful, energetic mood
CRITICAL: Preserve the exact subjects, same poses, same faces, same markings. Transform style only, not content."""

    negative_prompt = "photorealistic, photograph, realistic, 3D, shading, gradient, blurry, low quality, dark colors, muted colors, small segments, too many patterns"

    print(f"\nContent image: {content_image_path}")
    print(f"Style reference: {style_image_path}")
    print("\nGenerating with Replicate...")
    print("This uses IP-Adapter to learn from your Britto example")
    print("Cost: ~$0.01-0.02 per image\n")

    # Use Replicate's InstantID or IP-Adapter model
    # This model supports style reference images + structure preservation
    output = replicate.run(
        "tencentarc/photomaker:ddfc2b08d209f9fa8c1eca692712918bd449f695dabb4a958da31802a9570fe4",
        input={
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "input_image": open(content_image_path, "rb"),
            "style_image": open(style_image_path, "rb"),
            "num_outputs": 1,
            "num_inference_steps": 50,
            "guidance_scale": 8.0,
            "style_strength": 0.8,  # How much to apply style
            "controlnet_strength": 1.2,  # How much to preserve structure
        }
    )

    # Download result
    print("Downloading result...")
    if isinstance(output, list):
        image_url = output[0]
    else:
        image_url = output

    response = requests.get(image_url)
    result = Image.open(BytesIO(response.content))

    # Save
    result.save(output_path)

    print(f"\n{'='*60}")
    print("Complete!")
    print(f"Saved to: {output_path}")
    print(f"{'='*60}\n")

    return result


if __name__ == '__main__':
    import sys

    # Check for API token
    if not os.environ.get("REPLICATE_API_TOKEN"):
        print("ERROR: Replicate API token not found!")
        print("\nGet your token:")
        print("1. Go to: https://replicate.com/account/api-tokens")
        print("2. Copy your token")
        print("3. Run: export REPLICATE_API_TOKEN='your-token'")
        print("\nOr pass it as a parameter in the code.")
        sys.exit(1)

    try:
        replicate_britto_style(
            content_image_path='input/wilburderby1/input.png',
            style_image_path='input/britto_examples/Britto_Dog.png',
            output_path='output/wilburderby1_replicate_britto.png'
        )
    except Exception as e:
        print(f"\nError: {e}")
        print("\nTrying alternative model...")

        # Fallback: Try SDXL with ControlNet + custom prompt
        output = replicate.run(
            "lucataco/sdxl-controlnet:cef1c14e36fe8d9a2e95d5de72c8c06ec6f8f2bb0dde8c16f4cb0e67db8aaa7b",
            input={
                "image": open('input/wilburderby1/input.png', "rb"),
                "prompt": """Romero Britto pop art style artwork. VERY thick bold black outlines like stained glass. LARGE geometric sections. Each section ONE bright vibrant color (neon red, blue, yellow, hot pink, orange, lime green, purple, turquoise). Mostly solid flat colors. Some simple patterns (hearts, dots, stripes, flowers). Flat 2D, no shading. Maximum brightness and saturation. Preserve exact subject, same pose, same features.""",
                "negative_prompt": "photorealistic, photograph, 3D, shading, dark, muted colors, realistic",
                "num_inference_steps": 50,
                "controlnet_conditioning_scale": 1.2,
                "guidance_scale": 8.5,
            }
        )

        # Download
        if isinstance(output, list):
            image_url = output[0]
        else:
            image_url = output

        response = requests.get(image_url)
        result = Image.open(BytesIO(response.content))
        result.save('output/wilburderby1_replicate_britto.png')

        print(f"\nSaved to: output/wilburderby1_replicate_britto.png")
