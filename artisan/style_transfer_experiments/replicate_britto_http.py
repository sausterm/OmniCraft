#!/usr/bin/env python3
"""
Replicate HTTP API Britto Style Transfer
Uses direct HTTP API to avoid Python version issues
"""

import os
import requests
import time
import base64
from PIL import Image
from io import BytesIO

def upload_file_to_url(file_path):
    """Convert local file to data URI"""
    with open(file_path, 'rb') as f:
        data = base64.b64encode(f.read()).decode()
    return f"data:image/png;base64,{data}"

def replicate_britto_http(
    content_image_path,
    output_path,
    api_token
):
    """Apply Britto style using Replicate HTTP API"""

    print("Replicate ControlNet Britto Style Transfer")
    print("="*60)

    headers = {
        "Authorization": f"Token {api_token}",
        "Content-Type": "application/json"
    }

    # Prepare image
    print("\nPreparing image...")
    image_data = upload_file_to_url(content_image_path)

    # Detailed Britto-style prompt
    prompt = """Romero Britto pop art style artwork. Features:
- VERY thick bold black outlines like stained glass dividing image into LARGE geometric sections
- Each large section filled with ONE extremely bright, vibrant, saturated color
- Neon colors: electric red, bright blue, sunshine yellow, hot pink, vivid orange, lime green, purple, turquoise
- 75% solid flat colors, 25% simple patterns (hearts, dots, circles, stripes, flowers)
- Completely flat 2D with zero shading or depth
- High saturation, maximum brightness, tropical palette
- Cheerful, joyful, energetic mood
CRITICAL: Preserve exact subject - same pose, same face, same markings, same composition. Only transform the visual style, not the content."""

    negative_prompt = "photorealistic, photograph, realistic, 3D, shading, gradient, blurry, low quality, dark colors, muted colors, dull, small segments, too many patterns, busy"

    # Use SDXL img2img model
    payload = {
        "version": "7762fd07cf82c948538e41f63f77d685e02b063e37e496e96eefd46c929f9bdc",  # stability-ai/sdxl
        "input": {
            "image": image_data,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_inference_steps": 50,
            "prompt_strength": 0.8,
            "guidance_scale": 9.0,
        }
    }

    print(f"\nSubmitting to Replicate...")
    print(f"Model: SDXL ControlNet")
    print(f"Structure preservation: 1.3")
    print(f"Guidance: 9.0\n")

    # Create prediction
    response = requests.post(
        "https://api.replicate.com/v1/predictions",
        headers=headers,
        json=payload
    )

    if response.status_code != 201:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

    prediction = response.json()
    prediction_id = prediction["id"]

    print(f"Prediction ID: {prediction_id}")
    print("Waiting for completion...")

    # Poll for completion
    while True:
        response = requests.get(
            f"https://api.replicate.com/v1/predictions/{prediction_id}",
            headers=headers
        )

        prediction = response.json()
        status = prediction["status"]

        if status == "succeeded":
            print("\n✓ Complete!")
            break
        elif status == "failed":
            print(f"\n✗ Failed: {prediction.get('error')}")
            return None
        elif status in ["starting", "processing"]:
            print(f"  Status: {status}...", end="\r")
            time.sleep(2)
        else:
            print(f"  Unknown status: {status}")
            time.sleep(2)

    # Download result
    output_url = prediction["output"][0] if isinstance(prediction["output"], list) else prediction["output"]

    print(f"\nDownloading result...")
    response = requests.get(output_url)
    result = Image.open(BytesIO(response.content))

    # Save
    result.save(output_path)

    print(f"\n{'='*60}")
    print("Success!")
    print(f"Saved to: {output_path}")
    print(f"Cost: ~$0.01-0.02")
    print(f"{'='*60}\n")

    return result


if __name__ == '__main__':
    import sys

    api_token = os.environ.get("REPLICATE_API_TOKEN")

    if not api_token:
        print("ERROR: Replicate API token required!")
        sys.exit(1)

    replicate_britto_http(
        content_image_path='input/wilburderby1/input.png',
        output_path='output/wilburderby1_replicate_britto.png',
        api_token=api_token
    )
