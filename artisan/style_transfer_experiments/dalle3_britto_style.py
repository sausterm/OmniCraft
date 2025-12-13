#!/usr/bin/env python3
"""
DALL-E 3 Britto Style Transfer
Uses OpenAI's DALL-E 3 to recreate image in Britto style
"""

import os
from pathlib import Path
from openai import OpenAI
import base64
from PIL import Image
import requests
from io import BytesIO

def encode_image(image_path):
    """Encode image to base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def dalle3_britto_style(input_image_path, output_path, api_key=None):
    """
    Use DALL-E 3 to recreate image in Britto style

    Args:
        input_image_path: Path to input image
        output_path: Where to save result
        api_key: OpenAI API key (or set OPENAI_API_KEY env var)
    """

    # Initialize OpenAI client
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable or pass api_key parameter")

    client = OpenAI(api_key=api_key)

    # First, use GPT-4 Vision to describe the image
    print("Step 1: Analyzing your image with GPT-4 Vision...")

    base64_image = encode_image(input_image_path)

    vision_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe this image in detail, focusing on: the subjects (dogs), their poses, positions, colors, and any distinctive features. Be specific and detailed."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        max_tokens=500
    )

    description = vision_response.choices[0].message.content
    print(f"\nImage description:\n{description}\n")

    # Create Britto-style prompt
    britto_prompt = f"""Create an artwork in the signature style of Romero Britto featuring: {description}

IMPORTANT STYLE REQUIREMENTS:
- Romero Britto's iconic pop art style
- VERY thick, bold black outlines defining every shape (like stained glass)
- Break the subjects into geometric, angular sections
- Each section should be a SOLID FLAT COLOR (most sections)
- Only add decorative patterns (hearts, dots, stripes, flowers) to SOME sections (about 25% of sections)
- Use extremely bright, saturated colors: primary colors (red, blue, yellow), hot pink, orange, lime green, purple, turquoise
- Completely flat, 2D aesthetic with NO shading or depth
- High contrast, cheerful, optimistic mood
- Geometric background with vertical patterned sections
- Maintain the recognizable features and poses of the subjects
- More solid colors than patterns - balance is key
- Pattern types: hearts, dots, circles, stripes, flowers, swirls
- Contemporary Brazilian pop art aesthetic

The overall effect should be joyful, energetic, and immediately recognizable as Romero Britto's style."""

    print("Step 2: Generating Britto-style artwork with DALL-E 3...")
    print(f"\nPrompt:\n{britto_prompt[:200]}...\n")

    # Generate with DALL-E 3
    response = client.images.generate(
        model="dall-e-3",
        prompt=britto_prompt,
        size="1024x1024",
        quality="hd",  # Use HD quality for best results
        n=1,
    )

    # Get the generated image URL
    image_url = response.data[0].url

    print("Step 3: Downloading result...")

    # Download the image
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, quality=95)

    print(f"\n{'='*60}")
    print("Success!")
    print(f"{'='*60}")
    print(f"Saved to: {output_path}")
    print(f"Original description: {description[:100]}...")
    print(f"{'='*60}\n")

    return img

if __name__ == '__main__':
    import sys

    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OpenAI API key not found!")
        print("\nPlease set your API key:")
        print("  export OPENAI_API_KEY='your-key-here'")
        print("\nOr pass it as a parameter in the code.")
        sys.exit(1)

    print("DALL-E 3 Britto Style Transfer")
    print("="*60)

    dalle3_britto_style(
        'input/wilburderby1/input.png',
        'output/wilburderby1_dalle3_britto.png'
    )
