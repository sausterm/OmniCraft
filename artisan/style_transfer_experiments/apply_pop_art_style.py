#!/usr/bin/env python3
"""
Apply pop art style to the dogs image
Standalone script for immediate use
"""

import sys
sys.path.insert(0, '/Users/sloan/Documents/Projects/Omni/OmniCraft/artisan')

from PIL import Image
from style_transfer.engines.controlnet_engine import ControlNetEngine

# Load the custom prompt
with open('examples/prompts/pop_art_geometric.txt', 'r') as f:
    custom_prompt = f.read().strip()

print("Initializing ControlNet engine...")
engine = ControlNetEngine(control_mode="canny")

print("Loading image...")
image = Image.open("input/wilburderby1/input.png")

print(f"Applying pop art style with prompt:")
print(f"  {custom_prompt[:100]}...")
print()
print("This will take 3-5 minutes on your Mac Mini...")

result = engine.apply_style(
    image=image,
    style="custom",
    custom_prompt=custom_prompt,
    num_inference_steps=50,  # More steps for quality
    guidance_scale=9.5,  # Higher guidance for style
    controlnet_conditioning_scale=1.8,  # MUCH stronger structure preservation
    seed=42
)

print(f"\nSaving result...")
result.save("output/wilburderby1_pop_art_controlnet.png")

print(f"\n{'='*60}")
print("Style Transfer Complete!")
print(f"{'='*60}")
print(f"Output: output/wilburderby1_pop_art_controlnet.png")
print(f"Processing time: {result.processing_time:.2f} seconds")
print(f"Original size: {result.metadata.get('original_size')}")
print(f"Final size: {result.metadata.get('final_size')}")
print(f"{'='*60}\n")
