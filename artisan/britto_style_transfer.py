#!/usr/bin/env python3
"""
Romero Britto Style Transfer
Converts images to Romero Britto's signature pop art style:
- Bright, vibrant colors
- Flat color areas (posterization)
- Bold black outlines
- Geometric patterns
- Simple shapes
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import argparse
from pathlib import Path
import random


def enhance_saturation(image, factor=1.8):
    """Boost color saturation for vibrant pop art look"""
    enhancer = ImageEnhance.Color(image)
    return enhancer.enhance(factor)


def posterize_image(image, levels=4):
    """Reduce colors to create flat color areas"""
    # Convert to numpy array
    img_array = np.array(image)

    # Posterize each channel
    bits = 8 - int(np.log2(256 // levels))
    posterized = np.left_shift(np.right_shift(img_array, bits), bits)

    return Image.fromarray(posterized.astype(np.uint8))


def brighten_and_boost_saturation(image):
    """Brighten image and boost saturation for Britto-style vivid colors"""
    img_array = np.array(image)

    # Convert to HSV
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV).astype(np.float32)

    # Boost saturation significantly
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.8 + 50, 0, 255)

    # Boost brightness/value significantly
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.5 + 60, 100, 255)

    # Convert back to RGB
    rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    return Image.fromarray(rgb)


def remove_background(image, threshold=30):
    """Detect and remove dark background using edge detection and contours"""
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # Find the main subject using edge detection and morphology
    edges = cv2.Canny(gray, 30, 100)
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Create mask for foreground (largest contours)
        mask = np.zeros(gray.shape, dtype=np.uint8)
        # Get the largest contours
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]
        cv2.drawContours(mask, contours, -1, 255, -1)

        # Dilate mask to include edges
        mask = cv2.dilate(mask, kernel, iterations=5)

        # Apply mask - make background white
        result = img_array.copy()
        result[mask == 0] = [255, 255, 255]

        return Image.fromarray(result)
    else:
        return image


def simplify_image(image, bilateral_d=15, sigma_color=80, sigma_space=80):
    """Simplify image while preserving edges using bilateral filter"""
    img_array = np.array(image)

    # Apply bilateral filter multiple times for stronger smoothing
    simplified = img_array.copy()
    for _ in range(3):
        simplified = cv2.bilateralFilter(simplified, bilateral_d, sigma_color, sigma_space)

    return Image.fromarray(simplified)


def quantize_to_britto_palette(image):
    """Quantize to Britto's signature bright color palette"""
    # Britto's signature color palette - ONLY bright, vibrant colors (no dark colors!)
    BRITTO_PALETTE = np.array([
        [255, 0, 0],      # Pure Red
        [0, 100, 255],    # Bright Blue
        [255, 255, 0],    # Pure Yellow
        [0, 255, 50],     # Bright Green
        [255, 140, 0],    # Orange
        [255, 20, 147],   # Hot Pink
        [0, 255, 255],    # Cyan
        [200, 0, 255],    # Bright Purple/Magenta
        [255, 255, 255],  # White
        [255, 100, 180],  # Pink
        [150, 255, 0],    # Lime Green
        [255, 180, 0],    # Amber
    ], dtype=np.float32)

    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Convert to numpy array
    img_array = np.array(image)
    h, w, c = img_array.shape

    # Reshape for processing
    pixels = img_array.reshape(-1, 3).astype(np.float32)

    # Find closest color in palette for each pixel
    distances = np.sqrt(((pixels[:, np.newaxis, :] - BRITTO_PALETTE[np.newaxis, :, :]) ** 2).sum(axis=2))
    closest_colors = np.argmin(distances, axis=1)

    # Map to palette colors
    quantized = BRITTO_PALETTE[closest_colors]
    quantized = quantized.reshape(h, w, 3).astype(np.uint8)

    return Image.fromarray(quantized)


def create_bold_outlines(image, thickness=8):
    """Create VERY bold black outlines characteristic of Britto's style"""
    # Convert to numpy array
    img_array = np.array(image)

    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # Detect edges using multiple methods to catch all boundaries
    # 1. Canny edges (very sensitive)
    edges_canny = cv2.Canny(gray, 10, 50)

    # 2. Color boundaries - detect where colors change
    # Create a mask for color changes
    edges_color = np.zeros_like(gray)
    for i in range(3):  # Check each RGB channel
        channel = img_array[:, :, i]
        # Detect horizontal and vertical color changes
        diff_h = np.abs(np.diff(channel.astype(np.int16), axis=0))
        diff_v = np.abs(np.diff(channel.astype(np.int16), axis=1))

        # Pad to match original size
        diff_h = np.vstack([diff_h, np.zeros((1, channel.shape[1]))])
        diff_v = np.hstack([diff_v, np.zeros((channel.shape[0], 1))])

        # Mark significant color changes
        edges_color = np.maximum(edges_color, (diff_h > 5).astype(np.uint8) * 255)
        edges_color = np.maximum(edges_color, (diff_v > 5).astype(np.uint8) * 255)

    # Combine edge detection methods
    edges = np.maximum(edges_canny, edges_color)

    # Dilate edges to make them VERY thick and prominent (Britto style)
    kernel = np.ones((thickness, thickness), np.uint8)
    thick_edges = cv2.dilate(edges, kernel, iterations=3)

    # Create a colored version with black outlines
    result = img_array.copy()
    result[thick_edges > 0] = [0, 0, 0]  # Black outlines

    return Image.fromarray(result)


def get_color_regions(image):
    """Segment image into distinct color regions"""
    img_array = np.array(image)
    h, w = img_array.shape[:2]

    # Create labels for each unique color region
    img_flat = img_array.reshape(-1, 3)
    unique_colors = np.unique(img_flat, axis=0, return_inverse=True)
    labels = unique_colors[1].reshape(h, w)

    return labels, unique_colors[0]


def draw_pattern_dots(draw, x, y, w, h, color, size):
    """Draw dots pattern in a region"""
    spacing = size * 6  # Increased spacing
    for dy in range(0, h, spacing):
        for dx in range(0, w, spacing):
            px, py = x + dx, y + dy
            draw.ellipse([px, py, px + size*2, py + size*2], fill=color)  # Larger dots


def draw_pattern_stripes(draw, x, y, w, h, color, size, diagonal=False):
    """Draw stripes pattern in a region"""
    spacing = size * 8  # Increased spacing
    stripe_width = size * 2  # Wider stripes
    if diagonal:
        # Diagonal stripes
        for offset in range(-max(w, h), max(w, h), spacing):
            draw.line([(x, y + offset), (x + w, y + offset - w)], fill=color, width=stripe_width)
    else:
        # Vertical stripes
        for dx in range(0, w, spacing):
            draw.rectangle([x + dx, y, x + dx + stripe_width, y + h], fill=color)


def draw_pattern_hearts(draw, x, y, w, h, color, size):
    """Draw hearts pattern in a region"""
    spacing = size * 8  # Increased spacing
    for dy in range(0, h, spacing):
        for dx in range(0, w, spacing):
            px, py = x + dx, y + dy
            # Simple heart using two circles and a triangle
            s = size * 2  # Larger hearts
            draw.ellipse([px, py, px + s, py + s], fill=color)
            draw.ellipse([px + s, py, px + s*2, py + s], fill=color)
            draw.polygon([(px, py + s), (px + s*2, py + s), (px + s, py + s*2)], fill=color)


def draw_pattern_flowers(draw, x, y, w, h, color, size):
    """Draw flower pattern in a region"""
    spacing = size * 10  # Increased spacing
    for dy in range(0, h, spacing):
        for dx in range(0, w, spacing):
            px, py = x + dx, y + dy
            s = size * 2  # Larger flowers
            # Center circle
            draw.ellipse([px + s, py + s, px + s*2, py + s*2], fill=color)
            # Petals
            for angle in [0, 60, 120, 180, 240, 300]:
                offset_x = int(s * 1.5 * np.cos(np.radians(angle)))
                offset_y = int(s * 1.5 * np.sin(np.radians(angle)))
                draw.ellipse([px + s + offset_x, py + s + offset_y,
                            px + s*2 + offset_x, py + s*2 + offset_y], fill=color)


def draw_pattern_spirals(draw, x, y, w, h, color, size):
    """Draw spiral pattern in a region"""
    spacing = size * 12  # Increased spacing
    for dy in range(0, h, spacing):
        for dx in range(0, w, spacing):
            px, py = x + dx, y + dy
            # Draw a simple spiral using arcs
            for r in range(size, size * 4, size):
                draw.arc([px - r, py - r, px + r, py + r], 0, 270, fill=color, width=3)


def draw_pattern_stars(draw, x, y, w, h, color, size):
    """Draw stars pattern in a region"""
    spacing = size * 10  # Increased spacing
    for dy in range(0, h, spacing):
        for dx in range(0, w, spacing):
            px, py = x + dx + size*3, y + dy + size*3
            # Simple 5-pointed star
            points = []
            for i in range(10):
                angle = i * 36 - 90
                r = size*3 if i % 2 == 0 else size*1.5  # Larger stars
                points.append((px + r * np.cos(np.radians(angle)),
                             py + r * np.sin(np.radians(angle))))
            draw.polygon(points, fill=color)


def draw_pattern_squiggles(draw, x, y, w, h, color, size):
    """Draw squiggly lines pattern in a region"""
    spacing = size * 8  # Increased spacing
    for dy in range(0, h, spacing):
        # Draw wavy lines
        points = []
        for dx in range(0, w, size*2):
            wave_y = y + dy + int(size * 2 * np.sin(dx / (size * 3)))
            points.append((x + dx, wave_y))
        if len(points) > 1:
            draw.line(points, fill=color, width=3)


def add_britto_patterns(image, intensity=0.7):
    """Add diverse region-specific patterns like Britto's work"""
    width, height = image.size
    img_array = np.array(image)

    # Get color regions
    labels, unique_colors = get_color_regions(image)

    # Create pattern overlay
    overlay = Image.new('RGBA', (width, height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    # Pattern scale based on image size
    base_size = max(4, min(width, height) // 50)  # Larger base size

    # Pattern types
    pattern_types = [
        draw_pattern_dots,
        draw_pattern_stripes,
        lambda draw, x, y, w, h, color, size: draw_pattern_stripes(draw, x, y, w, h, color, size, diagonal=True),
        draw_pattern_hearts,
        draw_pattern_flowers,
        draw_pattern_spirals,
        draw_pattern_stars,
        draw_pattern_squiggles,
    ]

    # Pattern colors (white and black work best on bright backgrounds)
    pattern_colors = [
        (255, 255, 255, int(intensity * 255)),  # White
        (0, 0, 0, int(intensity * 180)),        # Black (slightly more transparent)
        (255, 255, 255, int(intensity * 255)),  # White (more common)
    ]

    # Apply different pattern to each color region
    random.seed(42)  # For reproducibility
    for region_id in range(len(unique_colors)):
        # Get bounding box of this region
        mask = (labels == region_id)
        if not mask.any():
            continue

        # Skip white backgrounds (no patterns on white)
        region_color = unique_colors[region_id]
        if np.all(region_color > 240):  # Skip near-white regions
            continue

        ys, xs = np.where(mask)
        if len(ys) < 500:  # Skip small regions
            continue

        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()

        # Choose random pattern and color for this region
        pattern_func = random.choice(pattern_types)
        pattern_color = random.choice(pattern_colors)

        # Apply pattern to this region
        pattern_func(draw, x_min, y_min, x_max - x_min, y_max - y_min,
                    pattern_color, base_size)

    # Composite overlay
    image_rgba = image.convert('RGBA')
    result = Image.alpha_composite(image_rgba, overlay)
    return result.convert('RGB')


def apply_britto_style(input_path, output_path, outline_thickness=8,
                       add_patterns=True, pattern_intensity=0.7, remove_bg=True):
    """
    Apply Romero Britto style transfer to an image

    Args:
        input_path: Path to input image
        output_path: Path to save output image
        outline_thickness: Thickness of black outlines (default: 8)
        add_patterns: Whether to add pattern overlays (default: True)
        pattern_intensity: Intensity of pattern overlay (default: 0.7)
        remove_bg: Whether to remove dark backgrounds (default: True)
    """
    print(f"Loading image from {input_path}...")
    image = Image.open(input_path)

    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Step 0: Remove dark background and make it white
    if remove_bg:
        print("Removing dark background...")
        image = remove_background(image, threshold=50)

    # Step 1: Simplify the image aggressively to create large flat regions
    print("Simplifying image to create flat, cartoonish regions...")
    # Apply multiple rounds of bilateral filtering for more aggressive simplification
    image = simplify_image(image, bilateral_d=20, sigma_color=100, sigma_space=100)
    image = simplify_image(image, bilateral_d=15, sigma_color=80, sigma_space=80)

    # Step 2: Quantize to Britto's bright color palette
    print("Applying Britto's signature bright color palette...")
    image = quantize_to_britto_palette(image)

    # Step 3: Create VERY bold black outlines (signature Britto style)
    print("Adding BOLD black outlines...")
    image = create_bold_outlines(image, thickness=outline_thickness)

    # Step 4: Add diverse geometric patterns to each color region (optional)
    if add_patterns:
        print("Adding Britto-style geometric patterns to each region...")
        image = add_britto_patterns(image, intensity=pattern_intensity)

    # Save result
    print(f"Saving result to {output_path}...")
    image.save(output_path, quality=95)
    print("Done!")

    return image


def main():
    parser = argparse.ArgumentParser(description='Romero Britto Style Transfer')
    parser.add_argument('input', type=str, help='Input image path')
    parser.add_argument('output', type=str, help='Output image path')
    parser.add_argument('--outline-thickness', type=int, default=8,
                       help='Outline thickness (default: 8)')
    parser.add_argument('--no-patterns', action='store_true',
                       help='Disable pattern overlay')
    parser.add_argument('--pattern-intensity', type=float, default=0.7,
                       help='Pattern intensity 0-1 (default: 0.7)')

    args = parser.parse_args()

    # Validate input file exists
    if not Path(args.input).exists():
        print(f"Error: Input file '{args.input}' not found!")
        return 1

    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Apply style transfer
    apply_britto_style(
        args.input,
        args.output,
        outline_thickness=args.outline_thickness,
        add_patterns=not args.no_patterns,
        pattern_intensity=args.pattern_intensity
    )

    return 0


if __name__ == '__main__':
    exit(main())
