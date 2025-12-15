#!/usr/bin/env python3
"""
Style Transfer Examples

Demonstrates various ways to use the Artisan style transfer engine
to apply custom artistic styles to images.
"""

import sys
from pathlib import Path
from PIL import Image

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from artisan.transfer.engines import ControlNetEngine


def example_1_preset_style():
    """Example 1: Use a preset style (Britto pop art)"""
    print("\n" + "="*60)
    print("Example 1: Preset Britto Style")
    print("="*60)

    # Initialize engine
    engine = ControlNetEngine(control_mode="canny")

    # Load image
    image = Image.open("input/my_image/my_image.png")

    # Apply preset style
    result = engine.apply_style(
        image=image,
        style="britto_style",
        num_inference_steps=30,
        guidance_scale=7.5,
    )

    # Save result
    result.save("output/my_image_britto.png")

    print(f"Saved to: output/my_image_britto.png")
    print(f"Processing time: {result.processing_time:.2f}s")


def example_2_custom_prompt():
    """Example 2: Use a custom text prompt"""
    print("\n" + "="*60)
    print("Example 2: Custom Prompt")
    print("="*60)

    engine = ControlNetEngine(control_mode="canny")

    image = Image.open("input/my_image/my_image.png")

    # Custom style description
    custom_prompt = (
        "vibrant pop art style with bold thick black outlines, "
        "geometric patterns, extremely bright saturated colors, "
        "decorative pattern fills with hearts dots stripes, "
        "flat 2D aesthetic, playful optimistic mood, high contrast"
    )

    result = engine.apply_style(
        image=image,
        style="custom",
        custom_prompt=custom_prompt,
        num_inference_steps=40,  # More steps for better quality
        guidance_scale=8.5,      # Higher guidance for style adherence
    )

    result.save("output/my_image_custom_pop_art.png")

    print(f"Saved to: output/my_image_custom_pop_art.png")
    print(f"Prompt: {custom_prompt[:80]}...")


def example_3_from_prompt_file():
    """Example 3: Load prompt from a text file"""
    print("\n" + "="*60)
    print("Example 3: Prompt from File")
    print("="*60)

    engine = ControlNetEngine(control_mode="canny")

    image = Image.open("input/my_image/my_image.png")

    # Load prompt from file
    prompt_file = Path(__file__).parent / "prompts" / "pop_art_geometric.txt"
    with open(prompt_file, 'r') as f:
        custom_prompt = f.read().strip()

    print(f"Loaded prompt from: {prompt_file}")

    result = engine.apply_style(
        image=image,
        style="custom",
        custom_prompt=custom_prompt,
        num_inference_steps=35,
        guidance_scale=8.0,
        controlnet_conditioning_scale=1.1,  # Stronger structure preservation
        seed=42  # For reproducibility
    )

    result.save("output/my_image_pop_art_geometric.png")

    print(f"Saved to: output/my_image_pop_art_geometric.png")


def example_4_batch_processing():
    """Example 4: Process multiple images with the same style"""
    print("\n" + "="*60)
    print("Example 4: Batch Processing")
    print("="*60)

    engine = ControlNetEngine(control_mode="canny")

    # Load images
    input_dir = Path("input")
    image_files = list(input_dir.glob("*/*.png")) + list(input_dir.glob("*/*.jpg"))

    if not image_files:
        print("No images found in input directory")
        return

    # Style to apply
    style_prompt = (
        "vibrant Romero Britto pop art style, bold black outlines, "
        "geometric patterns, bright saturated primary colors, "
        "decorative hearts dots stripes, flat 2D, joyful energetic"
    )

    for img_path in image_files[:3]:  # Process first 3 images
        print(f"\nProcessing: {img_path}")

        image = Image.open(img_path)

        result = engine.apply_style(
            image=image,
            style="custom",
            custom_prompt=style_prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
        )

        # Save with _pop_art suffix
        output_path = f"output/{img_path.stem}_pop_art.png"
        result.save(output_path)

        print(f"  Saved to: {output_path}")
        print(f"  Time: {result.processing_time:.2f}s")


def example_5_style_variations():
    """Example 5: Generate variations with different parameters"""
    print("\n" + "="*60)
    print("Example 5: Style Variations")
    print("="*60)

    engine = ControlNetEngine(control_mode="canny")

    image = Image.open("input/my_image/my_image.png")

    base_prompt = "vibrant pop art, bold outlines, geometric patterns, bright colors"

    # Generate variations with different parameters
    variations = [
        {"guidance": 5.0, "strength": 0.8, "name": "subtle"},
        {"guidance": 7.5, "strength": 1.0, "name": "balanced"},
        {"guidance": 10.0, "strength": 1.2, "name": "strong"},
    ]

    for var in variations:
        print(f"\nGenerating {var['name']} variation...")

        result = engine.apply_style(
            image=image,
            style="custom",
            custom_prompt=base_prompt,
            num_inference_steps=30,
            guidance_scale=var["guidance"],
            controlnet_conditioning_scale=var["strength"],
            seed=42  # Same seed for fair comparison
        )

        output_path = f"output/my_image_variation_{var['name']}.png"
        result.save(output_path)

        print(f"  Saved: {output_path}")
        print(f"  Guidance: {var['guidance']}, Strength: {var['strength']}")


def example_6_different_control_modes():
    """Example 6: Compare different control modes"""
    print("\n" + "="*60)
    print("Example 6: Different Control Modes")
    print("="*60)

    image = Image.open("input/my_image/my_image.png")

    prompt = "vibrant pop art style, bold outlines, bright colors"

    # Try different control modes
    control_modes = ["canny", "hed"]  # Add "openpose" for human figures

    for mode in control_modes:
        print(f"\nUsing control mode: {mode}")

        # Create engine with specific control mode
        engine = ControlNetEngine(control_mode=mode)

        result = engine.apply_style(
            image=image,
            style="custom",
            custom_prompt=prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
        )

        output_path = f"output/my_image_control_{mode}.png"
        result.save(output_path)

        print(f"  Saved: {output_path}")


def example_7_save_control_image():
    """Example 7: Save the preprocessed control image for inspection"""
    print("\n" + "="*60)
    print("Example 7: Save Control Image")
    print("="*60)

    engine = ControlNetEngine(control_mode="canny")

    image = Image.open("input/my_image/my_image.png")

    # Save control image to see what the model is using
    engine.save_control_image(image, "output/my_image_control_canny.png")

    print("Control image saved to: output/my_image_control_canny.png")
    print("This shows the edge detection that guides the style transfer")


def main():
    """Run all examples"""
    print("\n" + "="*60)
    print("Artisan Style Transfer Examples")
    print("="*60)

    # Create output directory
    Path("output").mkdir(exist_ok=True)

    # Run examples
    try:
        # Comment out any examples you don't want to run

        # example_1_preset_style()
        # example_2_custom_prompt()
        example_3_from_prompt_file()  # This uses your exact pop art prompt!
        # example_4_batch_processing()
        # example_5_style_variations()
        # example_6_different_control_modes()
        # example_7_save_control_image()

        print("\n" + "="*60)
        print("All examples complete!")
        print("="*60 + "\n")

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nMake sure you have:")
        print("1. Created input/my_image/my_image.png (or update the paths)")
        print("2. Installed dependencies: pip install -r requirements.txt")
        print("3. Have sufficient disk space for model downloads (~5GB)\n")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
