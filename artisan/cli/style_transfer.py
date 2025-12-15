#!/usr/bin/env python3
"""
Style Transfer CLI

Apply artistic styles to images using text prompts.
Supports local processing (ControlNet) and cloud APIs (Replicate).

Examples:
    # Use preset style
    python cli/style_transfer.py input.jpg output.jpg --style britto_style

    # Use custom text prompt
    python cli/style_transfer.py input.jpg output.jpg --prompt "vibrant pop art with bold outlines"

    # Use custom prompt from file
    python cli/style_transfer.py input.jpg output.jpg --prompt-file my_style.txt

    # Advanced options
    python cli/style_transfer.py input.jpg output.jpg --prompt "anime style" \\
        --steps 50 --guidance 8.5 --seed 42

    # Use Replicate API (requires REPLICATE_API_TOKEN)
    python cli/style_transfer.py input.jpg output.jpg --engine replicate --style van_gogh
"""

import argparse
import sys
from pathlib import Path
from PIL import Image

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from artisan.transfer.engines import ControlNetEngine, ReplicateEngine, REPLICATE_AVAILABLE


def load_prompt_from_file(filepath: str) -> str:
    """Load prompt text from a file"""
    with open(filepath, 'r') as f:
        return f.read().strip()


def main():
    parser = argparse.ArgumentParser(
        description='Apply artistic styles to images using text prompts',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Required arguments
    parser.add_argument('input', type=str, help='Input image path')
    parser.add_argument('output', type=str, help='Output image path')

    # Style specification (choose one)
    style_group = parser.add_mutually_exclusive_group(required=True)
    style_group.add_argument(
        '--style', '-s',
        type=str,
        help='Preset style name (britto_style, pop_art, van_gogh, anime, etc.)'
    )
    style_group.add_argument(
        '--prompt', '-p',
        type=str,
        help='Custom style description text'
    )
    style_group.add_argument(
        '--prompt-file', '-pf',
        type=str,
        help='Path to text file containing style description'
    )

    # Engine selection
    parser.add_argument(
        '--engine', '-e',
        type=str,
        choices=['controlnet', 'replicate', 'auto'],
        default='auto',
        help='Style transfer engine to use (default: auto)'
    )

    # ControlNet-specific options
    parser.add_argument(
        '--control-mode',
        type=str,
        choices=['canny', 'hed', 'depth', 'openpose', 'mlsd'],
        default='canny',
        help='Control signal type for ControlNet (default: canny edges)'
    )

    # Generation parameters
    parser.add_argument(
        '--steps',
        type=int,
        default=30,
        help='Number of inference steps (20-50, default: 30)'
    )
    parser.add_argument(
        '--guidance',
        type=float,
        default=7.5,
        help='Guidance scale - how closely to follow prompt (7-15, default: 7.5)'
    )
    parser.add_argument(
        '--control-strength',
        type=float,
        default=1.0,
        help='How much to preserve original structure (0.5-1.5, default: 1.0)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility (optional)'
    )
    parser.add_argument(
        '--negative-prompt',
        type=str,
        default='blurry, low quality, distorted, deformed, ugly, bad anatomy',
        help='Negative prompt - what to avoid (default: quality issues)'
    )

    # Output options
    parser.add_argument(
        '--save-control',
        action='store_true',
        help='Save the control image (preprocessed edges/depth/etc.)'
    )
    parser.add_argument(
        '--quality',
        type=int,
        default=95,
        help='JPEG quality (1-100, default: 95)'
    )

    args = parser.parse_args()

    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file '{args.input}' not found!")
        return 1

    # Load image
    print(f"Loading image from {args.input}...")
    try:
        image = Image.open(args.input)
        if image.mode != 'RGB':
            image = image.convert('RGB')
    except Exception as e:
        print(f"Error loading image: {e}")
        return 1

    # Determine style/prompt
    if args.style:
        style = args.style
        custom_prompt = None
        print(f"Using preset style: {style}")
    elif args.prompt:
        style = "custom"
        custom_prompt = args.prompt
        print(f"Using custom prompt: {custom_prompt[:100]}...")
    else:  # args.prompt_file
        style = "custom"
        try:
            custom_prompt = load_prompt_from_file(args.prompt_file)
            print(f"Loaded prompt from {args.prompt_file}")
            print(f"Prompt: {custom_prompt[:100]}...")
        except Exception as e:
            print(f"Error loading prompt file: {e}")
            return 1

    # Select engine
    engine_name = args.engine
    if engine_name == 'auto':
        # Default to ControlNet for local processing
        engine_name = 'controlnet'

    # Initialize engine
    print(f"\nInitializing {engine_name} engine...")
    try:
        if engine_name == 'controlnet':
            engine = ControlNetEngine(
                control_mode=args.control_mode
            )
        elif engine_name == 'replicate':
            if not REPLICATE_AVAILABLE:
                print("Error: Replicate engine not available. Install with: pip install replicate")
                return 1
            engine = ReplicateEngine()
        else:
            print(f"Error: Unknown engine '{engine_name}'")
            return 1
    except Exception as e:
        print(f"Error initializing engine: {e}")
        return 1

    # Apply style transfer
    print(f"\nApplying style transfer...")
    try:
        if engine_name == 'controlnet':
            result = engine.apply_style(
                image=image,
                style=style,
                custom_prompt=custom_prompt,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                controlnet_conditioning_scale=args.control_strength,
                negative_prompt=args.negative_prompt,
                seed=args.seed,
            )
        else:  # replicate
            result = engine.apply_style(
                image=image,
                style=style,
                prompt_override=custom_prompt,
                strength=args.control_strength,
                guidance_scale=args.guidance,
            )
    except Exception as e:
        print(f"Error during style transfer: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Save result
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving result to {args.output}...")
    try:
        if output_path.suffix.lower() in ['.jpg', '.jpeg']:
            result.save(args.output, quality=args.quality)
        else:
            result.save(args.output)
    except Exception as e:
        print(f"Error saving result: {e}")
        return 1

    # Save control image if requested
    if args.save_control and engine_name == 'controlnet':
        control_path = output_path.parent / f"{output_path.stem}_control{output_path.suffix}"
        print(f"Saving control image to {control_path}...")
        engine.save_control_image(image, str(control_path))

    # Print summary
    print(f"\n{'='*60}")
    print("Style Transfer Complete!")
    print(f"{'='*60}")
    print(f"Input:           {args.input}")
    print(f"Output:          {args.output}")
    print(f"Engine:          {result.engine_name}")
    print(f"Style:           {result.style_name}")
    if custom_prompt:
        print(f"Custom Prompt:   {custom_prompt[:80]}...")
    print(f"Processing Time: {result.processing_time:.2f} seconds")
    print(f"Original Size:   {result.metadata.get('original_size', 'N/A')}")
    print(f"Final Size:      {result.metadata.get('final_size', 'N/A')}")

    if engine_name == 'controlnet':
        print(f"\nControlNet Parameters:")
        print(f"  Control Mode:  {result.metadata.get('control_mode', 'N/A')}")
        print(f"  Steps:         {result.metadata.get('num_inference_steps', 'N/A')}")
        print(f"  Guidance:      {result.metadata.get('guidance_scale', 'N/A')}")
        print(f"  Strength:      {result.metadata.get('controlnet_conditioning_scale', 'N/A')}")
        if args.seed:
            print(f"  Seed:          {args.seed}")

    print(f"{'='*60}\n")

    return 0


if __name__ == '__main__':
    sys.exit(main())
