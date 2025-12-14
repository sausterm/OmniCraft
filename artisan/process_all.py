#!/usr/bin/env python3
"""
Batch processing script for paint-by-numbers generation.

IMPORTANT: Run with venv python for YOLO support:
    ./venv/bin/python process_all.py

Usage:
    ./venv/bin/python process_all.py                    # Process all input folders
    ./venv/bin/python process_all.py dogs fireweed      # Process specific folders

Input/Output mapping:
    input/{name}/*.png  -->  output/{name}/

No suffixes are added - clean folder name mapping.
"""

import os
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from artisan.generators.yolo_bob_ross_paint import YOLOBobRossPaint


def find_main_image(input_dir: Path) -> Path:
    """
    Find the main input image in a folder.
    Priority: exact folder name match > first png
    """
    folder_name = input_dir.name

    # Try exact match first (e.g., dogs/dogs.png)
    exact_match = input_dir / f"{folder_name}.png"
    if exact_match.exists():
        return exact_match

    # Try common variations
    for variant in [f"{folder_name}.jpg", f"{folder_name}.jpeg"]:
        path = input_dir / variant
        if path.exists():
            return path

    # Find first png (excluding variants like _compressed, _light, etc.)
    for f in sorted(input_dir.glob("*.png")):
        name = f.stem.lower()
        # Skip obvious variants
        if any(x in name for x in ['compressed', 'light', 'simplified', 'variant']):
            continue
        return f

    # Fall back to any png
    pngs = list(input_dir.glob("*.png"))
    if pngs:
        return pngs[0]

    return None


def process_folder(input_name: str, base_path: Path, model_size: str = "m"):
    """
    Process a single input folder.

    Args:
        input_name: Name of the input folder (e.g., "dogs")
        base_path: Base path containing input/output folders
        model_size: YOLO model size (n, s, m, l, x)
    """
    input_dir = base_path / "input" / input_name
    output_dir = base_path / "output" / input_name  # Clean name, no suffix

    if not input_dir.exists():
        print(f"[ERROR] Input folder not found: {input_dir}")
        return False

    # Find main image
    image_path = find_main_image(input_dir)
    if not image_path:
        print(f"[ERROR] No valid image found in: {input_dir}")
        return False

    print("=" * 70)
    print(f"PROCESSING: {input_name}")
    print("=" * 70)
    print(f"  Input:  {image_path}")
    print(f"  Output: {output_dir}")
    print()

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process
    try:
        painter = YOLOBobRossPaint(
            str(image_path),
            model_size=model_size,
            conf_threshold=0.2,
            substeps_per_region=4
        )
        painter.process()
        painter.save_all(str(output_dir))

        print()
        print(f"[SUCCESS] {input_name} completed!")
        return True

    except Exception as e:
        print(f"[ERROR] Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point."""
    base_path = Path(__file__).parent

    # Default folders to process
    default_folders = ["dogs", "fireweed", "aurora_maria"]

    # Parse command line
    if len(sys.argv) > 1:
        folders = sys.argv[1:]
    else:
        folders = default_folders

    print()
    print("=" * 70)
    print("ARTISAN BATCH PROCESSOR")
    print("=" * 70)
    print(f"Folders to process: {folders}")
    print()

    results = {}

    for folder in folders:
        success = process_folder(folder, base_path, model_size="m")
        results[folder] = success
        print()

    # Summary
    print()
    print("=" * 70)
    print("PROCESSING SUMMARY")
    print("=" * 70)

    for folder, success in results.items():
        status = "OK" if success else "FAILED"
        print(f"  {folder}: {status}")

    success_count = sum(1 for v in results.values() if v)
    print()
    print(f"Completed: {success_count}/{len(results)}")
    print()

    # Show output structure
    print("Output structure:")
    for folder in folders:
        output_dir = base_path / "output" / folder
        if output_dir.exists():
            print(f"  output/{folder}/")
            for item in sorted(output_dir.iterdir()):
                if item.is_dir():
                    count = len(list(item.glob("*")))
                    print(f"    {item.name}/ ({count} files)")
                else:
                    print(f"    {item.name}")


if __name__ == "__main__":
    main()
