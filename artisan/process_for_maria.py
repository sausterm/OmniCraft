#!/usr/bin/env python3
"""
Process ForMaria images with optimal settings for each.

Run with: ./venv/bin/python process_for_maria.py
"""

import os
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from artisan.paint.generators.yolo_bob_ross_paint import YOLOBobRossPaint


# Image configurations with optimal settings
IMAGES = [
    {
        "name": "fisherman_river",
        "file": "fisherman_river.png",
        "style": "oil",
        "simplify": 1,
        "conf_threshold": 0.25,
        "description": "Person + landscape - oil enhances natural feel"
    },
    {
        "name": "lake_pine_trees",
        "file": "lake_pine_trees.png",
        "style": "oil",
        "simplify": 0,
        "conf_threshold": 0.2,
        "description": "Classic Bob Ross landscape - keep full detail"
    },
    {
        "name": "moraine_canoes",
        "file": "moraine_canoes.png",
        "style": "photo",
        "simplify": 1,
        "conf_threshold": 0.2,
        "description": "Colorful canoes - keep colors crisp"
    },
    {
        "name": "yellowstone_couple",
        "file": "yellowstone_couple.png",
        "style": "impressionist",
        "simplify": 2,
        "conf_threshold": 0.25,
        "description": "Portrait + hot spring - impressionist for colorful bands"
    },
]


def process_image(config: dict, base_path: Path):
    """Process a single image with its optimal settings."""
    input_path = base_path / "input" / "ForMaria" / config["file"]
    output_dir = base_path / "output" / "ForMaria" / config["name"]

    if not input_path.exists():
        print(f"[ERROR] Image not found: {input_path}")
        return False

    print("=" * 70)
    print(f"PROCESSING: {config['name']}")
    print("=" * 70)
    print(f"  Description: {config['description']}")
    print(f"  Style: {config['style']}, Simplify: {config['simplify']}")
    print(f"  Input: {input_path}")
    print(f"  Output: {output_dir}")
    print()

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        painter = YOLOBobRossPaint(
            str(input_path),
            model_size="m",
            conf_threshold=config["conf_threshold"],
            substeps_per_region=4,
            paint_style=config["style"],
            simplify=config["simplify"]
        )
        painter.process()
        painter.save_all(str(output_dir))

        print()
        print(f"[SUCCESS] {config['name']} completed!")
        return True

    except Exception as e:
        print(f"[ERROR] Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    base_path = Path(__file__).parent

    print()
    print("=" * 70)
    print("PROCESSING FORMAIRA IMAGES")
    print("=" * 70)
    print()

    results = {}

    for config in IMAGES:
        success = process_image(config, base_path)
        results[config["name"]] = success
        print()

    # Summary
    print()
    print("=" * 70)
    print("PROCESSING SUMMARY")
    print("=" * 70)

    for name, success in results.items():
        status = "OK" if success else "FAILED"
        print(f"  {name}: {status}")

    success_count = sum(1 for v in results.values() if v)
    print()
    print(f"Completed: {success_count}/{len(results)}")


if __name__ == "__main__":
    main()
