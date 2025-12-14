# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Artisan is a multi-medium art instruction generator that converts images into step-by-step creation guides. The system currently supports acrylic painting with paint-by-numbers templates, Bob Ross-style painting guidance, and AI style transfer.

## Common Commands

### Setup
```bash
cd artisan
pip install -r requirements.txt
```

### Basic Generation
```bash
# Generate all instruction levels + paint kit
python cli/generate.py <image_name> all

# Generate specific level: beginner, easy, intermediate, advanced, kit
python cli/generate.py <image_name> advanced

# Generate with budget constraint
python cli/generate.py <image_name> kit 15 16x20 75

# Compare budget tiers
python cli/generate.py <image_name> compare
```

### YOLO + Bob Ross Pipeline (Advanced)
```bash
# Process image with YOLO semantic segmentation + Bob Ross style instructions
python -m generators.yolo_bob_ross_paint <image_path> [output_dir] [--style=STYLE] [--simplify=LEVEL]

# Style options: photo, oil, impressionist, poster, watercolor
# Simplify levels: 0 (none), 1 (light), 2 (medium), 3 (heavy)
```

### Style Transfer
```bash
# Using preset style
python cli/style_transfer.py input.jpg output.jpg --style britto_style

# Using custom prompt
python cli/style_transfer.py input.jpg output.jpg --prompt "vibrant pop art with bold outlines"
```

## Architecture

### Core Processing Pipeline

```
core/
├── paint_by_numbers.py    # Image → template conversion
│                          # K-means clustering in LAB space
│                          # Connected component segmentation
├── color_matcher.py       # Match extracted colors to paints
│                          # Generate mixing recipes
└── paint_database.py      # 40+ commercial acrylic paints (Golden, Liquitex, W&N)
```

### YOLO + Bob Ross System (generators/yolo_bob_ross_paint.py)

The most advanced pipeline combining:
1. **Scene Context Analysis** - Detects time of day, weather, lighting, mood
2. **YOLO Semantic Segmentation** - Identifies objects (dogs, trees, sky, etc.)
3. **Advanced Scene Analysis** - Determines lighting roles (emitter, silhouette, reflector)
4. **Value Progression Planning** - Dark-to-light, glow-then-edges, silhouette-rim
5. **Bob Ross Style Instructions** - Technique-aware painting guidance

Key classes:
- `YOLOBobRossPaint` - Main orchestrator
- `SemanticPaintingLayer` - Semantic region with painting substeps
- `PaintingSubstep` - Individual painting step with mask, technique, brush

### Perception Module (perception/)

Scene understanding components:
- `yolo_segmentation.py` - YOLO-based object detection
- `scene_context.py` - Time/weather/lighting analysis
- `scene_analyzer.py` - Lighting roles, depth layers, value progressions
- `layering_strategies.py` - Subject-specific painting strategies
- `painting_planner.py` - Generates context-aware substeps

### Generation Hierarchy

```
generators/
├── bob_ross/
│   ├── generator.py       # Bob Ross style step generator
│   ├── steps.py           # PaintingStep dataclass
│   └── constants.py       # Brushes, strokes, encouragements
├── instruction_generator.py    # Multi-level instruction output
├── paint_kit_generator.py      # Shopping lists, mixing guides
├── smart_paint_by_numbers.py   # Technique-aware templates
├── semantic_paint_by_numbers.py # Semantic-region-aware templates
├── yolo_smart_paint.py         # YOLO + smart paint
└── yolo_bob_ross_paint.py      # YOLO + Bob Ross (most advanced)
```

### Budget Optimization (optimization/budget_optimizer.py)

Optimizes paint selection for budget tiers:
- $40-55: Minimal (5 paints, heavy mixing)
- $55-80: Budget (6-7 paints)
- $80-120: Standard (8-10 paints)
- $120-180: Premium (12-14 paints)
- $180+: Professional (20+ paints)

## Input/Output Structure

```
input/<image_name>/<image_name>.png    # Source image

output/<image_name>/
├── beginner/          # Simple numbered template
├── easy/              # With layer order guidance
├── intermediate/      # Technique-aware
├── advanced/          # Bob Ross style
├── paint_kit/         # Standard kit outputs
│   ├── shopping_list.png
│   ├── color_chart.png
│   ├── mixing_guide.png
│   └── paint_kit.json
└── <name>_yolo_bob_ross/  # YOLO + Bob Ross outputs
    ├── steps/
    │   ├── cumulative/    # Progressive build-up
    │   ├── context/       # Highlighted in full image
    │   └── isolated/      # Region on white canvas
    ├── steps_by_region/
    ├── painting_guide.json
    └── scene_analysis.json
```

## Key Dependencies

- **numpy, opencv-python, scikit-learn, scipy**: Core image processing
- **matplotlib, Pillow**: Visualization
- **torch, diffusers, transformers**: Style transfer (ControlNet)
- **ultralytics**: YOLO segmentation (optional, for yolo_* generators)

## Style Options in YOLO Bob Ross

The `paint_style` parameter applies painterly effects:
- `photo`: Original image
- `oil`: Rich colors, visible brush strokes
- `impressionist`: Soft, dreamy, posterized
- `poster`: Flat colors with strong edges
- `watercolor`: Soft, transparent, bleeding edges

The `simplify` parameter (0-3) reduces detail before YOLO detection to produce cleaner regions.
