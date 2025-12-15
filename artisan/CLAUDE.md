# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Artisan is a multi-medium art instruction generator that converts images into step-by-step painting guides. The system uses YOLO semantic segmentation combined with Bob Ross-style painting methodology.

## Common Commands

```bash
# Setup
cd artisan && pip install -r requirements.txt

# Process an image with YOLO + Bob Ross
cd /path/to/OmniCraft
./artisan/venv/bin/python -c "
import sys; sys.path.insert(0, '.')
from artisan.paint.generators.yolo_bob_ross_paint import YOLOBobRossPaint
painter = YOLOBobRossPaint('artisan/input/image.png', paint_style='photo', simplify=0)
painter.process()
painter.save_all('artisan/output/image_name')
"

# Style options: photo, oil, impressionist, poster, watercolor
# Simplify levels: 0 (none), 1 (light), 2 (medium), 3 (heavy)
```

## Package Structure (v6.0)

```
artisan/
├── core/                    # Fundamental types and utilities
│   ├── types.py             # Shared types: BrushType, StrokeMotion, CanvasArea
│   ├── paint_database.py    # 40+ commercial paints (Golden, Liquitex, W&N)
│   ├── color_matcher.py     # Color mixing and matching
│   └── constraints.py       # Art constraints system
│
├── vision/                  # Image understanding (perception + analysis)
│   ├── segmentation/        # Object detection
│   │   ├── yolo_segmentation.py   # YOLO-based segmentation
│   │   ├── semantic.py            # SAM-based fallback
│   │   └── subject_detector.py    # Subject classification
│   └── analysis/            # Scene analysis
│       ├── scene_context.py       # Time/weather/lighting analysis
│       ├── scene_analyzer.py      # Lighting roles, value progressions
│       ├── layering_strategies.py # Subject-specific strategies
│       └── technique_analyzer.py  # Painting technique analysis
│
├── paint/                   # Paint-by-numbers generation
│   ├── generators/          # Main generators
│   │   ├── yolo_bob_ross_paint.py # PRIMARY - YOLO + Bob Ross
│   │   ├── instruction_generator.py
│   │   ├── paint_kit_generator.py
│   │   └── deprecated/      # Legacy generators (with deprecation warnings)
│   ├── bob_ross/            # Bob Ross methodology
│   │   ├── generator.py     # Bob Ross instruction generator
│   │   ├── steps.py         # PaintingStep dataclass
│   │   └── constants.py     # Re-exports from core/types.py
│   ├── planning/            # Painting/lesson planning
│   │   ├── painting_planner.py
│   │   └── lesson_plan.py
│   └── optimization/        # Budget optimization
│       └── budget_optimizer.py
│
├── mediums/                 # Medium-specific implementations
│   ├── base.py              # MediumBase abstract class
│   ├── acrylic/             # Acrylic-specific
│   └── renderers/           # Output rendering
│
├── transfer/                # Style transfer
│   ├── engines/             # ControlNet, Replicate, etc.
│   └── experiments/         # Experimental implementations
│
├── api/                     # FastAPI backend
├── cli/                     # Command-line tools
└── examples/                # Usage examples
```

## Key Imports

```python
# Primary generator (recommended)
from artisan.paint.generators.yolo_bob_ross_paint import YOLOBobRossPaint

# Vision components
from artisan.vision.segmentation.yolo_segmentation import YOLOSemanticSegmenter
from artisan.vision.analysis.scene_context import SceneContextAnalyzer
from artisan.vision.analysis.scene_analyzer import SceneAnalyzer

# Core types
from artisan.core.types import BrushType, StrokeMotion, CanvasArea

# Utilities
from artisan.paint.optimization.budget_optimizer import BudgetOptimizer
from artisan.paint.generators.paint_kit_generator import PaintKitGenerator
```

## Processing Pipeline

```
Input Image
    ↓
YOLOBobRossPaint (orchestrator)
    ├→ YOLOSemanticSegmenter (detection)
    ├→ SceneContextAnalyzer (context)
    ├→ SceneAnalyzer (lighting/depth)
    ├→ PaintingPlanner (substep generation)
    └→ LayeringStrategyEngine (strategies)
    ↓
Output:
├── steps/cumulative/     - Progressive build-up
├── steps/context/        - Highlighted in full image
├── steps/isolated/       - Region on white canvas
├── progress_overview.png - Visual summary
├── painting_guide.json   - Bob Ross instructions
└── scene_analysis.json   - Detection details
```

## Dependencies

- **numpy, opencv-python, scikit-learn, scipy**: Core image processing
- **matplotlib, Pillow**: Visualization
- **ultralytics**: YOLO segmentation
- **torch, diffusers** (optional): Style transfer
