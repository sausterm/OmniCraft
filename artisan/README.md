# Artisan - Multi-Medium Art Instruction Generator

Convert any image into step-by-step creation guides for multiple artistic mediums: painting, fiber arts, pixel art, and more.

> **Current Status:** Acrylic painting fully supported. Other mediums (LEGO, cross-stitch, watercolor, crochet) coming soon.

## Features

- **Image to Template**: Convert any image to a numbered paint-by-numbers template
- **Budget Optimization**: Generate kits optimized for your budget ($40-$200+)
- **Smart Color Matching**: Match extracted colors to 40+ commercial acrylic paints
- **Mixing Recipes**: Automatic mixing instructions for colors not directly available
- **Multi-Level Instructions**: From simple numbered regions to Bob Ross-style guidance
- **Technique Detection**: Identify painting techniques (glazing, wet-on-wet) from image analysis

## Quick Start

### Setup

```bash
cd artisan
pip install -r requirements.txt
```

### Basic Usage

1. Place your image in `input/<name>/<name>.png`
2. Run the generator:

```bash
# Generate everything (all instruction levels + paint kit)
python cli/generate.py my_image all

# Generate paint kit only
python cli/generate.py my_image kit

# Generate with $75 budget constraint
python cli/generate.py my_image kit 15 16x20 75

# Compare budget options
python cli/generate.py my_image compare
```

### Output Structure

```
output/my_image/
├── beginner/           # Simple paint-by-numbers
├── easy/               # Guided with layer order
├── intermediate/       # Technique-aware instructions
├── advanced/           # Bob Ross style step-by-step
├── paint_kit/          # Standard kit (no budget)
└── paint_kit_$75/      # Budget-optimized kit
    ├── shopping_list.png
    ├── color_chart.png
    ├── mixing_guide.png
    └── paint_kit.json
```

## Command Reference

```bash
python cli/generate.py <image_name> [level] [n_colors] [canvas_size] [budget]
```

### Levels

| Level | Description |
|-------|-------------|
| `beginner` | Simple numbered template |
| `easy` | Guided with suggested paint order |
| `intermediate` | Technique-aware with brush tips |
| `advanced` | Bob Ross style step-by-step |
| `kit` | Shopping list and mixing guide |
| `compare` | Compare budget tiers |
| `compare-all` | Generate kits for all budgets |
| `all` | Everything |

### Budget Tiers

| Budget | Tier | Paints | Mixing | Accuracy |
|--------|------|--------|--------|----------|
| $40-55 | Minimal | 5 | Heavy | ~85% |
| $55-80 | Budget | 6-7 | Moderate | ~90% |
| $80-120 | Standard | 8-10 | Light | ~95% |
| $120-180 | Premium | 12-14 | Minimal | ~98% |
| $180+ | Professional | 20+ | None | ~100% |

### Canvas Sizes

8x10, 9x12, 11x14, 12x16, 16x20, 18x24, 24x30, 24x36 inches

## Python API

```python
from artisan import PaintByNumbers, PaintKitGenerator, BudgetOptimizer

# Basic paint-by-numbers
pbn = PaintByNumbers('image.jpg', n_colors=15)
pbn.process_all('output/')

# Budget-optimized kit
kit_gen = PaintKitGenerator('image.jpg', n_colors=15, budget=75)
kit = kit_gen.generate_kit()
kit_gen.generate_all_outputs('output/')

# Budget analysis
palette = [(255, 0, 0), (0, 255, 0), ...]  # RGB colors
optimizer = BudgetOptimizer(palette)
analysis = optimizer.analyze_budget(75)
print(f"Accuracy: {analysis.average_accuracy}%")
min_budget = optimizer.find_minimum_budget(target_accuracy=90)
```

## Package Structure

```
artisan/
├── __init__.py                 # Main package exports
├── core/                       # Core processing modules
│   ├── paint_by_numbers.py     # Image-to-template conversion
│   ├── color_matcher.py        # Color matching & mixing recipes
│   └── paint_database.py       # Commercial paint database (40+ colors)
├── generators/                 # Instruction & kit generators
│   ├── instruction_generator.py    # Multi-level instructions
│   ├── paint_kit_generator.py      # Complete kit generation
│   ├── smart_paint_by_numbers.py   # Technique-aware templates
│   └── bob_ross/               # Bob Ross style instructions
│       ├── generator.py        # Main generator class
│       ├── steps.py            # Painting step dataclass
│       └── constants.py        # Brushes, motions, encouragements
├── analysis/                   # Image analysis
│   ├── technique_analyzer.py   # Painting technique detection
│   └── technique_visualizer.py # Technique visualization
├── optimization/               # Budget optimization
│   └── budget_optimizer.py     # Budget-aware paint selection
├── cli/                        # Command-line tools
│   ├── generate.py             # Main CLI entry point
│   └── generate_multi.py       # Batch generation
├── input/                      # Input images
└── output/                     # Generated outputs
```

## Configuration

### Detail Levels

```python
# Beginner (easy)
pbn = PaintByNumbers('image.jpg', n_colors=8, min_region_size=200)

# Balanced
pbn = PaintByNumbers('image.jpg', n_colors=15, min_region_size=50)

# Advanced (detailed)
pbn = PaintByNumbers('image.jpg', n_colors=25, min_region_size=20)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image_path` | str | required | Path to input image |
| `n_colors` | int | 20 | Number of colors (6-30 recommended) |
| `min_region_size` | int | 10 | Minimum pixels per region |
| `budget` | float | None | Budget constraint in USD |
| `canvas_size` | str | "16x20" | Canvas dimensions |

## Budget Comparison Example

```
Budget     Tier         Paints   Accuracy   Cost       Mixing
----------------------------------------------------------------------
$50        Minimal      5        92%        $60        SIMPLE
$75        Budget       6        94%        $68        SIMPLE
$100       Standard     8        96%        $96        SIMPLE
$150       Premium      12       98%        $144       SIMPLE

Recommendations:
  Minimum for 85% accuracy: $40
  Minimum for 95% accuracy: $100
```

## Generated Outputs

### Shopping List (`shopping_list.png`)
- Paint names and brands
- Quantity needed (oz) with 30% safety margin
- Number of 2oz tubes to purchase
- Price per tube and total cost
- Which color numbers use each paint
- Budget tier info (if budget mode)

### Mixing Guide (`mixing_guide.png`)
- Target color swatch
- Component paints with percentages
- Step-by-step mixing instructions
- Expected result swatch

### Color Chart (`color_chart.png`)
- All colors with RGB/hex values
- Coverage percentage
- Paint match or "MIX REQUIRED"
- Accuracy percentage

## Algorithm Overview

### Color Quantization
1. Convert image to LAB color space (perceptually uniform)
2. Apply K-means clustering to reduce to N colors
3. Sort palette by pixel coverage (most used first)

### Region Segmentation
1. Create binary mask for each color
2. Find connected components (contiguous regions)
3. Merge small regions (<min_region_size) into neighbors
4. Generate boundary outlines with Canny edge detection

### Budget Optimization
1. Define optimal paint sets for different budget levels
2. For each palette color, find best achievable match
3. Try single paints, then 2-paint mixes, then 3-paint mixes
4. Calculate weighted accuracy by pixel coverage
5. Recommend minimum budget for target accuracy

## Paint Database

Includes 40+ commercial acrylic paints from:
- Golden Heavy Body Acrylics
- Liquitex Heavy Body
- Winsor & Newton Professional

Each paint includes:
- RGB values for color matching
- Price per 2oz tube
- Coverage rate (sq ft/oz)
- Pigment codes

## Business Integration

### Complete Workflow
1. Customer uploads image
2. Convert to paint-by-numbers (this software)
3. Print template on canvas
4. Match colors to paint inventory
5. Assemble kit (canvas + paints + brushes)
6. Ship to customer

### Cost Example (16x20" kit, $75 budget tier)
```
Canvas (printed):     $5.00
6 paints (2oz each):  $68.00
Brushes (3pc):        $5.00
Packaging:            $2.00
Color guide:          $1.00
----------------------------
Total COGS:           $81.00

Retail Price:         $149-199
Profit:               $68-118 (46-59% margin)
```

## Tips for Best Results

### Image Selection
- High resolution (800x600+)
- Clear subjects with good contrast
- Avoid very dark or overexposed images
- Simple compositions work best for beginners

### Color Count Guidelines
- **6-8 colors**: Beginner-friendly, quick projects
- **10-15 colors**: Balanced detail, most paintings
- **20-30 colors**: Advanced, detailed artwork
- **30+ colors**: Expert level, photorealistic

### Budget Selection
- **$50-75**: Good for simple images, beginners
- **$100-150**: Best balance of quality and cost
- **$180+**: Professional quality, minimal mixing

## Requirements

```
numpy
opencv-python
scikit-learn
scipy
matplotlib
Pillow
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Too many small regions | Increase `min_region_size` |
| Poor paint matches | Increase budget or add custom colors |
| Template too complex | Reduce `n_colors` |
| Not enough detail | Increase `n_colors`, decrease `min_region_size` |
| High mixing complexity | Increase budget tier |

## Examples

### Generate All Levels
```bash
python cli/generate.py aurora_maria all
```

### Budget-Constrained Kit
```bash
python cli/generate.py aurora_maria kit 15 16x20 75
```

### Compare Budgets Before Deciding
```bash
python cli/generate.py aurora_maria compare
```

### Python API
```python
from artisan import PaintByNumbers

pbn = PaintByNumbers('photo.jpg', n_colors=15)
matched_colors = pbn.process_all(output_dir='output')
print(f"Created template with {len(matched_colors)} colors")
```

## License

MIT License - Use this code for your paint-by-numbers business!

---

**Ready to start?**

```bash
pip install -r requirements.txt
python cli/generate.py --help
```
