# Paint by Numbers Image Converter 🎨

Convert any image into a paint-by-numbers template with automatic color matching!

## Quick Start

### Install
```bash
pip install -r requirements.txt
```

### Use
```python
from paint_by_numbers import PaintByNumbers

# Convert your image
pbn = PaintByNumbers('your_image.jpg', n_colors=15)
matched_colors = pbn.process_all(output_dir='output')
```

### Output Files
```
output/
├── visualization.png      # Overview (original + quantized + template + palette)
├── template.png          # Paint-by-numbers template (PRINT THIS!)
├── color_guide.json      # Color matching data
└── color_reference.png   # Visual paint guide
```

## Features

✨ **Smart Color Conversion** - K-means clustering in LAB color space  
🎨 **Paint Matching** - Matches to commercial paint brands  
🔢 **Numbered Regions** - Clean, paintable areas  
📊 **Complete Outputs** - Template, color guide, visualization  
⚙️ **Configurable** - Adjust colors, detail level, brands  

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

## Custom Paint Colors

```python
from paint_colors import get_all_paint_colors

# Use built-in database
paint_db = get_all_paint_colors()

# Or define your own
my_paints = {
    "My Red": [255, 0, 0],
    "My Blue": [0, 0, 255],
    # ... your paint inventory
}

pbn = PaintByNumbers('image.jpg', n_colors=15)
pbn.quantize_colors()
pbn.create_regions()
pbn.create_template()
matched = pbn.match_to_paint_colors(paint_database=my_paints)
```

## API Reference

### `PaintByNumbers(image_path, n_colors=20, min_region_size=10)`
Main converter class

#### Methods:
- **`quantize_colors()`** - Reduce to N colors
- **`create_regions()`** - Segment into paintable regions
- **`create_template(font_size=8, show_numbers=True)`** - Generate template
- **`match_to_paint_colors(paint_database=None)`** - Match to paints
- **`process_all(output_dir)`** - Run complete pipeline
- **`save_template(path)`** - Save template PNG
- **`visualize_results(path)`** - Create visualization

## Business Integration

### Complete Workflow
1. Customer uploads image
2. Convert to paint-by-numbers (this software)
3. Print template on canvas
4. Match colors to paint inventory
5. Assemble kit (canvas + paints + brushes)
6. Ship to customer

### Cost Example (16x20" kit)
```
Canvas (printed): $5.00
15 paints (10ml): $7.50
Brushes (3pc):    $2.00
Packaging:        $1.50
Color guide:      $0.50
----------------------------
Total COGS:       $16.50

Retail Price:     $49-69
Profit:           $32-52 (67-75% margin)
```

## Tips for Best Results

### Image Selection
- ✅ High resolution (800x600+)
- ✅ Clear subjects
- ✅ Good lighting and contrast
- ⚠️ Avoid very dark/bright images

### Color Count
- **6-8 colors**: Beginner-friendly
- **10-15 colors**: Balanced detail
- **20-30 colors**: Advanced
- **30+ colors**: Expert level

### Region Size
- **Large (100-200px)**: Easy to paint
- **Medium (50-100px)**: Good balance
- **Small (20-50px)**: Detailed

## Examples

### Basic Usage
```python
from paint_by_numbers import PaintByNumbers

pbn = PaintByNumbers('photo.jpg', n_colors=15)
matched_colors = pbn.process_all(output_dir='output')

print(f"Created template with {len(matched_colors)} colors")
```

### Step by Step
```python
pbn = PaintByNumbers('photo.jpg', n_colors=15)

# Step 1: Quantize colors
palette, quantized = pbn.quantize_colors()

# Step 2: Create regions
regions = pbn.create_regions()

# Step 3: Create template
template = pbn.create_template()

# Step 4: Match to paints
matched = pbn.match_to_paint_colors()

# Step 5: Save
pbn.save_template('template.png')
```

### Batch Processing
```python
import glob

for img_path in glob.glob('images/*.jpg'):
    pbn = PaintByNumbers(img_path, n_colors=15)
    output_name = img_path.split('/')[-1].replace('.jpg', '')
    pbn.process_all(output_dir=f'output/{output_name}')
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Too many small regions | Increase `min_region_size` |
| Poor paint matches | Add custom colors or try different brand |
| Template too complex | Reduce `n_colors` |
| Not enough detail | Increase `n_colors`, decrease `min_region_size` |

## Dependencies

- numpy - Array operations
- opencv-python - Image processing
- scikit-learn - K-means clustering
- scipy - Distance calculations
- matplotlib - Visualizations
- Pillow - Image handling

## License

Use this code for your paint-by-numbers business!

## Support

- Check `demo.py` for examples
- See code comments for detailed explanations
- All methods have docstrings

---

**Ready to start your paint-by-numbers business? 🚀**

```bash
pip install -r requirements.txt
python demo.py
```
