# Quick Start Guide

## Get Running in 60 Seconds ⚡

### Step 1: Install (10 seconds)
```bash
pip install -r requirements.txt
```

### Step 2: Create Your First Template (30 seconds)
```python
from paint_by_numbers import PaintByNumbers

# Convert any image
pbn = PaintByNumbers('your_photo.jpg', n_colors=15)
matched_colors = pbn.process_all(output_dir='output')
```

### Step 3: Check Your Results (20 seconds)
```bash
ls output/
# You'll see:
#   - template.png (print this on canvas!)
#   - color_guide.json
#   - visualization.png
#   - color_reference.png
```

## What You Get

### 1. Template (`template.png`)
- **Ready to print on canvas**
- **Numbered regions** for painting
- **Black outlines** for clarity
- Take to local canvas printer or upload to POD service

### 2. Color Guide (`color_guide.json`)
```json
{
  "total_colors": 15,
  "colors": [
    {
      "number": 1,
      "paint_name": "Cerulean Blue",
      "extracted_hex": "#87ceeb",
      "paint_hex": "#2a52be",
      "pixel_count": 12543
    }
  ]
}
```

### 3. Visualization (`visualization.png`)
- Shows original image
- Quantized version
- Template with numbers
- Color palette

### 4. Color Reference (`color_reference.png`)
- Visual paint matching guide
- Take to art supply store
- Buy the matched paints

## Adjust Settings

### Easy Mode (Beginners)
```python
pbn = PaintByNumbers('image.jpg', n_colors=8, min_region_size=200)
```
- Fewer colors = easier
- Larger regions = faster to paint

### Detailed Mode (Advanced)
```python
pbn = PaintByNumbers('image.jpg', n_colors=25, min_region_size=20)
```
- More colors = more detail
- Smaller regions = more painting work

## Use Your Paint Colors

```python
my_paints = {
    "Red #1": [255, 0, 0],
    "Blue #2": [0, 0, 255],
    # Add all your paints...
}

pbn = PaintByNumbers('image.jpg', n_colors=15)
pbn.quantize_colors()
pbn.create_regions()
pbn.create_template()
matched = pbn.match_to_paint_colors(paint_database=my_paints)
```

## Business Workflow

### For Each Customer Order:

1. **Receive image** from customer
2. **Process**: `pbn.process_all()`
3. **Print template** on 16x20" canvas ($5)
4. **Match paints** from color_guide.json
5. **Package kit**: canvas + 15 paints + 3 brushes
6. **Ship** to customer

### Pricing:
- **COGS**: ~$17
- **Retail**: $49-69
- **Profit**: $32-52 per kit

## Common Adjustments

| Want this | Do this |
|-----------|---------|
| Easier to paint | Reduce `n_colors` to 8-10 |
| More detail | Increase `n_colors` to 20-25 |
| Fewer small areas | Increase `min_region_size` to 100+ |
| More painting challenge | Decrease `min_region_size` to 20-30 |

## Next Steps

1. ✅ Test with 5-10 different images
2. ✅ Print a template on paper to check quality
3. ✅ Get canvas printing quotes
4. ✅ Source paint colors (see earlier research)
5. ✅ Create your first complete kit!

## Need Help?

- **Full docs**: README.md
- **Examples**: demo.py
- **Business workflow**: See earlier discussion on canvas/paint sourcing

---

**That's it! You're ready to start your paint-by-numbers business! 🎨**
