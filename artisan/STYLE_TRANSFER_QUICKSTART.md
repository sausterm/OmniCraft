# Style Transfer Quick Start

Get started in 5 minutes transforming images with custom text prompts.

## Setup (One-Time)

```bash
cd artisan

# Install dependencies
pip install -r requirements.txt
```

**Note:** First run will download AI models (~5GB). This happens once automatically.

## Basic Usage

### Command Line

```bash
# Your exact use case: Pop art style
python cli/style_transfer.py my_photo.jpg my_art.jpg \
    --prompt-file examples/prompts/pop_art_geometric.txt

# Custom prompt
python cli/style_transfer.py input.jpg output.jpg \
    --prompt "vibrant pop art with bold outlines and bright colors"

# Preset style
python cli/style_transfer.py input.jpg output.jpg --style britto_style
```

### Python

```python
from PIL import Image
from artisan.style_transfer.engines import ControlNetEngine

# Initialize
engine = ControlNetEngine()

# Load image
image = Image.open("input.jpg")

# Apply style
result = engine.apply_style(
    image=image,
    style="custom",
    custom_prompt="vibrant pop art style with bold black outlines, "
                  "geometric patterns, bright saturated colors, "
                  "decorative hearts dots stripes, flat 2D aesthetic"
)

# Save
result.save("output.png")
print(f"Done! Took {result.processing_time:.1f} seconds")
```

## Your Pop Art Use Case

The exact style you described is ready to use:

```bash
python cli/style_transfer.py dogs_photo.jpg dogs_pop_art.jpg \
    --prompt-file examples/prompts/pop_art_geometric.txt \
    --steps 40 \
    --guidance 8.5 \
    --control-strength 1.1
```

This will:
- Apply vibrant pop art style with bold outlines
- Use geometric patterns and bright colors
- Add decorative patterns (hearts, dots, stripes)
- **Maintain the dogs' identity and likeness**

Expected processing time on Mac Mini: 3-5 minutes

## Key Parameters

- `--steps 40`: Higher quality (20-50 range, default 30)
- `--guidance 8.5`: Follow style closely (5-15 range, default 7.5)
- `--control-strength 1.1`: Preserve composition (0.5-1.5 range, default 1.0)
- `--seed 42`: Reproducible results (optional)

## What's Included

### Preset Styles
- `britto_style` - Your signature pop art style
- `pop_art` - Classic Warhol-style
- `van_gogh` - Impressionist swirls
- `anime` - Japanese animation
- `watercolor` - Soft painting
- More in the guide...

### Example Prompts
- `examples/prompts/pop_art_geometric.txt` - Your exact style description
- Create your own `.txt` files here

### Examples
- `examples/style_transfer_examples.py` - 7 complete examples
- Shows preset styles, custom prompts, batch processing, variations

## Next Steps

1. **Try your pop art style** with the command above
2. **Read the full guide**: `STYLE_TRANSFER_GUIDE.md`
3. **Experiment with parameters** to fine-tune results
4. **Create custom prompts** and save them in `examples/prompts/`

## Troubleshooting

**Slow on first run?**
Models are downloading (~5GB). Only happens once.

**Out of memory?**
Resize your image first or use a cloud API (see guide).

**Style not strong enough?**
Increase `--guidance` to 9-12.

**Subjects look different?**
Increase `--control-strength` to 1.2-1.3.

## Full Documentation

- **Complete Guide**: `STYLE_TRANSFER_GUIDE.md`
- **Working Examples**: `examples/style_transfer_examples.py`
- **Your Prompt**: `examples/prompts/pop_art_geometric.txt`

---

**Transform your first image now:**

```bash
python cli/style_transfer.py my_photo.jpg my_art.jpg \
    --prompt "vibrant pop art with bold outlines and bright colors"
```
