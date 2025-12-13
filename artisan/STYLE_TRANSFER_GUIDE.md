# Style Transfer Guide

Transform any image into any artistic style using custom text prompts.

## Overview

The Artisan style transfer system lets you apply unlimited artistic styles to images by describing the style in text. Unlike traditional neural style transfer that requires reference images, this system uses **ControlNet + Stable Diffusion** to understand and apply styles from text descriptions while preserving the structure and composition of your input image.

### Key Features

- **Custom Text Prompts**: Describe any style in natural language
- **Structure Preservation**: Maintains composition, poses, and identity of subjects
- **Local Processing**: Runs on your Mac (Apple Silicon optimized) or any GPU
- **Multiple Engines**: Choose between local ControlNet or cloud APIs (Replicate)
- **Multiple Control Modes**: Edges, depth, pose detection for different use cases
- **Reproducible**: Set seeds for consistent results

## Quick Start

### 1. Installation

```bash
cd artisan

# Install dependencies
pip install -r requirements.txt
```

**Note**: First run will download models (~5GB). This only happens once.

### 2. Basic Usage

#### Command Line (Recommended)

```bash
# Use preset style
python cli/style_transfer.py input.jpg output.jpg --style britto_style

# Use custom prompt
python cli/style_transfer.py input.jpg output.jpg \
    --prompt "vibrant pop art with bold outlines and bright colors"

# Use prompt from file
python cli/style_transfer.py input.jpg output.jpg \
    --prompt-file examples/prompts/pop_art_geometric.txt
```

#### Python API

```python
from PIL import Image
from artisan.style_transfer.engines import ControlNetEngine

# Initialize engine
engine = ControlNetEngine()

# Load image
image = Image.open("input.jpg")

# Apply style
result = engine.apply_style(
    image=image,
    style="custom",
    custom_prompt="vibrant pop art style with bold black outlines, "
                  "geometric patterns, bright saturated colors"
)

# Save result
result.save("output.png")
```

## Use Case: Pop Art Style Transfer

This is the exact use case you described - transforming photos into vibrant pop art while maintaining subject identity.

### Your Custom Pop Art Prompt

The exact style you requested is saved in `examples/prompts/pop_art_geometric.txt`:

```
vibrant pop art style with bold, thick black outlines defining all shapes
and forms. Geometric patterns and cubist influences - break subjects into
angular, simplified shapes. Extremely bright, saturated colors - emphasize
primary colors (red, blue, yellow) along with hot pink, orange, lime green,
and purple. Decorative pattern fills - use hearts, dots, stripes, diamonds,
flowers, swirls, and checkered patterns. Flat, 2D aesthetic - no shading
or depth. Playful, optimistic mood. High contrast. Modern pop art meets
Cuban folk art. Contemporary Brazilian pop art with neo-cubist geometric
patterns and tropical color palette. Maintain subject likeness and identity.
```

### Usage

```bash
# Using the saved prompt file
python cli/style_transfer.py my_photo.jpg pop_art_output.jpg \
    --prompt-file examples/prompts/pop_art_geometric.txt \
    --steps 40 \
    --guidance 8.5 \
    --control-strength 1.1

# Or run the example script
python examples/style_transfer_examples.py
```

### Parameters for Best Results

For the pop art style, these parameters work well:

- **Steps**: 35-50 (higher = more detailed patterns)
- **Guidance**: 8.0-9.5 (higher = follows style description more closely)
- **Control Strength**: 1.0-1.2 (higher = preserves original composition better)
- **Control Mode**: `canny` (edge detection) works best for geometric styles

```bash
python cli/style_transfer.py dogs_photo.jpg dogs_pop_art.jpg \
    --prompt-file examples/prompts/pop_art_geometric.txt \
    --steps 40 \
    --guidance 8.5 \
    --control-strength 1.1 \
    --seed 42
```

## Available Engines

### 1. ControlNet Engine (Local)

**Pros:**
- Runs locally on your Mac (Apple Silicon optimized)
- No API costs
- Full privacy
- Unlimited generations
- Best structure preservation

**Cons:**
- Requires model download (~5GB, one-time)
- Slower than cloud APIs (2-5 min per image on Mac Mini)
- Requires 8GB+ RAM

**Best for:** High-quality results, batch processing, privacy

### 2. Replicate Engine (Cloud API)

**Pros:**
- Very fast (<30 seconds)
- No local compute needed
- Access to 100+ models

**Cons:**
- Requires API key and costs ~$0.005 per image
- Requires internet connection
- Less control over parameters

**Best for:** Quick tests, limited hardware

```bash
# Set API key
export REPLICATE_API_TOKEN="your-token-here"

# Use Replicate engine
python cli/style_transfer.py input.jpg output.jpg \
    --engine replicate \
    --style pop_art
```

## Control Modes

Different control modes extract different information from your input image:

### Canny (Edge Detection)
**Best for:** Most styles, geometric art, illustrations
```bash
--control-mode canny
```
Detects sharp edges and outlines. **Recommended for pop art, Britto style, cartoons.**

### HED (Soft Edges)
**Best for:** Paintings, watercolor, organic styles
```bash
--control-mode hed
```
Detects softer, more natural edges. Better for painterly styles.

### OpenPose (Human Pose)
**Best for:** Photos with people
```bash
--control-mode openpose
```
Detects human body poses. Excellent for maintaining posture and gesture when transforming photos of people.

## Parameter Guide

### Core Parameters

| Parameter | Range | Default | Effect |
|-----------|-------|---------|--------|
| `--steps` | 20-50 | 30 | More steps = better quality, slower |
| `--guidance` | 5-15 | 7.5 | Higher = follows prompt more closely |
| `--control-strength` | 0.5-1.5 | 1.0 | Higher = preserves structure better |
| `--seed` | any int | random | Set for reproducible results |

### Recommended Settings by Style

#### Pop Art / Britto Style
```bash
--steps 40 --guidance 8.5 --control-strength 1.1
```

#### Van Gogh / Impressionist
```bash
--steps 35 --guidance 7.0 --control-strength 0.9 --control-mode hed
```

#### Anime / Manga
```bash
--steps 30 --guidance 8.0 --control-strength 1.2
```

#### Watercolor
```bash
--steps 25 --guidance 6.5 --control-strength 0.7 --control-mode hed
```

## Advanced Usage

### Batch Processing

Process multiple images with the same style:

```python
from pathlib import Path
from PIL import Image
from artisan.style_transfer.engines import ControlNetEngine

engine = ControlNetEngine()

# Load custom prompt
with open("examples/prompts/pop_art_geometric.txt") as f:
    prompt = f.read()

# Process all images in a folder
for img_path in Path("input").glob("*.jpg"):
    image = Image.open(img_path)

    result = engine.apply_style(
        image=image,
        style="custom",
        custom_prompt=prompt,
        num_inference_steps=40,
        guidance_scale=8.5,
        seed=42  # Same seed for consistent style
    )

    result.save(f"output/{img_path.stem}_pop_art.png")
    print(f"Processed: {img_path.name}")
```

### Style Variations

Generate multiple variations to choose from:

```bash
# Variation 1: Subtle style
python cli/style_transfer.py input.jpg output_subtle.jpg \
    --prompt "pop art style" --guidance 6.0 --control-strength 0.8

# Variation 2: Balanced
python cli/style_transfer.py input.jpg output_balanced.jpg \
    --prompt "pop art style" --guidance 7.5 --control-strength 1.0

# Variation 3: Strong style
python cli/style_transfer.py input.jpg output_strong.jpg \
    --prompt "pop art style" --guidance 9.0 --control-strength 1.3
```

### Debugging: View Control Image

See what the AI is using to preserve structure:

```bash
python cli/style_transfer.py input.jpg output.jpg \
    --prompt "pop art" \
    --save-control
```

This creates `output_control.jpg` showing the edge detection used to guide the transformation.

## Preset Styles

The ControlNet engine includes these preset styles:

| Style | Description |
|-------|-------------|
| `pop_art` | Bold colors, high contrast, Warhol-style |
| `britto_style` | Romero Britto style - geometric, decorative, vibrant |
| `van_gogh` | Swirling brushstrokes, impressionist |
| `picasso_cubist` | Geometric, multiple perspectives |
| `anime` | Japanese animation style |
| `watercolor` | Soft, translucent painting |
| `oil_painting` | Rich colors, visible brushstrokes |
| `sketch` | Pencil drawing, hand-drawn |

```bash
# Use preset
python cli/style_transfer.py input.jpg output.jpg --style britto_style
```

## Creating Custom Prompts

### Prompt Engineering Tips

Good prompts are **specific and descriptive**. Include:

1. **Art movement/style**: "pop art", "impressionist", "cubist"
2. **Colors**: "bright saturated primary colors", "pastel tones"
3. **Technique**: "bold outlines", "soft brushstrokes", "geometric patterns"
4. **Mood**: "playful", "dramatic", "serene"
5. **Details**: "decorative patterns with hearts and dots"
6. **Subject preservation**: "maintain subject likeness and identity"

### Example Prompts

**Vibrant Pop Art:**
```
vibrant pop art style with bold thick black outlines, geometric patterns,
extremely bright saturated colors with primary colors and hot pink,
decorative fills with hearts dots stripes, flat 2D aesthetic,
playful optimistic mood, high contrast, maintain subject identity
```

**Impressionist Painting:**
```
impressionist painting in the style of Monet, soft visible brushstrokes,
dappled light, vibrant colors, outdoor scene feeling, artistic oil painting,
gentle color blending, maintain composition and subjects
```

**Anime Style:**
```
anime style illustration, vibrant cel shading, clean sharp lines,
large expressive eyes, manga aesthetic, Japanese animation style,
bold colors, maintain character features and identity
```

**Geometric Abstract:**
```
geometric abstract art, angular shapes, fragmented cubist forms,
bold primary colors, hard edges, Bauhaus influence,
modernist composition, maintain subject structure through geometric forms
```

### Save Prompts for Reuse

Create `.txt` files in `examples/prompts/`:

```bash
echo "your custom style description here" > examples/prompts/my_style.txt

python cli/style_transfer.py input.jpg output.jpg \
    --prompt-file examples/prompts/my_style.txt
```

## Troubleshooting

### Models Not Downloading
**Issue:** Slow or failed model downloads

**Solution:**
```bash
# Set Hugging Face cache directory
export HF_HOME="$HOME/.cache/huggingface"

# Download models manually
python -c "from diffusers import ControlNetModel; ControlNetModel.from_pretrained('lllyasviel/control_v11p_sd15_canny')"
```

### Out of Memory on Mac
**Issue:** Process crashes with memory error

**Solutions:**
1. Reduce image size before processing
2. Use fp32 instead of fp16 (default on Mac)
3. Close other applications
4. Process one image at a time

```python
# Resize large images first
from PIL import Image
img = Image.open("large.jpg")
img = img.resize((1024, 768))  # Resize to reasonable size
```

### Style Not Applying Strongly Enough
**Issue:** Output looks too similar to input

**Solutions:**
1. Increase `--guidance` (try 9-12)
2. Decrease `--control-strength` (try 0.7-0.9)
3. Use more detailed prompt
4. Increase `--steps` (try 40-50)

### Subjects Look Different
**Issue:** People/objects look different after transformation

**Solutions:**
1. Increase `--control-strength` (try 1.1-1.3)
2. For people, use `--control-mode openpose`
3. Add "maintain subject identity and likeness" to prompt
4. Use higher `--steps` (40-50)

### Results Not Consistent
**Issue:** Each run produces different results

**Solution:**
```bash
--seed 42  # Use same seed for reproducibility
```

## Performance

### Mac Mini (Apple Silicon)

- **First run**: ~5GB model download (one-time)
- **Per image**: 2-5 minutes
- **Memory**: 8GB+ recommended
- **Quality**: Excellent

### Tips for Faster Processing

1. **Reduce steps**: Use `--steps 25` for drafts
2. **Resize images**: Process at 512x512, upscale later
3. **Batch smartly**: Same style, different seeds
4. **Use cloud API**: For quick iterations, switch to Replicate

## Best Practices

### For Best Quality

1. **Use high-quality input images** (well-lit, in focus)
2. **Start with preset styles** to understand the system
3. **Iterate on prompts** - save good prompts for reuse
4. **Use adequate steps** (35-50 for final outputs)
5. **Set seeds** when you find a good result

### Workflow Example

1. **Quick test** with preset style (`--style pop_art --steps 20`)
2. **Refine prompt** based on result
3. **Generate variations** with different seeds
4. **Final render** with high steps (`--steps 50`)

### Prompt Library

Build a collection of proven prompts in `examples/prompts/`:

```
examples/prompts/
├── pop_art_geometric.txt       # Your signature style
├── watercolor_soft.txt
├── anime_vibrant.txt
└── van_gogh_swirls.txt
```

## Examples Gallery

See `examples/style_transfer_examples.py` for complete working examples:

- Example 1: Preset styles
- Example 2: Custom prompts
- Example 3: Prompt from file (**your pop art use case**)
- Example 4: Batch processing
- Example 5: Style variations
- Example 6: Different control modes
- Example 7: Save control image

## API Reference

See the code documentation:

- `ControlNetEngine`: Local ControlNet processing
- `ReplicateEngine`: Replicate API integration
- `StyleResult`: Result object with image and metadata
- `StyleRegistry`: Engine discovery and management

## What's Next?

Experiment with:

1. **Your own prompt library** - build a collection of styles
2. **Batch processing** - transform entire photo albums
3. **Style mixing** - combine multiple style descriptions
4. **Fine-tuning** - adjust parameters for your specific use cases

## Support

- Check `examples/` for working code
- Read the code comments for detailed explanations
- Experiment with parameters - they're forgiving!

---

**Ready to transform your images?**

```bash
python cli/style_transfer.py my_photo.jpg my_art.jpg \
    --prompt-file examples/prompts/pop_art_geometric.txt
```
