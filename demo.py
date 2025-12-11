"""
Paint by Numbers Demo Script - Simple Example
"""

from paint_by_numbers import PaintByNumbers
from paint_colors import get_all_paint_colors


def demo_basic():
    """
    Basic demo showing how to convert an image to paint-by-numbers
    """
    print("="*80)
    print("Paint by Numbers Converter - Basic Demo")
    print("="*80)
    
    # Example: Convert your image
    print("\nUsage Example:")
    print("-" * 80)
    print("""
# 1. Create converter
pbn = PaintByNumbers(
    image_path='your_image.jpg',
    n_colors=15,           # Number of colors
    min_region_size=50     # Minimum region size
)

# 2. Process and save everything
matched_colors = pbn.process_all(output_dir='output')

# That's it! Check the 'output' folder for:
#   - visualization.png
#   - template.png (print this on canvas!)
#   - color_guide.json
#   - color_reference.png
""")
    
    print("\n" + "="*80)
    print("Features:")
    print("="*80)
    print("✓ Convert any image to N colors")
    print("✓ Create numbered paint-by-numbers template")
    print("✓ Match colors to commercial paints")
    print("✓ Generate printable canvas-ready files")
    print()
    print("Adjust settings:")
    print("  - n_colors: 8-30 (fewer = easier, more = detailed)")
    print("  - min_region_size: 20-200 (smaller = more detail)")
    print()


def example_with_custom_colors():
    """
    Example showing how to use custom paint colors
    """
    print("="*80)
    print("Example: Using Custom Paint Colors")
    print("="*80)
    print("""
# Define your actual paint inventory
my_paints = {
    "My Red Paint": [255, 0, 0],
    "My Blue Paint": [0, 0, 255],
    "My Green Paint": [0, 255, 0],
    # ... add all your paints
}

# Create converter
pbn = PaintByNumbers('image.jpg', n_colors=15)

# Process with your colors
pbn.quantize_colors()
pbn.create_regions()
pbn.create_template()
matched = pbn.match_to_paint_colors(paint_database=my_paints)

# Save outputs
pbn.save_template('template.png')
""")


def example_business_workflow():
    """
    Example showing business integration workflow
    """
    print("="*80)
    print("Business Workflow Example")
    print("="*80)
    print("""
# Complete order processing
def process_customer_order(image_path, customer_id):
    # Convert image
    pbn = PaintByNumbers(image_path, n_colors=15)
    matched_colors = pbn.process_all(f'orders/{customer_id}')
    
    # Calculate kit components
    kit = {
        'canvas': '16x20 stretched',
        'paints': matched_colors,
        'cost': calculate_cost(matched_colors)
    }
    
    # Generate packing slip
    # Send template to canvas printer
    # Assemble paint kit
    # Ship to customer
    
    return kit

# See BUSINESS_INTEGRATION.md for full details!
""")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("🎨 PAINT BY NUMBERS CONVERTER")
    print("="*80)
    
    demo_basic()
    print("\n\n")
    example_with_custom_colors()
    print("\n\n")
    example_business_workflow()
    
    print("\n" + "="*80)
    print("Ready to start? Just run:")
    print("  pbn = PaintByNumbers('your_image.jpg', n_colors=15)")
    print("  pbn.process_all()")
    print("="*80)
