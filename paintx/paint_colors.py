"""
Comprehensive Paint Color Database
Contains colors from major paint brands for matching
"""

# Expanded paint color database with common acrylic paint colors
PAINT_DATABASE = {
    # Whites
    "Titanium White": [255, 255, 255],
    "Zinc White": [253, 252, 242],
    
    # Blacks & Grays
    "Ivory Black": [41, 36, 33],
    "Mars Black": [30, 30, 30],
    "Payne's Gray": [83, 104, 120],
    "Neutral Gray": [146, 146, 146],
    
    # Yellows
    "Cadmium Yellow Light": [255, 236, 0],
    "Cadmium Yellow Medium": [255, 200, 8],
    "Lemon Yellow": [255, 244, 79],
    "Yellow Ochre": [227, 168, 87],
    "Naples Yellow": [250, 218, 94],
    "Raw Sienna": [199, 97, 20],
    
    # Oranges
    "Cadmium Orange": [255, 121, 0],
    "Burnt Sienna": [138, 54, 15],
    "Burnt Umber": [138, 51, 36],
    
    # Reds
    "Cadmium Red": [227, 0, 34],
    "Alizarin Crimson": [227, 38, 54],
    "Vermilion": [227, 66, 52],
    "Magenta": [202, 31, 123],
    
    # Blues
    "Ultramarine Blue": [18, 10, 143],
    "Cobalt Blue": [0, 71, 171],
    "Cerulean Blue": [42, 82, 190],
    "Prussian Blue": [0, 49, 83],
    "Turquoise": [64, 224, 208],
    
    # Greens
    "Viridian Green": [64, 130, 109],
    "Sap Green": [48, 98, 48],
    "Chrome Green": [56, 97, 39],
    "Olive Green": [128, 128, 0],
    
    # Earth Tones
    "Raw Umber": [115, 74, 18],
    "Sepia": [112, 66, 20],
    
    # Purples
    "Violet": [143, 0, 255],
}


def get_all_paint_colors():
    """Get complete paint database"""
    return PAINT_DATABASE


if __name__ == "__main__":
    print(f"Paint Color Database - {len(PAINT_DATABASE)} colors")
    for name, rgb in sorted(PAINT_DATABASE.items()):
        hex_code = '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])
        print(f"  {name:30s} RGB: {rgb}  HEX: {hex_code}")
