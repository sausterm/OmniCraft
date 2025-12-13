"""
Paint Database - Commercial paint colors from major brands

Includes:
- Acrylic paints (Golden, Liquitex, Winsor & Newton)
- Basic mixing primaries
- Common colors with RGB values
- Price estimates and coverage info
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
from enum import Enum


class PaintBrand(Enum):
    GOLDEN = "Golden"
    LIQUITEX = "Liquitex"
    WINSOR_NEWTON = "Winsor & Newton"
    GENERIC = "Generic/Store Brand"


class PaintType(Enum):
    HEAVY_BODY = "Heavy Body Acrylic"
    FLUID = "Fluid Acrylic"
    SOFT_BODY = "Soft Body Acrylic"


@dataclass
class Paint:
    """A single paint color."""
    name: str
    brand: PaintBrand
    rgb: Tuple[int, int, int]
    paint_type: PaintType
    price_2oz: float  # Price for 2oz tube in USD
    coverage_sqft_per_oz: float  # Square feet per ounce
    is_primary: bool = False  # Is this a mixing primary?
    is_transparent: bool = False
    pigment_code: str = ""  # e.g., "PB29" for Ultramarine Blue


# Comprehensive paint database
PAINT_DATABASE: Dict[str, Paint] = {
    # === WHITES ===
    "titanium_white": Paint(
        name="Titanium White",
        brand=PaintBrand.GOLDEN,
        rgb=(255, 255, 255),
        paint_type=PaintType.HEAVY_BODY,
        price_2oz=8.50,
        coverage_sqft_per_oz=4.0,
        is_primary=True,
        pigment_code="PW6"
    ),
    "zinc_white": Paint(
        name="Zinc White",
        brand=PaintBrand.GOLDEN,
        rgb=(250, 250, 250),
        paint_type=PaintType.HEAVY_BODY,
        price_2oz=9.00,
        coverage_sqft_per_oz=3.5,
        is_transparent=True,
        pigment_code="PW4"
    ),

    # === BLACKS ===
    "mars_black": Paint(
        name="Mars Black",
        brand=PaintBrand.GOLDEN,
        rgb=(28, 28, 30),
        paint_type=PaintType.HEAVY_BODY,
        price_2oz=8.50,
        coverage_sqft_per_oz=5.0,
        is_primary=True,
        pigment_code="PBk11"
    ),
    "ivory_black": Paint(
        name="Ivory Black",
        brand=PaintBrand.WINSOR_NEWTON,
        rgb=(41, 36, 33),
        paint_type=PaintType.HEAVY_BODY,
        price_2oz=9.50,
        coverage_sqft_per_oz=4.5,
        pigment_code="PBk9"
    ),
    "carbon_black": Paint(
        name="Carbon Black",
        brand=PaintBrand.GOLDEN,
        rgb=(20, 20, 20),
        paint_type=PaintType.HEAVY_BODY,
        price_2oz=8.50,
        coverage_sqft_per_oz=5.5,
        pigment_code="PBk7"
    ),

    # === REDS ===
    "cadmium_red_medium": Paint(
        name="Cadmium Red Medium",
        brand=PaintBrand.GOLDEN,
        rgb=(227, 26, 28),
        paint_type=PaintType.HEAVY_BODY,
        price_2oz=18.00,
        coverage_sqft_per_oz=3.5,
        is_primary=True,
        pigment_code="PR108"
    ),
    "cadmium_red_light": Paint(
        name="Cadmium Red Light",
        brand=PaintBrand.GOLDEN,
        rgb=(237, 28, 36),
        paint_type=PaintType.HEAVY_BODY,
        price_2oz=18.00,
        coverage_sqft_per_oz=3.5,
        pigment_code="PR108"
    ),
    "alizarin_crimson": Paint(
        name="Alizarin Crimson Hue",
        brand=PaintBrand.GOLDEN,
        rgb=(227, 38, 54),
        paint_type=PaintType.HEAVY_BODY,
        price_2oz=10.50,
        coverage_sqft_per_oz=3.0,
        is_transparent=True,
        pigment_code="PR177"
    ),
    "quinacridone_magenta": Paint(
        name="Quinacridone Magenta",
        brand=PaintBrand.GOLDEN,
        rgb=(142, 36, 99),
        paint_type=PaintType.HEAVY_BODY,
        price_2oz=14.00,
        coverage_sqft_per_oz=2.5,
        is_primary=True,
        is_transparent=True,
        pigment_code="PR122"
    ),
    "naphthol_red_light": Paint(
        name="Naphthol Red Light",
        brand=PaintBrand.GOLDEN,
        rgb=(224, 60, 49),
        paint_type=PaintType.HEAVY_BODY,
        price_2oz=10.50,
        coverage_sqft_per_oz=3.5,
        pigment_code="PR112"
    ),
    "pyrrole_red": Paint(
        name="Pyrrole Red",
        brand=PaintBrand.GOLDEN,
        rgb=(200, 40, 40),
        paint_type=PaintType.HEAVY_BODY,
        price_2oz=14.00,
        coverage_sqft_per_oz=3.5,
        pigment_code="PR254"
    ),

    # === ORANGES ===
    "cadmium_orange": Paint(
        name="Cadmium Orange",
        brand=PaintBrand.GOLDEN,
        rgb=(237, 135, 45),
        paint_type=PaintType.HEAVY_BODY,
        price_2oz=18.00,
        coverage_sqft_per_oz=3.5,
        pigment_code="PO20"
    ),
    "pyrrole_orange": Paint(
        name="Pyrrole Orange",
        brand=PaintBrand.GOLDEN,
        rgb=(232, 104, 33),
        paint_type=PaintType.HEAVY_BODY,
        price_2oz=14.00,
        coverage_sqft_per_oz=3.5,
        pigment_code="PO73"
    ),

    # === YELLOWS ===
    "cadmium_yellow_medium": Paint(
        name="Cadmium Yellow Medium",
        brand=PaintBrand.GOLDEN,
        rgb=(255, 236, 0),
        paint_type=PaintType.HEAVY_BODY,
        price_2oz=18.00,
        coverage_sqft_per_oz=3.5,
        is_primary=True,
        pigment_code="PY37"
    ),
    "cadmium_yellow_light": Paint(
        name="Cadmium Yellow Light",
        brand=PaintBrand.GOLDEN,
        rgb=(255, 247, 63),
        paint_type=PaintType.HEAVY_BODY,
        price_2oz=18.00,
        coverage_sqft_per_oz=3.5,
        pigment_code="PY37"
    ),
    "hansa_yellow_medium": Paint(
        name="Hansa Yellow Medium",
        brand=PaintBrand.GOLDEN,
        rgb=(252, 211, 0),
        paint_type=PaintType.HEAVY_BODY,
        price_2oz=10.50,
        coverage_sqft_per_oz=3.0,
        is_transparent=True,
        pigment_code="PY74"
    ),
    "yellow_ochre": Paint(
        name="Yellow Ochre",
        brand=PaintBrand.GOLDEN,
        rgb=(227, 168, 87),
        paint_type=PaintType.HEAVY_BODY,
        price_2oz=8.50,
        coverage_sqft_per_oz=4.0,
        pigment_code="PY43"
    ),

    # === GREENS ===
    "phthalo_green_bs": Paint(
        name="Phthalo Green (Blue Shade)",
        brand=PaintBrand.GOLDEN,
        rgb=(0, 78, 56),
        paint_type=PaintType.HEAVY_BODY,
        price_2oz=10.50,
        coverage_sqft_per_oz=2.5,
        is_primary=True,
        is_transparent=True,
        pigment_code="PG7"
    ),
    "phthalo_green_ys": Paint(
        name="Phthalo Green (Yellow Shade)",
        brand=PaintBrand.GOLDEN,
        rgb=(0, 100, 66),
        paint_type=PaintType.HEAVY_BODY,
        price_2oz=10.50,
        coverage_sqft_per_oz=2.5,
        is_transparent=True,
        pigment_code="PG36"
    ),
    "sap_green": Paint(
        name="Sap Green Hue",
        brand=PaintBrand.GOLDEN,
        rgb=(48, 98, 48),
        paint_type=PaintType.HEAVY_BODY,
        price_2oz=10.50,
        coverage_sqft_per_oz=3.0,
        is_transparent=True,
        pigment_code="PG7+PY110"
    ),
    "chromium_oxide_green": Paint(
        name="Chromium Oxide Green",
        brand=PaintBrand.GOLDEN,
        rgb=(76, 114, 72),
        paint_type=PaintType.HEAVY_BODY,
        price_2oz=10.50,
        coverage_sqft_per_oz=4.0,
        pigment_code="PG17"
    ),
    "permanent_green_light": Paint(
        name="Permanent Green Light",
        brand=PaintBrand.GOLDEN,
        rgb=(84, 185, 72),
        paint_type=PaintType.HEAVY_BODY,
        price_2oz=10.50,
        coverage_sqft_per_oz=3.5,
        pigment_code="PG7+PY3"
    ),
    "viridian": Paint(
        name="Viridian Hue",
        brand=PaintBrand.GOLDEN,
        rgb=(64, 130, 109),
        paint_type=PaintType.HEAVY_BODY,
        price_2oz=10.50,
        coverage_sqft_per_oz=3.0,
        is_transparent=True,
        pigment_code="PG7"
    ),
    "jenkins_green": Paint(
        name="Jenkins Green",
        brand=PaintBrand.GOLDEN,
        rgb=(52, 92, 72),
        paint_type=PaintType.HEAVY_BODY,
        price_2oz=10.50,
        coverage_sqft_per_oz=3.5,
        pigment_code="PG7+PBk7"
    ),

    # === BLUES ===
    "phthalo_blue_gs": Paint(
        name="Phthalo Blue (Green Shade)",
        brand=PaintBrand.GOLDEN,
        rgb=(0, 38, 110),
        paint_type=PaintType.HEAVY_BODY,
        price_2oz=10.50,
        coverage_sqft_per_oz=2.5,
        is_primary=True,
        is_transparent=True,
        pigment_code="PB15:3"
    ),
    "phthalo_blue_rs": Paint(
        name="Phthalo Blue (Red Shade)",
        brand=PaintBrand.GOLDEN,
        rgb=(0, 47, 135),
        paint_type=PaintType.HEAVY_BODY,
        price_2oz=10.50,
        coverage_sqft_per_oz=2.5,
        is_transparent=True,
        pigment_code="PB15:1"
    ),
    "ultramarine_blue": Paint(
        name="Ultramarine Blue",
        brand=PaintBrand.GOLDEN,
        rgb=(18, 10, 143),
        paint_type=PaintType.HEAVY_BODY,
        price_2oz=8.50,
        coverage_sqft_per_oz=3.5,
        is_transparent=True,
        pigment_code="PB29"
    ),
    "cerulean_blue": Paint(
        name="Cerulean Blue",
        brand=PaintBrand.GOLDEN,
        rgb=(42, 82, 190),
        paint_type=PaintType.HEAVY_BODY,
        price_2oz=22.00,
        coverage_sqft_per_oz=3.0,
        pigment_code="PB36"
    ),
    "cobalt_blue": Paint(
        name="Cobalt Blue",
        brand=PaintBrand.GOLDEN,
        rgb=(0, 71, 171),
        paint_type=PaintType.HEAVY_BODY,
        price_2oz=22.00,
        coverage_sqft_per_oz=3.5,
        pigment_code="PB28"
    ),
    "prussian_blue": Paint(
        name="Prussian Blue Hue",
        brand=PaintBrand.GOLDEN,
        rgb=(0, 49, 83),
        paint_type=PaintType.HEAVY_BODY,
        price_2oz=10.50,
        coverage_sqft_per_oz=3.0,
        is_transparent=True,
        pigment_code="PB27"
    ),
    "turquoise_phthalo": Paint(
        name="Turquoise (Phthalo)",
        brand=PaintBrand.GOLDEN,
        rgb=(0, 134, 139),
        paint_type=PaintType.HEAVY_BODY,
        price_2oz=10.50,
        coverage_sqft_per_oz=3.0,
        is_transparent=True,
        pigment_code="PB15:3+PG36"
    ),
    "manganese_blue_hue": Paint(
        name="Manganese Blue Hue",
        brand=PaintBrand.GOLDEN,
        rgb=(0, 150, 215),
        paint_type=PaintType.HEAVY_BODY,
        price_2oz=10.50,
        coverage_sqft_per_oz=3.0,
        pigment_code="PB15:3+PW6"
    ),

    # === VIOLETS/PURPLES ===
    "dioxazine_purple": Paint(
        name="Dioxazine Purple",
        brand=PaintBrand.GOLDEN,
        rgb=(93, 36, 131),
        paint_type=PaintType.HEAVY_BODY,
        price_2oz=10.50,
        coverage_sqft_per_oz=2.5,
        is_transparent=True,
        pigment_code="PV23"
    ),
    "ultramarine_violet": Paint(
        name="Ultramarine Violet",
        brand=PaintBrand.GOLDEN,
        rgb=(90, 54, 134),
        paint_type=PaintType.HEAVY_BODY,
        price_2oz=10.50,
        coverage_sqft_per_oz=3.0,
        is_transparent=True,
        pigment_code="PV15"
    ),
    "quinacridone_violet": Paint(
        name="Quinacridone Violet",
        brand=PaintBrand.GOLDEN,
        rgb=(128, 39, 110),
        paint_type=PaintType.HEAVY_BODY,
        price_2oz=14.00,
        coverage_sqft_per_oz=2.5,
        is_transparent=True,
        pigment_code="PV19"
    ),
    "prism_violet": Paint(
        name="Prism Violet",
        brand=PaintBrand.GOLDEN,
        rgb=(45, 42, 75),
        paint_type=PaintType.HEAVY_BODY,
        price_2oz=10.50,
        coverage_sqft_per_oz=3.5,
        pigment_code="PV23+PBk7"
    ),

    # === EARTH TONES ===
    "burnt_sienna": Paint(
        name="Burnt Sienna",
        brand=PaintBrand.GOLDEN,
        rgb=(138, 54, 15),
        paint_type=PaintType.HEAVY_BODY,
        price_2oz=8.50,
        coverage_sqft_per_oz=4.0,
        is_transparent=True,
        pigment_code="PBr7"
    ),
    "burnt_umber": Paint(
        name="Burnt Umber",
        brand=PaintBrand.GOLDEN,
        rgb=(138, 51, 36),
        paint_type=PaintType.HEAVY_BODY,
        price_2oz=8.50,
        coverage_sqft_per_oz=4.0,
        is_transparent=True,
        pigment_code="PBr7"
    ),
    "raw_sienna": Paint(
        name="Raw Sienna",
        brand=PaintBrand.GOLDEN,
        rgb=(199, 97, 20),
        paint_type=PaintType.HEAVY_BODY,
        price_2oz=8.50,
        coverage_sqft_per_oz=4.0,
        is_transparent=True,
        pigment_code="PBr7"
    ),
    "raw_umber": Paint(
        name="Raw Umber",
        brand=PaintBrand.GOLDEN,
        rgb=(115, 74, 18),
        paint_type=PaintType.HEAVY_BODY,
        price_2oz=8.50,
        coverage_sqft_per_oz=4.0,
        is_transparent=True,
        pigment_code="PBr7"
    ),
    "van_dyke_brown": Paint(
        name="Van Dyke Brown",
        brand=PaintBrand.LIQUITEX,
        rgb=(65, 43, 32),
        paint_type=PaintType.HEAVY_BODY,
        price_2oz=8.50,
        coverage_sqft_per_oz=4.0,
        pigment_code="PBr7+PBk7"
    ),
    "dark_sienna": Paint(
        name="Dark Sienna",
        brand=PaintBrand.LIQUITEX,
        rgb=(60, 20, 10),
        paint_type=PaintType.HEAVY_BODY,
        price_2oz=8.50,
        coverage_sqft_per_oz=4.0,
        pigment_code="PBr7+PBk7"
    ),

    # === GRAYS ===
    "paynes_gray": Paint(
        name="Payne's Gray",
        brand=PaintBrand.GOLDEN,
        rgb=(83, 104, 120),
        paint_type=PaintType.HEAVY_BODY,
        price_2oz=10.50,
        coverage_sqft_per_oz=4.0,
        is_transparent=True,
        pigment_code="PB29+PBk9"
    ),
    "neutral_gray_n5": Paint(
        name="Neutral Gray N5",
        brand=PaintBrand.GOLDEN,
        rgb=(128, 128, 128),
        paint_type=PaintType.HEAVY_BODY,
        price_2oz=8.50,
        coverage_sqft_per_oz=4.0,
        pigment_code="PW6+PBk7"
    ),

    # === SPECIALTY ===
    "iridescent_gold": Paint(
        name="Iridescent Gold (Fine)",
        brand=PaintBrand.GOLDEN,
        rgb=(212, 175, 55),
        paint_type=PaintType.HEAVY_BODY,
        price_2oz=12.00,
        coverage_sqft_per_oz=3.0,
        is_transparent=True,
    ),
    "fluorescent_pink": Paint(
        name="Fluorescent Pink",
        brand=PaintBrand.GOLDEN,
        rgb=(255, 20, 147),
        paint_type=PaintType.HEAVY_BODY,
        price_2oz=12.00,
        coverage_sqft_per_oz=3.0,
    ),
    "fluorescent_green": Paint(
        name="Fluorescent Green",
        brand=PaintBrand.GOLDEN,
        rgb=(0, 255, 0),
        paint_type=PaintType.HEAVY_BODY,
        price_2oz=12.00,
        coverage_sqft_per_oz=3.0,
    ),
}


# Essential mixing set - minimum paints needed to mix most colors
ESSENTIAL_MIXING_SET = [
    "titanium_white",
    "mars_black",
    "cadmium_yellow_medium",
    "cadmium_red_medium",
    "quinacridone_magenta",
    "phthalo_blue_gs",
    "phthalo_green_bs",
    "ultramarine_blue",
    "burnt_sienna",
    "yellow_ochre",
]


def get_paint(name: str) -> Paint:
    """Get a paint by its key name."""
    return PAINT_DATABASE.get(name)


def get_all_paints() -> Dict[str, Paint]:
    """Get all paints in the database."""
    return PAINT_DATABASE


def get_paints_by_brand(brand: PaintBrand) -> Dict[str, Paint]:
    """Get all paints from a specific brand."""
    return {k: v for k, v in PAINT_DATABASE.items() if v.brand == brand}


def get_primary_colors() -> Dict[str, Paint]:
    """Get all primary mixing colors."""
    return {k: v for k, v in PAINT_DATABASE.items() if v.is_primary}


def get_transparent_colors() -> Dict[str, Paint]:
    """Get all transparent colors (good for glazing)."""
    return {k: v for k, v in PAINT_DATABASE.items() if v.is_transparent}


def rgb_to_lab(rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
    """Convert RGB to LAB color space for better color matching."""
    import numpy as np

    # Normalize RGB
    r, g, b = [x / 255.0 for x in rgb]

    # sRGB to linear RGB
    def linearize(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = linearize(r), linearize(g), linearize(b)

    # Linear RGB to XYZ
    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

    # XYZ to LAB (D65 illuminant)
    x /= 0.95047
    y /= 1.00000
    z /= 1.08883

    def f(t):
        return t ** (1/3) if t > 0.008856 else (7.787 * t) + (16/116)

    L = (116 * f(y)) - 16
    a = 500 * (f(x) - f(y))
    b_val = 200 * (f(y) - f(z))

    return (L, a, b_val)


def color_distance_lab(rgb1: Tuple[int, int, int], rgb2: Tuple[int, int, int]) -> float:
    """Calculate perceptual color distance using CIEDE2000 approximation."""
    lab1 = rgb_to_lab(rgb1)
    lab2 = rgb_to_lab(rgb2)

    # Simple Euclidean in LAB space (good approximation)
    return ((lab1[0] - lab2[0])**2 +
            (lab1[1] - lab2[1])**2 +
            (lab1[2] - lab2[2])**2) ** 0.5


if __name__ == "__main__":
    print("Paint Database")
    print("=" * 50)
    print(f"Total paints: {len(PAINT_DATABASE)}")
    print(f"Primary colors: {len(get_primary_colors())}")
    print(f"Transparent colors: {len(get_transparent_colors())}")
    print()
    print("Essential Mixing Set:")
    for paint_key in ESSENTIAL_MIXING_SET:
        paint = get_paint(paint_key)
        print(f"  - {paint.name} (${paint.price_2oz:.2f}/2oz)")
