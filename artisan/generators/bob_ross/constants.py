"""Constants for Bob Ross style instruction generation."""

from enum import Enum


class BrushType(Enum):
    """Bob Ross brush types"""
    TWO_INCH = "2-inch brush"
    ONE_INCH = "1-inch brush"
    FAN_BRUSH = "fan brush"
    LINER_BRUSH = "liner brush (script)"
    FILBERT = "filbert brush"
    PALETTE_KNIFE = "palette knife"
    ROUND = "small round brush"


class StrokeMotion(Enum):
    """Brush stroke motions"""
    CRISS_CROSS = "criss-cross strokes"
    VERTICAL_PULL = "pull down vertically"
    HORIZONTAL_PULL = "pull horizontally"
    TAP = "tap gently"
    STIPPLE = "stipple (pounce up and down)"
    CIRCULAR = "small circular motions"
    BLEND = "blend softly back and forth"
    LOAD_AND_PULL = "load brush and pull in one stroke"
    FLICK = "flick outward from center"


# Bob Ross paint color names (mapped from common colors)
PAINT_NAMES = {
    'black': 'Midnight Black',
    'white': 'Titanium White',
    'blue': 'Prussian Blue',
    'light_blue': 'Pthalo Blue',
    'green': 'Sap Green',
    'light_green': 'Cadmium Yellow + Sap Green mix',
    'yellow': 'Cadmium Yellow',
    'orange': 'Cadmium Yellow + Alizarin Crimson',
    'red': 'Alizarin Crimson',
    'magenta': 'Alizarin Crimson + a touch of Pthalo Blue',
    'purple': 'Alizarin Crimson + Prussian Blue',
    'brown': 'Van Dyke Brown',
    'pink': 'Titanium White + Alizarin Crimson',
    'dark_sienna': 'Dark Sienna',
}


# Encouragements Bob Ross would say
ENCOURAGEMENTS = [
    "There are no mistakes, only happy accidents.",
    "You can do this. Anyone can paint.",
    "Let's get a little crazy here.",
    "This is your world. You can do anything you want.",
    "We don't make mistakes, we just have happy accidents.",
    "Take your time. There's no pressure here.",
    "Just let it happen. Let your brush do the work.",
    "Isn't that fantastic? Look at that.",
    "That's what makes painting so wonderful.",
    "Now then, let's get a little braver.",
    "See how easy that was?",
    "We're just having fun here.",
    "Let your imagination take over.",
    "Don't be afraid of the canvas.",
    "Every day is a good day when you paint.",
]
