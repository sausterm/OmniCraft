"""
Scene Context Analyzer

Analyzes an image to determine:
- Time of day (dawn, day, dusk, night)
- Weather/atmosphere (clear, cloudy, foggy, snowy, rainy)
- Setting (outdoor nature, outdoor urban, indoor)
- Lighting conditions (front-lit, back-lit, dramatic, flat)
- Mood (warm, cool, neutral, dramatic)

This context informs how each semantic region should be painted.
"""

import numpy as np
import cv2
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum


class TimeOfDay(Enum):
    DAWN = "dawn"
    DAY = "day"
    GOLDEN_HOUR = "golden_hour"
    DUSK = "dusk"
    NIGHT = "night"


class Weather(Enum):
    CLEAR = "clear"
    CLOUDY = "cloudy"
    OVERCAST = "overcast"
    FOGGY = "foggy"
    SNOWY = "snowy"
    RAINY = "rainy"
    STORMY = "stormy"


class Setting(Enum):
    NATURE_OPEN = "nature_open"       # Fields, meadows, beaches
    NATURE_FOREST = "nature_forest"   # Woods, dense vegetation
    NATURE_MOUNTAIN = "nature_mountain"
    NATURE_WATER = "nature_water"     # Lakes, ocean, rivers
    URBAN_CITY = "urban_city"
    URBAN_SUBURBAN = "urban_suburban"
    INDOOR = "indoor"


class LightingType(Enum):
    FRONT_LIT = "front_lit"           # Light from viewer's direction
    BACK_LIT = "back_lit"             # Silhouette, light behind subject
    SIDE_LIT = "side_lit"             # Dramatic shadows
    TOP_LIT = "top_lit"               # Midday sun
    AMBIENT = "ambient"               # Soft, diffused
    DRAMATIC = "dramatic"             # High contrast
    LOW_KEY = "low_key"               # Mostly dark
    HIGH_KEY = "high_key"             # Mostly bright


class Mood(Enum):
    WARM = "warm"
    COOL = "cool"
    NEUTRAL = "neutral"
    DRAMATIC = "dramatic"
    PEACEFUL = "peaceful"
    ENERGETIC = "energetic"


@dataclass
class SceneContext:
    """Complete scene context analysis."""
    time_of_day: TimeOfDay
    weather: Weather
    setting: Setting
    lighting: LightingType
    mood: Mood

    # Numeric values for fine-tuning
    avg_luminosity: float          # 0-1
    luminosity_range: Tuple[float, float]  # (min, max)
    color_temperature: float       # <0.5 = cool, >0.5 = warm
    contrast: float                # 0-1
    saturation: float              # 0-1

    # Dominant colors
    dominant_hue: float            # 0-360
    sky_color: Optional[Tuple[int, int, int]] = None

    # Spatial light direction estimate
    light_direction: str = "unknown"  # top, bottom, left, right, center

    # Confidence
    confidence: float = 0.5


class SceneContextAnalyzer:
    """
    Analyzes images to determine scene context for intelligent painting strategies.
    """

    def __init__(self, image: np.ndarray):
        """
        Initialize with RGB image.
        """
        self.image = image
        self.h, self.w = image.shape[:2]

        # Precompute color spaces
        self.hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        self.lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

        # Extract channels
        self.hue = self.hsv[:, :, 0] / 179.0  # Normalize to 0-1
        self.saturation = self.hsv[:, :, 1] / 255.0
        self.value = self.hsv[:, :, 2] / 255.0
        self.luminosity = self.lab[:, :, 0] / 255.0

    def analyze(self) -> SceneContext:
        """
        Perform complete scene analysis.
        """
        # Basic statistics
        avg_lum = float(np.mean(self.luminosity))
        lum_range = (float(np.percentile(self.luminosity, 5)),
                     float(np.percentile(self.luminosity, 95)))
        avg_sat = float(np.mean(self.saturation))
        contrast = lum_range[1] - lum_range[0]

        # Color temperature (warm vs cool)
        color_temp = self._analyze_color_temperature()

        # Dominant hue
        dominant_hue = self._get_dominant_hue()

        # Sky color (from top portion)
        sky_color = self._get_sky_color()

        # Determine context attributes
        time_of_day = self._detect_time_of_day(avg_lum, color_temp, sky_color)
        weather = self._detect_weather(avg_lum, contrast, avg_sat, sky_color)
        setting = self._detect_setting()
        lighting = self._detect_lighting(avg_lum, contrast, lum_range)
        mood = self._detect_mood(color_temp, contrast, avg_sat, time_of_day)

        # Light direction
        light_dir = self._estimate_light_direction()

        return SceneContext(
            time_of_day=time_of_day,
            weather=weather,
            setting=setting,
            lighting=lighting,
            mood=mood,
            avg_luminosity=avg_lum,
            luminosity_range=lum_range,
            color_temperature=color_temp,
            contrast=contrast,
            saturation=avg_sat,
            dominant_hue=dominant_hue * 360,
            sky_color=sky_color,
            light_direction=light_dir,
            confidence=0.7,  # Could be improved with ML
        )

    def _analyze_color_temperature(self) -> float:
        """
        Analyze color temperature. Returns 0-1 (0=very cool, 1=very warm).
        """
        # Use LAB a and b channels
        # a: green(-) to red(+)
        # b: blue(-) to yellow(+)
        a_channel = self.lab[:, :, 1].astype(float) - 128
        b_channel = self.lab[:, :, 2].astype(float) - 128

        # Warm colors have positive a (red) and positive b (yellow)
        # Cool colors have negative a (green) and negative b (blue)
        warmth = (np.mean(a_channel) + np.mean(b_channel)) / 2

        # Normalize to 0-1
        return float(np.clip((warmth + 50) / 100, 0, 1))

    def _get_dominant_hue(self) -> float:
        """Get dominant hue (0-1)."""
        # Weight by saturation (ignore desaturated pixels)
        weights = self.saturation.flatten()
        hues = self.hue.flatten()

        if np.sum(weights) > 0:
            # Circular mean for hue
            sin_sum = np.sum(np.sin(hues * 2 * np.pi) * weights)
            cos_sum = np.sum(np.cos(hues * 2 * np.pi) * weights)
            dominant = np.arctan2(sin_sum, cos_sum) / (2 * np.pi)
            return float(dominant % 1.0)
        return 0.0

    def _get_sky_color(self) -> Optional[Tuple[int, int, int]]:
        """Get average color of the sky (top 20% of image)."""
        sky_region = self.image[:int(self.h * 0.2), :]
        avg_color = np.mean(sky_region, axis=(0, 1))
        return tuple(avg_color.astype(int))

    def _detect_time_of_day(
        self,
        avg_lum: float,
        color_temp: float,
        sky_color: Optional[Tuple[int, int, int]]
    ) -> TimeOfDay:
        """Detect time of day from luminosity and color."""

        # Night: very dark
        if avg_lum < 0.2:
            return TimeOfDay.NIGHT

        # Check for golden hour / sunset colors
        if sky_color:
            r, g, b = sky_color
            # Orange/pink sky suggests golden hour or dusk
            if r > 150 and r > b * 1.3 and g < r:
                if avg_lum > 0.5:
                    return TimeOfDay.GOLDEN_HOUR
                else:
                    return TimeOfDay.DUSK

            # Purple/deep blue suggests dawn or dusk
            if b > r and b > 100 and avg_lum < 0.4:
                return TimeOfDay.DAWN if color_temp < 0.4 else TimeOfDay.DUSK

        # Warm overall color temp with good light = golden hour
        if color_temp > 0.6 and 0.4 < avg_lum < 0.7:
            return TimeOfDay.GOLDEN_HOUR

        # Default to day
        return TimeOfDay.DAY

    def _detect_weather(
        self,
        avg_lum: float,
        contrast: float,
        avg_sat: float,
        sky_color: Optional[Tuple[int, int, int]]
    ) -> Weather:
        """Detect weather/atmosphere conditions."""

        # Check for snow (high luminosity, low saturation, bluish whites)
        white_pixels = (self.luminosity > 0.8) & (self.saturation < 0.2)
        if np.mean(white_pixels) > 0.3:
            return Weather.SNOWY

        # Foggy: low contrast, desaturated
        if contrast < 0.3 and avg_sat < 0.3:
            return Weather.FOGGY

        # Overcast: medium luminosity, low contrast, gray sky
        if sky_color:
            r, g, b = sky_color
            gray_sky = abs(r - g) < 20 and abs(g - b) < 20
            if gray_sky and contrast < 0.5:
                if avg_lum > 0.5:
                    return Weather.OVERCAST
                else:
                    return Weather.CLOUDY

        # Stormy: dark, dramatic contrast
        if avg_lum < 0.4 and contrast > 0.5:
            return Weather.STORMY

        # Clear: good contrast, good saturation
        if contrast > 0.4 and avg_sat > 0.3:
            return Weather.CLEAR

        return Weather.CLOUDY

    def _detect_setting(self) -> Setting:
        """Detect the setting type."""
        # Analyze color distribution

        # Green detection (nature indicator)
        green_mask = (self.hue > 0.2) & (self.hue < 0.45) & (self.saturation > 0.2)
        green_ratio = np.mean(green_mask)

        # Blue detection (sky/water indicator)
        blue_mask = (self.hue > 0.5) & (self.hue < 0.7) & (self.saturation > 0.2)
        blue_ratio = np.mean(blue_mask)

        # Brown/earth tones
        brown_mask = (self.hue > 0.05) & (self.hue < 0.15) & (self.saturation > 0.2) & (self.value < 0.6)
        brown_ratio = np.mean(brown_mask)

        # Gray detection (urban/indoor indicator)
        gray_mask = self.saturation < 0.15
        gray_ratio = np.mean(gray_mask)

        # High green = nature
        if green_ratio > 0.3:
            if brown_ratio > 0.15:
                return Setting.NATURE_FOREST
            return Setting.NATURE_OPEN

        # Lots of blue in upper half = sky/water
        upper_blue = np.mean(blue_mask[:self.h//2, :])
        if upper_blue > 0.3:
            return Setting.NATURE_OPEN

        # Lots of gray = urban or indoor
        if gray_ratio > 0.4:
            return Setting.URBAN_CITY

        # Default
        return Setting.NATURE_OPEN

    def _detect_lighting(
        self,
        avg_lum: float,
        contrast: float,
        lum_range: Tuple[float, float]
    ) -> LightingType:
        """Detect lighting conditions."""

        # Low key (mostly dark)
        if avg_lum < 0.3 and lum_range[1] < 0.6:
            return LightingType.LOW_KEY

        # High key (mostly bright)
        if avg_lum > 0.7 and lum_range[0] > 0.3:
            return LightingType.HIGH_KEY

        # Dramatic (high contrast)
        if contrast > 0.7:
            return LightingType.DRAMATIC

        # Ambient (low contrast, medium luminosity)
        if contrast < 0.4 and 0.3 < avg_lum < 0.7:
            return LightingType.AMBIENT

        # Check for backlighting (bright top, dark bottom with subjects)
        top_lum = np.mean(self.luminosity[:self.h//3, :])
        bottom_lum = np.mean(self.luminosity[2*self.h//3:, :])
        if top_lum > bottom_lum + 0.2:
            return LightingType.BACK_LIT

        return LightingType.FRONT_LIT

    def _detect_mood(
        self,
        color_temp: float,
        contrast: float,
        avg_sat: float,
        time_of_day: TimeOfDay
    ) -> Mood:
        """Detect overall mood."""

        # Dramatic: high contrast or night
        if contrast > 0.6 or time_of_day == TimeOfDay.NIGHT:
            return Mood.DRAMATIC

        # Warm: warm color temperature
        if color_temp > 0.6:
            return Mood.WARM

        # Cool: cool color temperature
        if color_temp < 0.4:
            return Mood.COOL

        # Peaceful: low contrast, moderate saturation
        if contrast < 0.4 and avg_sat < 0.5:
            return Mood.PEACEFUL

        # Energetic: high saturation
        if avg_sat > 0.6:
            return Mood.ENERGETIC

        return Mood.NEUTRAL

    def _estimate_light_direction(self) -> str:
        """Estimate the primary light direction."""
        # Divide image into quadrants and find brightest
        mid_h, mid_w = self.h // 2, self.w // 2

        quadrants = {
            "top-left": np.mean(self.luminosity[:mid_h, :mid_w]),
            "top-right": np.mean(self.luminosity[:mid_h, mid_w:]),
            "bottom-left": np.mean(self.luminosity[mid_h:, :mid_w]),
            "bottom-right": np.mean(self.luminosity[mid_h:, mid_w:]),
        }

        brightest = max(quadrants, key=quadrants.get)

        # Map to direction
        if "top" in brightest:
            if "left" in brightest:
                return "top-left"
            return "top-right"
        else:
            if "left" in brightest:
                return "bottom-left"
            return "bottom-right"


def analyze_scene(image: np.ndarray) -> SceneContext:
    """Convenience function to analyze a scene."""
    analyzer = SceneContextAnalyzer(image)
    return analyzer.analyze()
