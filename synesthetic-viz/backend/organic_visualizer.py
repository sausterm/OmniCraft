"""
Organic Visualizer - Clean, flowing audio-reactive visualization
Focus: Simplicity, organic movement, mood-appropriate colors
"""

import numpy as np
import librosa
import cv2
import os
import subprocess
import tempfile
from typing import Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum
import colorsys
import math


# ============================================================================
# COLOR PALETTES - Mood-based, not rainbow
# ============================================================================

class ColorPalette:
    """Curated color palettes based on musical mood."""

    # Minor key - cooler, moodier
    MINOR_DARK = [
        (20, 25, 45),      # Deep blue-black
        (35, 50, 80),      # Navy
        (60, 80, 120),     # Steel blue
        (100, 120, 160),   # Dusty blue
        (140, 100, 140),   # Muted purple
    ]

    MINOR_WARM = [
        (30, 20, 25),      # Dark maroon
        (60, 35, 40),      # Deep burgundy
        (90, 50, 55),      # Wine
        (120, 70, 75),     # Dusty rose
        (80, 60, 50),      # Warm brown
    ]

    # Major key - warmer, brighter
    MAJOR_BRIGHT = [
        (40, 35, 25),      # Warm dark
        (80, 60, 40),      # Bronze
        (140, 100, 60),    # Gold
        (180, 140, 80),    # Soft gold
        (200, 170, 120),   # Cream
    ]

    MAJOR_COOL = [
        (25, 40, 45),      # Teal dark
        (40, 70, 80),      # Ocean
        (60, 110, 120),    # Aqua
        (90, 150, 160),    # Seafoam
        (130, 180, 180),   # Light teal
    ]

    # High energy
    ENERGETIC = [
        (45, 25, 35),      # Dark magenta
        (90, 40, 60),      # Deep pink
        (140, 60, 80),     # Rose
        (180, 90, 100),    # Coral
        (200, 130, 110),   # Peach
    ]

    # Calm/ambient
    AMBIENT = [
        (15, 20, 30),      # Near black
        (25, 35, 50),      # Dark blue
        (40, 55, 75),      # Slate
        (60, 80, 100),     # Gray blue
        (85, 105, 125),    # Silver blue
    ]

    @staticmethod
    def get_palette(mode: str, energy: float, valence: float) -> List[Tuple[int, int, int]]:
        """Select palette based on musical characteristics."""
        if energy > 0.7:
            return ColorPalette.ENERGETIC
        elif energy < 0.3:
            return ColorPalette.AMBIENT
        elif mode == 'minor':
            return ColorPalette.MINOR_DARK if valence < 0.5 else ColorPalette.MINOR_WARM
        else:
            return ColorPalette.MAJOR_BRIGHT if valence > 0.5 else ColorPalette.MAJOR_COOL

    @staticmethod
    def interpolate(palette: List[Tuple], t: float) -> Tuple[int, int, int]:
        """Smoothly interpolate between palette colors."""
        t = max(0, min(1, t))
        idx = t * (len(palette) - 1)
        i = int(idx)
        frac = idx - i

        if i >= len(palette) - 1:
            return palette[-1]

        c1, c2 = palette[i], palette[i + 1]
        return (
            int(c1[0] + (c2[0] - c1[0]) * frac),
            int(c1[1] + (c2[1] - c1[1]) * frac),
            int(c1[2] + (c2[2] - c1[2]) * frac),
        )


# ============================================================================
# SMOOTH AUDIO FEATURES
# ============================================================================

@dataclass
class SmoothFeatures:
    """Audio features with heavy smoothing for organic movement."""
    volume: np.ndarray
    bass: np.ndarray
    mids: np.ndarray
    highs: np.ndarray
    brightness: np.ndarray
    beats: np.ndarray
    energy: np.ndarray

    tempo: float
    key: str
    mode: str

    duration: float
    num_frames: int


class SmoothAudioAnalyzer:
    """Extract heavily smoothed audio features for organic visuals."""

    def __init__(self, fps: int = 30):
        self.fps = fps
        self.sample_rate = 22050

    def analyze(self, audio_path: str) -> SmoothFeatures:
        """Analyze audio with heavy smoothing."""
        print(f"[Analyzer] Loading: {audio_path}")

        y, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        duration = len(y) / sr
        hop_length = int(sr / self.fps)

        print("[Analyzer] Extracting features...")

        # Get mel spectrogram
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, hop_length=hop_length)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)

        # Frequency bands
        bass = np.mean(mel_norm[:8, :], axis=0)
        mids = np.mean(mel_norm[8:32, :], axis=0)
        highs = np.mean(mel_norm[32:, :], axis=0)

        # Volume
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        volume = rms / (rms.max() + 1e-8)

        # Brightness
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
        brightness = (centroid - centroid.min()) / (centroid.max() - centroid.min() + 1e-8)

        # Beat tracking
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
        tempo = float(tempo) if np.isscalar(tempo) else float(tempo[0])

        n_frames = len(volume)
        beats = np.zeros(n_frames)
        beat_frames = beat_frames[beat_frames < n_frames]
        beats[beat_frames] = 1.0

        # Key detection
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
        chroma_avg = np.mean(chroma, axis=1)
        key_idx = np.argmax(chroma_avg)
        keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        key = keys[key_idx]

        # Mode detection
        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        major_corr = np.corrcoef(np.roll(major_profile, key_idx), chroma_avg)[0, 1]
        minor_corr = np.corrcoef(np.roll(minor_profile, key_idx), chroma_avg)[0, 1]
        mode = 'major' if major_corr > minor_corr else 'minor'

        # Energy estimation
        energy = volume * 0.5 + brightness * 0.3 + (bass + highs) * 0.1

        print(f"[Analyzer] {key} {mode}, {tempo:.1f} BPM")

        # HEAVY SMOOTHING - this is the key to organic movement
        smooth_window = int(self.fps * 0.5)  # 500ms smoothing window

        def smooth(arr, window):
            """Apply heavy gaussian smoothing."""
            kernel = np.exp(-np.linspace(-2, 2, window)**2)
            kernel = kernel / kernel.sum()
            return np.convolve(arr, kernel, mode='same')

        def smooth_beats(arr, decay_frames):
            """Smooth beat triggers into organic swells."""
            result = np.zeros_like(arr)
            for i in range(len(arr)):
                if arr[i] > 0.5:
                    # Create a smooth swell on beats
                    for j in range(decay_frames):
                        if i + j < len(result):
                            # Quick attack, slow decay
                            t = j / decay_frames
                            envelope = np.exp(-t * 3) * (1 - np.exp(-t * 20))
                            result[i + j] = max(result[i + j], envelope)
            return result

        volume = smooth(volume, smooth_window)
        bass = smooth(bass, smooth_window)
        mids = smooth(mids, smooth_window)
        highs = smooth(highs, smooth_window)
        brightness = smooth(brightness, smooth_window)
        energy = smooth(energy, smooth_window)
        beats = smooth_beats(beats, int(self.fps * 0.8))  # 800ms beat decay

        # Normalize after smoothing
        for arr in [volume, bass, mids, highs, brightness, energy, beats]:
            arr /= (arr.max() + 1e-8)

        print(f"[Analyzer] {n_frames} frames extracted")

        return SmoothFeatures(
            volume=volume,
            bass=bass,
            mids=mids,
            highs=highs,
            brightness=brightness,
            beats=beats,
            energy=energy,
            tempo=tempo,
            key=key,
            mode=mode,
            duration=duration,
            num_frames=n_frames
        )


# ============================================================================
# ORGANIC VISUAL ELEMENTS
# ============================================================================

class FlowingParticles:
    """
    Gentle, flowing particle system.
    Particles drift smoothly, responding organically to audio.
    """

    def __init__(self, width: int, height: int, count: int = 800):
        self.width = width
        self.height = height
        self.count = count
        self.center = np.array([width / 2, height / 2])

        # Initialize particles in a soft cloud
        angles = np.random.uniform(0, 2 * np.pi, count)
        radii = np.random.exponential(100, count)  # Concentrated center, sparse edges

        self.positions = np.zeros((count, 2))
        self.positions[:, 0] = self.center[0] + radii * np.cos(angles)
        self.positions[:, 1] = self.center[1] + radii * np.sin(angles)

        self.base_radii = radii.copy()
        self.angles = angles.copy()
        self.sizes = np.random.uniform(1, 4, count)
        self.alphas = np.random.uniform(0.3, 0.8, count)

        # Smooth state
        self.rotation = 0.0
        self.expansion = 1.0
        self.target_expansion = 1.0

    def update(self, frame_idx: int, features: SmoothFeatures, palette: List[Tuple]):
        """Update with smooth, organic movement."""
        f = min(frame_idx, features.num_frames - 1)

        bass = features.bass[f]
        volume = features.volume[f]
        beats = features.beats[f]
        brightness = features.brightness[f]

        # Smooth rotation - very gentle
        rotation_speed = 0.002 + volume * 0.008
        self.rotation += rotation_speed

        # Smooth expansion on beats
        self.target_expansion = 1.0 + beats * 0.4 + bass * 0.2
        self.expansion += (self.target_expansion - self.expansion) * 0.05  # Ease toward target

        # Update particle positions with smooth orbiting
        for i in range(self.count):
            # Gentle orbital motion
            self.angles[i] += rotation_speed * (0.5 + self.base_radii[i] / 200)

            # Breathing expansion
            current_radius = self.base_radii[i] * self.expansion

            # Add gentle wave motion
            wave = np.sin(frame_idx * 0.02 + i * 0.1) * 10 * volume

            self.positions[i, 0] = self.center[0] + (current_radius + wave) * np.cos(self.angles[i])
            self.positions[i, 1] = self.center[1] + (current_radius + wave) * np.sin(self.angles[i])

        # Update sizes smoothly
        self.sizes = 1.5 + volume * 3 + beats * 2

        # Update alphas - brighter on beats
        self.alphas = 0.3 + volume * 0.4 + beats * 0.3

    def render(self, canvas: np.ndarray, features: SmoothFeatures, frame_idx: int, palette: List[Tuple]):
        """Render particles with palette colors."""
        f = min(frame_idx, features.num_frames - 1)
        brightness = features.brightness[f]
        volume = features.volume[f]

        for i in range(self.count):
            x, y = int(self.positions[i, 0]), int(self.positions[i, 1])

            if 0 <= x < self.width and 0 <= y < self.height:
                # Color based on distance from center and brightness
                dist = np.sqrt((x - self.center[0])**2 + (y - self.center[1])**2)
                max_dist = min(self.width, self.height) / 2
                t = min(1.0, dist / max_dist) * (0.5 + brightness * 0.5)

                color = ColorPalette.interpolate(palette, t)

                # Apply alpha
                alpha = self.alphas[i] if np.isscalar(self.alphas) else self.alphas
                color = tuple(int(c * alpha) for c in color)

                size = max(1, int(self.sizes if np.isscalar(self.sizes) else self.sizes))
                cv2.circle(canvas, (x, y), size, color[::-1], -1, cv2.LINE_AA)

        return canvas


class BreathingRing:
    """
    A single breathing ring that expands/contracts with the music.
    Simple, elegant, organic.
    """

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.center = (width // 2, height // 2)

        self.base_radius = min(width, height) * 0.25
        self.current_radius = self.base_radius
        self.target_radius = self.base_radius

        self.thickness = 2
        self.points = 180

    def update(self, frame_idx: int, features: SmoothFeatures):
        """Update ring with breathing motion."""
        f = min(frame_idx, features.num_frames - 1)

        volume = features.volume[f]
        bass = features.bass[f]
        beats = features.beats[f]

        # Target radius expands with volume and beats
        self.target_radius = self.base_radius * (0.8 + volume * 0.4 + beats * 0.3 + bass * 0.2)

        # Smooth easing toward target
        self.current_radius += (self.target_radius - self.current_radius) * 0.08

        # Thickness varies with energy
        self.thickness = 1 + int(volume * 4 + beats * 3)

    def render(self, canvas: np.ndarray, features: SmoothFeatures, frame_idx: int, palette: List[Tuple]):
        """Render the breathing ring."""
        f = min(frame_idx, features.num_frames - 1)

        brightness = features.brightness[f]
        volume = features.volume[f]
        mids = features.mids[f]

        cx, cy = self.center

        # Draw ring with subtle variations
        points = []
        for i in range(self.points):
            angle = (i / self.points) * 2 * np.pi

            # Subtle wave deformation based on mids
            wave = np.sin(angle * 4 + frame_idx * 0.03) * mids * 15
            radius = self.current_radius + wave

            x = int(cx + np.cos(angle) * radius)
            y = int(cy + np.sin(angle) * radius)
            points.append([x, y])

        points = np.array(points, dtype=np.int32)

        # Color from palette based on volume
        color = ColorPalette.interpolate(palette, 0.3 + volume * 0.5)

        # Draw the ring
        cv2.polylines(canvas, [points], True, color[::-1], self.thickness, cv2.LINE_AA)

        # Inner glow ring
        inner_color = ColorPalette.interpolate(palette, 0.1 + brightness * 0.3)
        inner_alpha = 0.3 + volume * 0.3
        inner_color = tuple(int(c * inner_alpha) for c in inner_color)

        inner_points = []
        for i in range(self.points):
            angle = (i / self.points) * 2 * np.pi
            wave = np.sin(angle * 4 + frame_idx * 0.03) * mids * 10
            radius = self.current_radius * 0.7 + wave
            x = int(cx + np.cos(angle) * radius)
            y = int(cy + np.sin(angle) * radius)
            inner_points.append([x, y])

        inner_points = np.array(inner_points, dtype=np.int32)
        cv2.polylines(canvas, [inner_points], True, inner_color[::-1], 1, cv2.LINE_AA)

        return canvas


class GentleWaves:
    """
    Horizontal waves that flow across the screen.
    Calm, meditative, responds to frequency bands.
    """

    def __init__(self, width: int, height: int, num_waves: int = 5):
        self.width = width
        self.height = height
        self.num_waves = num_waves
        self.phase = 0.0

    def update(self, frame_idx: int, features: SmoothFeatures):
        """Update wave phase."""
        f = min(frame_idx, features.num_frames - 1)

        # Phase advances with music tempo feeling
        tempo_factor = features.tempo / 120.0
        self.phase += 0.02 * tempo_factor + features.volume[f] * 0.01

    def render(self, canvas: np.ndarray, features: SmoothFeatures, frame_idx: int, palette: List[Tuple]):
        """Render flowing waves."""
        f = min(frame_idx, features.num_frames - 1)

        bass = features.bass[f]
        mids = features.mids[f]
        highs = features.highs[f]
        volume = features.volume[f]

        wave_heights = [bass, mids, highs, mids * 0.7, bass * 0.5]

        for w in range(self.num_waves):
            # Each wave at different vertical position
            base_y = int(self.height * (0.3 + w * 0.1))

            # Wave amplitude based on corresponding frequency
            amp = 30 + wave_heights[w % len(wave_heights)] * 80

            # Draw wave
            points = []
            for x in range(0, self.width, 4):
                # Multiple sine waves for organic look
                y = base_y
                y += np.sin(x * 0.01 + self.phase + w) * amp
                y += np.sin(x * 0.02 + self.phase * 1.5 + w * 0.5) * amp * 0.3
                y += np.sin(x * 0.005 + self.phase * 0.5) * amp * 0.5
                points.append([x, int(y)])

            points = np.array(points, dtype=np.int32)

            # Color: deeper waves are darker
            t = 0.2 + (w / self.num_waves) * 0.5 + volume * 0.2
            color = ColorPalette.interpolate(palette, t)

            # Fade alpha for depth
            alpha = 0.3 + (1 - w / self.num_waves) * 0.4
            color = tuple(int(c * alpha) for c in color)

            cv2.polylines(canvas, [points], False, color[::-1], 2, cv2.LINE_AA)

        return canvas


# ============================================================================
# ORGANIC VIDEO GENERATOR
# ============================================================================

class OrganicVisualizer:
    """Generate clean, organic audio-reactive visualizations."""

    MODES = ['particles', 'ring', 'waves', 'particles_ring', 'minimal']

    def __init__(
        self,
        width: int = 1920,
        height: int = 1080,
        fps: int = 30,
        mode: str = 'particles_ring'
    ):
        self.width = width
        self.height = height
        self.fps = fps
        self.mode = mode

        self.analyzer = SmoothAudioAnalyzer(fps=fps)
        self.elements = []

    def setup_elements(self, features: SmoothFeatures):
        """Setup visual elements based on mode."""
        self.elements = []

        if self.mode == 'particles':
            self.elements.append(FlowingParticles(self.width, self.height, count=1000))

        elif self.mode == 'ring':
            self.elements.append(BreathingRing(self.width, self.height))

        elif self.mode == 'waves':
            self.elements.append(GentleWaves(self.width, self.height, num_waves=6))

        elif self.mode == 'particles_ring':
            self.elements.append(FlowingParticles(self.width, self.height, count=600))
            self.elements.append(BreathingRing(self.width, self.height))

        elif self.mode == 'minimal':
            self.elements.append(BreathingRing(self.width, self.height))
            self.elements.append(FlowingParticles(self.width, self.height, count=200))

        print(f"[Visualizer] Mode: {self.mode}, Elements: {len(self.elements)}")

    def render_frame(self, frame_idx: int, features: SmoothFeatures, palette: List[Tuple]) -> np.ndarray:
        """Render a single frame."""
        f = min(frame_idx, features.num_frames - 1)

        # Create dark background
        bg_color = palette[0]
        canvas = np.full((self.height, self.width, 3), bg_color[::-1], dtype=np.uint8)

        # Update and render each element
        for element in self.elements:
            element.update(frame_idx, features)
            canvas = element.render(canvas, features, frame_idx, palette)

        # Subtle vignette
        canvas = self._apply_vignette(canvas, strength=0.3)

        # Very subtle bloom on high energy
        energy = features.energy[f]
        if energy > 0.5:
            canvas = self._apply_bloom(canvas, strength=0.2 * energy)

        return canvas

    def _apply_vignette(self, canvas: np.ndarray, strength: float = 0.3) -> np.ndarray:
        """Apply subtle vignette."""
        rows, cols = canvas.shape[:2]
        X = np.arange(0, cols)
        Y = np.arange(0, rows)
        X, Y = np.meshgrid(X, Y)

        center_x, center_y = cols / 2, rows / 2
        dist = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
        max_dist = np.sqrt(center_x ** 2 + center_y ** 2)

        vignette = 1 - (dist / max_dist) * strength
        vignette = np.clip(vignette, 0, 1)

        return (canvas * vignette[:, :, np.newaxis]).astype(np.uint8)

    def _apply_bloom(self, canvas: np.ndarray, strength: float = 0.3) -> np.ndarray:
        """Apply subtle bloom/glow."""
        blur_size = 21
        blurred = cv2.GaussianBlur(canvas, (blur_size, blur_size), 0)
        return cv2.addWeighted(canvas, 1.0, blurred, strength, 0)

    def generate(self, audio_path: str, output_path: str) -> str:
        """Generate visualization video."""
        print(f"\n{'='*50}")
        print("ORGANIC VISUALIZER")
        print(f"{'='*50}\n")

        # Analyze audio
        features = self.analyzer.analyze(audio_path)

        # Select palette based on mood
        avg_energy = np.mean(features.energy)
        avg_brightness = np.mean(features.brightness)
        palette = ColorPalette.get_palette(features.mode, avg_energy, avg_brightness)

        print(f"[Visualizer] Palette: {'Energetic' if avg_energy > 0.7 else 'Ambient' if avg_energy < 0.3 else features.mode}")

        # Setup elements
        self.setup_elements(features)

        # Create temp video
        temp_video = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(temp_video, fourcc, self.fps, (self.width, self.height))

        print(f"\n[Visualizer] Generating {features.num_frames} frames...")

        for frame_idx in range(features.num_frames):
            frame = self.render_frame(frame_idx, features, palette)
            writer.write(frame)

            if frame_idx % 60 == 0:
                progress = (frame_idx + 1) / features.num_frames * 100
                print(f"Progress: {progress:.1f}%")

        writer.release()
        print("\n[Visualizer] Adding audio...")

        # Combine with audio
        final_output = output_path if output_path.endswith('.mp4') else output_path + '.mp4'

        cmd = [
            'ffmpeg', '-y',
            '-i', temp_video,
            '-i', audio_path,
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '18',
            '-c:a', 'aac',
            '-b:a', '192k',
            '-shortest',
            '-pix_fmt', 'yuv420p',
            final_output
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"\n[Visualizer] Output: {final_output}")
        except Exception as e:
            print(f"[Visualizer] FFmpeg error, saving without audio")
            os.rename(temp_video, final_output)
        finally:
            if os.path.exists(temp_video):
                os.remove(temp_video)

        return final_output


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Organic Audio Visualizer')
    parser.add_argument('audio', help='Input audio file')
    parser.add_argument('-o', '--output', default='organic_output.mp4', help='Output video')
    parser.add_argument('--width', type=int, default=1920, help='Width')
    parser.add_argument('--height', type=int, default=1080, help='Height')
    parser.add_argument('--fps', type=int, default=30, help='FPS')
    parser.add_argument('--mode', choices=OrganicVisualizer.MODES, default='particles_ring',
                       help='Visualization mode')

    args = parser.parse_args()

    viz = OrganicVisualizer(
        width=args.width,
        height=args.height,
        fps=args.fps,
        mode=args.mode
    )

    viz.generate(args.audio, args.output)


if __name__ == '__main__':
    main()
