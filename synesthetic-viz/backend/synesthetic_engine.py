"""
Synesthetic Engine - Advanced Audio-Reactive Visualization System

Multi-layer visualization with source separation, fractal generation,
flow fields, and comprehensive musical analysis.
"""

import numpy as np
import librosa
import cv2
import os
import subprocess
import tempfile
from typing import Optional, Dict, List, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import colorsys
import math


# ============================================================================
# AUDIO ANALYSIS ENGINE
# ============================================================================

@dataclass
class AudioFeatures:
    """Frame-by-frame audio features for visualization."""
    # Source-separated stems (if available)
    drums: np.ndarray = None
    bass: np.ndarray = None
    vocals: np.ndarray = None
    other: np.ndarray = None

    # Frequency bands
    sub_bass: np.ndarray = None      # 20-60 Hz
    bass_band: np.ndarray = None     # 60-250 Hz
    low_mids: np.ndarray = None      # 250-500 Hz
    mids: np.ndarray = None          # 500-2000 Hz
    high_mids: np.ndarray = None     # 2000-4000 Hz
    highs: np.ndarray = None         # 4000-8000 Hz
    brilliance: np.ndarray = None    # 8000+ Hz

    # Musical features
    tempo: float = 120.0
    beats: np.ndarray = None
    downbeats: np.ndarray = None
    key: str = "C"
    mode: str = "major"
    chroma: np.ndarray = None

    # Timbral features
    brightness: np.ndarray = None    # Spectral centroid
    warmth: np.ndarray = None        # Low frequency energy ratio
    roughness: np.ndarray = None     # Spectral flux
    density: np.ndarray = None       # Spectral flatness

    # Dynamics
    volume: np.ndarray = None
    dynamics: np.ndarray = None      # Volume derivative
    onsets: np.ndarray = None
    transients: np.ndarray = None

    # Mood estimation
    energy: np.ndarray = None
    valence: np.ndarray = None       # Happy/sad estimation

    # Metadata
    duration: float = 0.0
    num_frames: int = 0
    fps: int = 30
    sample_rate: int = 22050


class AudioAnalyzer:
    """Advanced audio analysis with source separation."""

    def __init__(self, fps: int = 30, use_source_separation: bool = True):
        self.fps = fps
        self.sample_rate = 22050
        self.use_source_separation = use_source_separation
        self.hop_length = None

    def analyze(self, audio_path: str) -> AudioFeatures:
        """Perform comprehensive audio analysis."""
        print(f"[AudioAnalyzer] Loading: {audio_path}")

        # Load audio
        y, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        duration = len(y) / sr

        # Calculate hop length for desired FPS
        self.hop_length = int(sr / self.fps)

        features = AudioFeatures(
            duration=duration,
            fps=self.fps,
            sample_rate=sr
        )

        # Source separation
        if self.use_source_separation:
            features = self._separate_sources(audio_path, features)

        # Extract all features
        features = self._extract_frequency_bands(y, sr, features)
        features = self._extract_musical_features(y, sr, features)
        features = self._extract_timbral_features(y, sr, features)
        features = self._extract_dynamics(y, sr, features)
        features = self._estimate_mood(features)

        # Ensure consistent frame count
        features.num_frames = self._get_min_frames(features)
        features = self._trim_to_frames(features)

        print(f"[AudioAnalyzer] Extracted {features.num_frames} frames")
        return features

    def _separate_sources(self, audio_path: str, features: AudioFeatures) -> AudioFeatures:
        """Separate audio into stems using Demucs."""
        try:
            import torch
            import torchaudio
            from demucs.pretrained import get_model
            from demucs.apply import apply_model

            print("[AudioAnalyzer] Running source separation (Demucs)...")

            # Detect best available device
            if torch.backends.mps.is_available():
                device = 'mps'
                print("[AudioAnalyzer] Using Apple Silicon GPU (MPS)")
            elif torch.cuda.is_available():
                device = 'cuda'
                print("[AudioAnalyzer] Using NVIDIA GPU (CUDA)")
            else:
                device = 'cpu'
                print("[AudioAnalyzer] Using CPU")

            # Load model
            model = get_model('htdemucs')
            model.to(device)
            model.eval()

            # Load audio for demucs (needs stereo)
            wav, sr = torchaudio.load(audio_path)
            if wav.shape[0] == 1:
                wav = wav.repeat(2, 1)

            # Resample if needed
            if sr != model.samplerate:
                wav = torchaudio.functional.resample(wav, sr, model.samplerate)

            # Apply model on GPU
            with torch.no_grad():
                wav = wav.unsqueeze(0).to(device)  # Add batch dimension, move to device
                sources = apply_model(model, wav, device=device)
                sources = sources.squeeze(0).cpu()  # Remove batch, move to CPU

            # sources shape: [4, 2, samples] -> drums, bass, other, vocals
            stem_names = model.sources
            stems = {}

            for i, name in enumerate(stem_names):
                # Convert to mono and numpy
                stem_mono = sources[i].mean(dim=0).numpy()

                # Extract energy envelope at video frame rate
                hop = int(model.samplerate / self.fps)
                rms = librosa.feature.rms(y=stem_mono, hop_length=hop)[0]
                rms = rms / (rms.max() + 1e-8)
                stems[name] = rms

            features.drums = stems.get('drums', np.zeros(1))
            features.bass = stems.get('bass', np.zeros(1))
            features.vocals = stems.get('vocals', np.zeros(1))
            features.other = stems.get('other', np.zeros(1))

            print(f"[AudioAnalyzer] Source separation complete: {list(stems.keys())}")

        except Exception as e:
            print(f"[AudioAnalyzer] Source separation failed: {e}")
            print("[AudioAnalyzer] Falling back to frequency-based estimation")
            features = self._estimate_sources_from_frequency(features)

        return features

    def _estimate_sources_from_frequency(self, features: AudioFeatures) -> AudioFeatures:
        """Estimate source separation from frequency bands when Demucs fails."""
        # Will be populated by frequency band extraction
        return features

    def _extract_frequency_bands(self, y: np.ndarray, sr: int, features: AudioFeatures) -> AudioFeatures:
        """Extract detailed frequency band information."""
        print("[AudioAnalyzer] Extracting frequency bands...")

        # Mel spectrogram with more bands
        mel = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=128, hop_length=self.hop_length,
            fmin=20, fmax=sr//2
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)

        # Define frequency band ranges (in mel bins, roughly)
        # 128 mel bins spanning 20Hz to 11025Hz
        n_mels = mel_norm.shape[0]

        features.sub_bass = np.mean(mel_norm[:int(n_mels*0.05), :], axis=0)      # ~20-60Hz
        features.bass_band = np.mean(mel_norm[int(n_mels*0.05):int(n_mels*0.15), :], axis=0)  # ~60-250Hz
        features.low_mids = np.mean(mel_norm[int(n_mels*0.15):int(n_mels*0.25), :], axis=0)   # ~250-500Hz
        features.mids = np.mean(mel_norm[int(n_mels*0.25):int(n_mels*0.45), :], axis=0)       # ~500-2kHz
        features.high_mids = np.mean(mel_norm[int(n_mels*0.45):int(n_mels*0.60), :], axis=0)  # ~2-4kHz
        features.highs = np.mean(mel_norm[int(n_mels*0.60):int(n_mels*0.80), :], axis=0)      # ~4-8kHz
        features.brilliance = np.mean(mel_norm[int(n_mels*0.80):, :], axis=0)                 # ~8kHz+

        # If source separation failed, estimate from frequency bands
        if features.drums is None:
            features.drums = (features.sub_bass + features.high_mids) / 2  # Kick + snare
            features.bass = features.bass_band
            features.vocals = (features.mids + features.high_mids) / 2
            features.other = (features.highs + features.brilliance) / 2

        return features

    def _extract_musical_features(self, y: np.ndarray, sr: int, features: AudioFeatures) -> AudioFeatures:
        """Extract tempo, beats, key, and harmonic content."""
        print("[AudioAnalyzer] Extracting musical features...")

        # Tempo and beats
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=self.hop_length)
        features.tempo = float(tempo) if np.isscalar(tempo) else float(tempo[0])

        # Create beat signal (1 at beat frames, 0 elsewhere)
        n_frames = int(len(y) / self.hop_length) + 1
        features.beats = np.zeros(n_frames)
        beat_frames_valid = beat_frames[beat_frames < n_frames]
        features.beats[beat_frames_valid] = 1.0

        # Smooth beats for visualization
        features.beats = np.convolve(features.beats, np.exp(-np.arange(10)/3), mode='same')
        features.beats = features.beats / (features.beats.max() + 1e-8)

        # Downbeats (every 4th beat approximately)
        features.downbeats = np.zeros(n_frames)
        downbeat_frames = beat_frames_valid[::4]
        features.downbeats[downbeat_frames] = 1.0
        features.downbeats = np.convolve(features.downbeats, np.exp(-np.arange(15)/4), mode='same')
        features.downbeats = features.downbeats / (features.downbeats.max() + 1e-8)

        # Chromagram for key/harmony
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=self.hop_length)
        features.chroma = chroma  # 12 x n_frames

        # Estimate key
        chroma_avg = np.mean(chroma, axis=1)
        key_idx = np.argmax(chroma_avg)
        keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        features.key = keys[key_idx]

        # Estimate major/minor
        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

        # Rotate to match detected key
        major_corr = np.corrcoef(np.roll(major_profile, key_idx), chroma_avg)[0, 1]
        minor_corr = np.corrcoef(np.roll(minor_profile, key_idx), chroma_avg)[0, 1]
        features.mode = 'major' if major_corr > minor_corr else 'minor'

        print(f"[AudioAnalyzer] Detected: {features.key} {features.mode}, {features.tempo:.1f} BPM")

        return features

    def _extract_timbral_features(self, y: np.ndarray, sr: int, features: AudioFeatures) -> AudioFeatures:
        """Extract timbre-related features."""
        print("[AudioAnalyzer] Extracting timbral features...")

        # Spectral centroid (brightness)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=self.hop_length)[0]
        features.brightness = (centroid - centroid.min()) / (centroid.max() - centroid.min() + 1e-8)

        # Warmth (ratio of low to high frequencies)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, hop_length=self.hop_length)
        low_energy = np.sum(mel[:16, :], axis=0)
        high_energy = np.sum(mel[32:, :], axis=0) + 1e-8
        warmth = low_energy / (low_energy + high_energy)
        features.warmth = warmth

        # Spectral flux (roughness/change)
        flux = librosa.onset.onset_strength(y=y, sr=sr, hop_length=self.hop_length)
        features.roughness = flux / (flux.max() + 1e-8)

        # Spectral flatness (noisiness/density)
        flatness = librosa.feature.spectral_flatness(y=y, hop_length=self.hop_length)[0]
        features.density = flatness / (flatness.max() + 1e-8)

        return features

    def _extract_dynamics(self, y: np.ndarray, sr: int, features: AudioFeatures) -> AudioFeatures:
        """Extract dynamic features."""
        print("[AudioAnalyzer] Extracting dynamics...")

        # RMS volume
        rms = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]
        features.volume = rms / (rms.max() + 1e-8)

        # Dynamics (volume changes)
        dynamics = np.abs(np.diff(features.volume, prepend=features.volume[0]))
        features.dynamics = dynamics / (dynamics.max() + 1e-8)

        # Onset strength
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=self.hop_length)
        features.onsets = onset_env / (onset_env.max() + 1e-8)

        # Transients (sharp onsets)
        onset_peaks = librosa.util.peak_pick(onset_env, pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.2, wait=10)
        features.transients = np.zeros(len(onset_env))
        onset_peaks_valid = onset_peaks[onset_peaks < len(features.transients)]
        features.transients[onset_peaks_valid] = 1.0
        features.transients = np.convolve(features.transients, np.exp(-np.arange(8)/2), mode='same')

        return features

    def _estimate_mood(self, features: AudioFeatures) -> AudioFeatures:
        """Estimate mood/emotion features."""
        print("[AudioAnalyzer] Estimating mood...")

        # Energy: combination of volume, tempo, brightness
        tempo_factor = min(features.tempo / 120.0, 2.0) / 2.0
        features.energy = (features.volume * 0.4 + features.brightness * 0.3 +
                          features.roughness * 0.3) * (0.5 + tempo_factor * 0.5)

        # Valence (happy/sad): major mode + brightness + tempo
        mode_factor = 0.7 if features.mode == 'major' else 0.3
        features.valence = (features.brightness * 0.4 +
                           np.full_like(features.brightness, mode_factor) * 0.3 +
                           np.full_like(features.brightness, tempo_factor) * 0.3)

        return features

    def _get_min_frames(self, features: AudioFeatures) -> int:
        """Get minimum frame count across all arrays."""
        arrays = [
            features.drums, features.bass, features.vocals, features.other,
            features.sub_bass, features.bass_band, features.low_mids, features.mids,
            features.high_mids, features.highs, features.brilliance,
            features.beats, features.downbeats, features.brightness, features.warmth,
            features.roughness, features.density, features.volume, features.dynamics,
            features.onsets, features.transients, features.energy, features.valence
        ]
        lengths = [len(a) for a in arrays if a is not None and len(a) > 0]
        return min(lengths) if lengths else 0

    def _trim_to_frames(self, features: AudioFeatures) -> AudioFeatures:
        """Trim all arrays to the same frame count."""
        n = features.num_frames

        for attr in ['drums', 'bass', 'vocals', 'other', 'sub_bass', 'bass_band',
                     'low_mids', 'mids', 'high_mids', 'highs', 'brilliance',
                     'beats', 'downbeats', 'brightness', 'warmth', 'roughness',
                     'density', 'volume', 'dynamics', 'onsets', 'transients',
                     'energy', 'valence']:
            arr = getattr(features, attr)
            if arr is not None and len(arr) >= n:
                setattr(features, attr, arr[:n])

        if features.chroma is not None and features.chroma.shape[1] >= n:
            features.chroma = features.chroma[:, :n]

        return features


# ============================================================================
# VISUAL ELEMENTS
# ============================================================================

class VisualElement:
    """Base class for visual elements."""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.center = (width // 2, height // 2)

    def update(self, frame_idx: int, features: AudioFeatures) -> None:
        """Update internal state based on audio features."""
        pass

    def render(self, canvas: np.ndarray) -> np.ndarray:
        """Render element onto canvas (additive blending)."""
        return canvas


class ParticleSystem(VisualElement):
    """Enhanced particle system with multiple behaviors."""

    def __init__(self, width: int, height: int, count: int = 3000):
        super().__init__(width, height)
        self.count = count
        self.positions = None
        self.velocities = None
        self.base_radius = None
        self.colors = None
        self.sizes = None
        self.init_particles()

    def init_particles(self):
        """Initialize particles in a sphere."""
        theta = np.random.uniform(0, 2 * np.pi, self.count)
        phi = np.random.uniform(0, np.pi, self.count)
        radius = np.random.uniform(30, 250, self.count)

        self.positions = np.zeros((self.count, 2))
        self.positions[:, 0] = self.center[0] + radius * np.sin(phi) * np.cos(theta)
        self.positions[:, 1] = self.center[1] + radius * np.sin(phi) * np.sin(theta)

        self.base_radius = radius.copy()
        self.velocities = np.random.uniform(-0.5, 0.5, (self.count, 2))
        self.colors = np.random.uniform(0.5, 1.0, (self.count, 3))
        self.sizes = np.random.uniform(1, 3, self.count)

        # Particle types for different behaviors
        self.types = np.random.choice(3, self.count)  # 0: orbital, 1: radial, 2: chaotic

    def update(self, frame_idx: int, features: AudioFeatures):
        """Update particles based on separated sources."""
        f = frame_idx
        center = np.array(self.center)

        # Get features for this frame
        drums = features.drums[f] if features.drums is not None else 0.5
        bass = features.bass[f] if features.bass is not None else 0.5
        vocals = features.vocals[f] if features.vocals is not None else 0.5
        beats = features.beats[f] if features.beats is not None else 0.0
        volume = features.volume[f] if features.volume is not None else 0.5
        brightness = features.brightness[f] if features.brightness is not None else 0.5
        energy = features.energy[f] if features.energy is not None else 0.5

        # Direction from center
        dirs = self.positions - center
        dists = np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-8
        unit_dirs = dirs / dists

        # Different behaviors for different particle types
        for ptype in range(3):
            mask = self.types == ptype

            if ptype == 0:  # Orbital - respond to bass
                # Rotate around center
                angle = 0.02 + bass * 0.08
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                centered = self.positions[mask] - center
                rotated = np.zeros_like(centered)
                rotated[:, 0] = centered[:, 0] * cos_a - centered[:, 1] * sin_a
                rotated[:, 1] = centered[:, 0] * sin_a + centered[:, 1] * cos_a
                self.positions[mask] = rotated + center

                # Pulsate with bass
                pulsate = np.sin(frame_idx * 0.1) * bass * 40
                target_r = self.base_radius[mask] + pulsate
                curr_r = dists[mask].flatten()
                diff = (target_r - curr_r) * 0.1
                self.positions[mask] += unit_dirs[mask] * diff[:, np.newaxis]

            elif ptype == 1:  # Radial - respond to drums
                # Expand/contract on beats
                expand = beats * 50 + drums * 30
                self.positions[mask] += unit_dirs[mask] * expand * 0.3

                # Contract slowly
                self.positions[mask] -= unit_dirs[mask] * 2

            else:  # Chaotic - respond to vocals/brightness
                # Random movement influenced by vocals
                noise = np.random.uniform(-1, 1, self.positions[mask].shape)
                self.positions[mask] += noise * (2 + vocals * 8)

                # Slight attraction to center
                self.positions[mask] -= unit_dirs[mask] * 0.5

        # Update colors based on audio
        chroma = features.chroma[:, f] if features.chroma is not None else np.zeros(12)
        dominant_note = np.argmax(chroma)
        hue_base = dominant_note / 12.0

        for i in range(self.count):
            hue = (hue_base + self.types[i] * 0.1 + i * 0.0001) % 1.0
            sat = 0.6 + energy * 0.4
            val = 0.5 + volume * 0.5
            r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
            self.colors[i] = [r, g, b]

        # Update sizes
        self.sizes = 1 + volume * 4 + self.types * 0.5

        # Boundary handling
        margin = 50
        self.positions[:, 0] = np.clip(self.positions[:, 0], margin, self.width - margin)
        self.positions[:, 1] = np.clip(self.positions[:, 1], margin, self.height - margin)

    def render(self, canvas: np.ndarray) -> np.ndarray:
        """Render particles with glow."""
        for i in range(self.count):
            x, y = int(self.positions[i, 0]), int(self.positions[i, 1])
            if 0 <= x < self.width and 0 <= y < self.height:
                color = (int(self.colors[i, 2] * 255),
                        int(self.colors[i, 1] * 255),
                        int(self.colors[i, 0] * 255))
                size = max(1, int(self.sizes[i]))
                cv2.circle(canvas, (x, y), size, color, -1, cv2.LINE_AA)
        return canvas


class FractalSystem(VisualElement):
    """Julia set fractal that morphs with audio."""

    def __init__(self, width: int, height: int, resolution: float = 0.5):
        super().__init__(width, height)
        self.res_w = int(width * resolution)
        self.res_h = int(height * resolution)

        # Julia set parameters
        self.c_real = -0.7
        self.c_imag = 0.27015
        self.zoom = 1.0
        self.max_iter = 50

        # Pre-compute coordinate grid
        x = np.linspace(-2, 2, self.res_w)
        y = np.linspace(-2, 2, self.res_h)
        self.X, self.Y = np.meshgrid(x, y)

        self.current_frame = None
        self.hue_offset = 0.0

    def update(self, frame_idx: int, features: AudioFeatures):
        """Update fractal parameters based on audio."""
        f = frame_idx

        bass = features.bass[f] if features.bass is not None else 0.5
        mids = features.mids[f] if features.mids is not None else 0.5
        highs = features.highs[f] if features.highs is not None else 0.5
        brightness = features.brightness[f] if features.brightness is not None else 0.5
        beats = features.beats[f] if features.beats is not None else 0.0
        valence = features.valence[f] if features.valence is not None else 0.5

        # Morph Julia set parameters with audio
        base_angle = frame_idx * 0.02
        self.c_real = -0.7 + np.sin(base_angle) * 0.15 + bass * 0.1
        self.c_imag = 0.27015 + np.cos(base_angle * 0.7) * 0.15 + mids * 0.1

        # Zoom pulses with beats
        self.zoom = 1.0 + beats * 0.5 + np.sin(frame_idx * 0.05) * 0.2

        # Color based on mood
        self.hue_offset = valence * 0.5

        # Render fractal
        self._render_julia(brightness, highs)

    def _render_julia(self, brightness: float, highs: float):
        """Render Julia set fractal."""
        # Scale coordinates by zoom
        zx = self.X / self.zoom
        zy = self.Y / self.zoom

        # Julia set iteration
        c = complex(self.c_real, self.c_imag)
        z = zx + 1j * zy

        output = np.zeros((self.res_h, self.res_w), dtype=np.float32)
        mask = np.ones((self.res_h, self.res_w), dtype=bool)

        for i in range(self.max_iter):
            z[mask] = z[mask] ** 2 + c
            escaped = np.abs(z) > 2
            output[mask & escaped] = i
            mask = mask & ~escaped

        # Normalize
        output = output / self.max_iter

        # Apply coloring
        hue = (output * 0.7 + self.hue_offset) % 1.0
        sat = np.clip(0.7 + brightness * 0.3, 0, 1)
        val = np.clip(output * (0.5 + highs * 0.5), 0, 1)

        # Convert HSV to RGB
        frame = np.zeros((self.res_h, self.res_w, 3), dtype=np.uint8)
        for y in range(self.res_h):
            for x in range(self.res_w):
                if output[y, x] > 0:
                    r, g, b = colorsys.hsv_to_rgb(hue[y, x], sat, val[y, x])
                    frame[y, x] = [int(b * 255), int(g * 255), int(r * 255)]

        # Resize to full resolution
        self.current_frame = cv2.resize(frame, (self.width, self.height),
                                        interpolation=cv2.INTER_LINEAR)

    def render(self, canvas: np.ndarray) -> np.ndarray:
        """Blend fractal onto canvas."""
        if self.current_frame is not None:
            # Additive blending
            canvas = cv2.addWeighted(canvas, 1.0, self.current_frame, 0.3, 0)
        return canvas


class StarburstSystem(VisualElement):
    """Starburst/ray patterns that react to transients."""

    def __init__(self, width: int, height: int, num_rays: int = 24):
        super().__init__(width, height)
        self.num_rays = num_rays
        self.ray_lengths = np.zeros(num_rays)
        self.ray_widths = np.ones(num_rays) * 2
        self.ray_colors = np.zeros((num_rays, 3))
        self.rotation = 0.0
        self.intensity = 0.0

    def update(self, frame_idx: int, features: AudioFeatures):
        """Update starburst based on transients and beats."""
        f = frame_idx

        transients = features.transients[f] if features.transients is not None else 0.0
        beats = features.beats[f] if features.beats is not None else 0.0
        drums = features.drums[f] if features.drums is not None else 0.5
        highs = features.highs[f] if features.highs is not None else 0.5
        brightness = features.brightness[f] if features.brightness is not None else 0.5

        # Intensity spikes on transients
        self.intensity = self.intensity * 0.85 + (transients + beats * 0.5) * 0.5

        # Rotation
        self.rotation += 0.01 + drums * 0.05

        # Ray lengths vary with frequency bands
        chroma = features.chroma[:, f] if features.chroma is not None else np.zeros(12)
        for i in range(self.num_rays):
            base_length = 100 + self.intensity * 300
            freq_mod = chroma[i % 12] * 100
            self.ray_lengths[i] = base_length + freq_mod + np.sin(frame_idx * 0.1 + i) * 30

        # Colors based on position
        for i in range(self.num_rays):
            hue = (i / self.num_rays + brightness * 0.3) % 1.0
            r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.6 + highs * 0.4)
            self.ray_colors[i] = [r, g, b]

        self.ray_widths = 2 + self.intensity * 8

    def render(self, canvas: np.ndarray) -> np.ndarray:
        """Render starburst rays."""
        if self.intensity < 0.05:
            return canvas

        cx, cy = self.center

        for i in range(self.num_rays):
            angle = (i / self.num_rays) * 2 * np.pi + self.rotation
            length = self.ray_lengths[i]

            x2 = int(cx + np.cos(angle) * length)
            y2 = int(cy + np.sin(angle) * length)

            color = (int(self.ray_colors[i, 2] * 255 * self.intensity),
                    int(self.ray_colors[i, 1] * 255 * self.intensity),
                    int(self.ray_colors[i, 0] * 255 * self.intensity))

            thickness = max(1, int(self.ray_widths))
            cv2.line(canvas, (cx, cy), (x2, y2), color, thickness, cv2.LINE_AA)

        return canvas


class FlowField(VisualElement):
    """Flow field visualization responding to vocals and mids."""

    def __init__(self, width: int, height: int, grid_size: int = 30):
        super().__init__(width, height)
        self.grid_size = grid_size
        self.cols = width // grid_size
        self.rows = height // grid_size

        # Flow vectors
        self.angles = np.zeros((self.rows, self.cols))
        self.magnitudes = np.ones((self.rows, self.cols)) * 0.5

        # Trail particles
        self.trail_count = 500
        self.trail_pos = np.random.rand(self.trail_count, 2) * [width, height]
        self.trail_colors = np.random.rand(self.trail_count, 3)
        self.trail_life = np.random.rand(self.trail_count)

    def update(self, frame_idx: int, features: AudioFeatures):
        """Update flow field based on audio."""
        f = frame_idx

        vocals = features.vocals[f] if features.vocals is not None else 0.5
        mids = features.mids[f] if features.mids is not None else 0.5
        warmth = features.warmth[f] if features.warmth is not None else 0.5
        energy = features.energy[f] if features.energy is not None else 0.5

        # Update flow angles with Perlin-like noise influenced by audio
        for r in range(self.rows):
            for c in range(self.cols):
                noise = np.sin(c * 0.1 + frame_idx * 0.02) * np.cos(r * 0.1 + frame_idx * 0.015)
                self.angles[r, c] = noise * np.pi + vocals * np.pi * 0.5
                self.magnitudes[r, c] = 0.5 + mids * 2 + noise * 0.3

        # Update trail particles
        for i in range(self.trail_count):
            px, py = self.trail_pos[i]
            col = int(px / self.grid_size) % self.cols
            row = int(py / self.grid_size) % self.rows

            angle = self.angles[row, col]
            mag = self.magnitudes[row, col]

            self.trail_pos[i, 0] += np.cos(angle) * mag * 3
            self.trail_pos[i, 1] += np.sin(angle) * mag * 3

            # Wrap around
            self.trail_pos[i, 0] %= self.width
            self.trail_pos[i, 1] %= self.height

            # Update life and color
            self.trail_life[i] -= 0.01
            if self.trail_life[i] <= 0:
                self.trail_life[i] = 1.0
                self.trail_pos[i] = np.random.rand(2) * [self.width, self.height]

            # Color based on energy
            hue = (warmth + i * 0.001) % 1.0
            r, g, b = colorsys.hsv_to_rgb(hue, 0.7, energy)
            self.trail_colors[i] = [r, g, b]

    def render(self, canvas: np.ndarray) -> np.ndarray:
        """Render flow field trails."""
        for i in range(self.trail_count):
            x, y = int(self.trail_pos[i, 0]), int(self.trail_pos[i, 1])
            if 0 <= x < self.width and 0 <= y < self.height:
                alpha = self.trail_life[i]
                color = (int(self.trail_colors[i, 2] * 255 * alpha),
                        int(self.trail_colors[i, 1] * 255 * alpha),
                        int(self.trail_colors[i, 0] * 255 * alpha))
                size = int(2 + alpha * 3)
                cv2.circle(canvas, (x, y), size, color, -1, cv2.LINE_AA)
        return canvas


class WaveformRing(VisualElement):
    """Circular waveform visualization."""

    def __init__(self, width: int, height: int, base_radius: int = 150):
        super().__init__(width, height)
        self.base_radius = base_radius
        self.points = 180  # Points around the circle
        self.amplitudes = np.zeros(self.points)
        self.hue = 0.0

    def update(self, frame_idx: int, features: AudioFeatures):
        """Update waveform based on frequency data."""
        f = frame_idx

        # Get frequency bands
        sub = features.sub_bass[f] if features.sub_bass is not None else 0.5
        bass = features.bass_band[f] if features.bass_band is not None else 0.5
        lmids = features.low_mids[f] if features.low_mids is not None else 0.5
        mids = features.mids[f] if features.mids is not None else 0.5
        hmids = features.high_mids[f] if features.high_mids is not None else 0.5
        highs = features.highs[f] if features.highs is not None else 0.5
        brill = features.brilliance[f] if features.brilliance is not None else 0.5

        # Map frequency bands to circle positions
        bands = [sub, bass, lmids, mids, hmids, highs, brill]
        points_per_band = self.points // len(bands)

        for i, band_val in enumerate(bands):
            start = i * points_per_band
            end = start + points_per_band

            for j in range(start, min(end, self.points)):
                wave = np.sin(frame_idx * 0.1 + j * 0.2) * 0.3
                self.amplitudes[j] = band_val * 100 + wave * 20

        # Smooth the amplitudes
        self.amplitudes = np.convolve(self.amplitudes, np.ones(5)/5, mode='same')

        # Update color
        brightness = features.brightness[f] if features.brightness is not None else 0.5
        self.hue = (self.hue + 0.002 + brightness * 0.01) % 1.0

    def render(self, canvas: np.ndarray) -> np.ndarray:
        """Render circular waveform."""
        cx, cy = self.center

        points = []
        for i in range(self.points):
            angle = (i / self.points) * 2 * np.pi
            radius = self.base_radius + self.amplitudes[i]
            x = int(cx + np.cos(angle) * radius)
            y = int(cy + np.sin(angle) * radius)
            points.append([x, y])

        points = np.array(points, dtype=np.int32)

        # Draw filled polygon with color
        r, g, b = colorsys.hsv_to_rgb(self.hue, 0.8, 0.7)
        color = (int(b * 255), int(g * 255), int(r * 255))

        # Draw outline
        cv2.polylines(canvas, [points], True, color, 2, cv2.LINE_AA)

        # Draw inner ring
        inner_points = []
        for i in range(self.points):
            angle = (i / self.points) * 2 * np.pi
            radius = self.base_radius - self.amplitudes[i] * 0.3
            x = int(cx + np.cos(angle) * radius)
            y = int(cy + np.sin(angle) * radius)
            inner_points.append([x, y])

        inner_points = np.array(inner_points, dtype=np.int32)
        inner_color = (int(b * 200), int(g * 200), int(r * 200))
        cv2.polylines(canvas, [inner_points], True, inner_color, 1, cv2.LINE_AA)

        return canvas


class GeometricPattern(VisualElement):
    """Sacred geometry / mathematical patterns."""

    def __init__(self, width: int, height: int):
        super().__init__(width, height)
        self.rotation = 0.0
        self.scale = 1.0
        self.sides = 6
        self.layers = 5
        self.hue = 0.0

    def update(self, frame_idx: int, features: AudioFeatures):
        """Update geometric pattern."""
        f = frame_idx

        beats = features.beats[f] if features.beats is not None else 0.0
        downbeats = features.downbeats[f] if features.downbeats is not None else 0.0
        energy = features.energy[f] if features.energy is not None else 0.5
        valence = features.valence[f] if features.valence is not None else 0.5

        # Rotation speeds up with energy
        self.rotation += 0.005 + energy * 0.02

        # Scale pulses with downbeats
        self.scale = 1.0 + downbeats * 0.3 + beats * 0.1

        # Sides can change (5-8)
        self.sides = 5 + int(valence * 3)

        # Color
        self.hue = (self.hue + 0.001) % 1.0

    def render(self, canvas: np.ndarray) -> np.ndarray:
        """Render sacred geometry pattern."""
        cx, cy = self.center

        for layer in range(self.layers):
            radius = (50 + layer * 40) * self.scale
            offset_rot = self.rotation + layer * 0.1

            # Draw polygon
            points = []
            for i in range(self.sides):
                angle = offset_rot + (i / self.sides) * 2 * np.pi
                x = int(cx + np.cos(angle) * radius)
                y = int(cy + np.sin(angle) * radius)
                points.append([x, y])

            points = np.array(points, dtype=np.int32)

            # Color varies by layer
            hue = (self.hue + layer * 0.1) % 1.0
            alpha = 1.0 - layer * 0.15
            r, g, b = colorsys.hsv_to_rgb(hue, 0.7, 0.6 * alpha)
            color = (int(b * 255), int(g * 255), int(r * 255))

            cv2.polylines(canvas, [points], True, color, 2, cv2.LINE_AA)

            # Connect to next layer
            if layer < self.layers - 1:
                next_radius = (50 + (layer + 1) * 40) * self.scale
                for i in range(self.sides):
                    angle1 = offset_rot + (i / self.sides) * 2 * np.pi
                    angle2 = offset_rot + ((i + 0.5) / self.sides) * 2 * np.pi

                    x1 = int(cx + np.cos(angle1) * radius)
                    y1 = int(cy + np.sin(angle1) * radius)
                    x2 = int(cx + np.cos(angle2) * next_radius)
                    y2 = int(cy + np.sin(angle2) * next_radius)

                    cv2.line(canvas, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

        return canvas


# ============================================================================
# VIDEO GENERATOR
# ============================================================================

class SynestheticEngine:
    """Main engine for generating synesthetic visualizations."""

    def __init__(
        self,
        width: int = 1920,
        height: int = 1080,
        fps: int = 30,
        use_source_separation: bool = True
    ):
        self.width = width
        self.height = height
        self.fps = fps
        self.use_source_separation = use_source_separation

        self.analyzer = AudioAnalyzer(fps=fps, use_source_separation=use_source_separation)
        self.elements: List[VisualElement] = []

    def setup_elements(self):
        """Initialize all visual elements."""
        self.elements = [
            FractalSystem(self.width, self.height, resolution=0.3),
            FlowField(self.width, self.height, grid_size=40),
            WaveformRing(self.width, self.height, base_radius=180),
            GeometricPattern(self.width, self.height),
            ParticleSystem(self.width, self.height, count=2500),
            StarburstSystem(self.width, self.height, num_rays=32),
        ]
        print(f"[SynestheticEngine] Initialized {len(self.elements)} visual elements")

    def render_frame(self, frame_idx: int, features: AudioFeatures) -> np.ndarray:
        """Render a single frame."""
        # Create canvas
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Update and render each element
        for element in self.elements:
            element.update(frame_idx, features)
            canvas = element.render(canvas)

        # Post-processing
        canvas = self._apply_post_processing(canvas, frame_idx, features)

        return canvas

    def _apply_post_processing(self, canvas: np.ndarray, frame_idx: int,
                               features: AudioFeatures) -> np.ndarray:
        """Apply post-processing effects."""
        f = frame_idx

        volume = features.volume[f] if features.volume is not None else 0.5
        energy = features.energy[f] if features.energy is not None else 0.5

        # Bloom effect
        if energy > 0.3:
            blur_size = int(11 + energy * 20)
            if blur_size % 2 == 0:
                blur_size += 1
            bloom = cv2.GaussianBlur(canvas, (blur_size, blur_size), 0)
            canvas = cv2.addWeighted(canvas, 1.0, bloom, 0.4, 0)

        # Vignette
        rows, cols = canvas.shape[:2]
        X = np.arange(0, cols)
        Y = np.arange(0, rows)
        X, Y = np.meshgrid(X, Y)
        center_x, center_y = cols / 2, rows / 2
        vignette = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
        vignette = 1 - vignette / vignette.max() * 0.4
        canvas = (canvas * vignette[:, :, np.newaxis]).astype(np.uint8)

        # Subtle chromatic aberration on high energy
        if energy > 0.6:
            shift = int(energy * 3)
            b, g, r = cv2.split(canvas)
            rows, cols = canvas.shape[:2]
            M_left = np.float32([[1, 0, -shift], [0, 1, 0]])
            M_right = np.float32([[1, 0, shift], [0, 1, 0]])
            r = cv2.warpAffine(r, M_right, (cols, rows))
            b = cv2.warpAffine(b, M_left, (cols, rows))
            canvas = cv2.merge([b, g, r])

        return canvas

    def generate(
        self,
        audio_path: str,
        output_path: str,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> str:
        """Generate visualization video."""
        print(f"\n{'='*60}")
        print("SYNESTHETIC ENGINE - Advanced Audio Visualization")
        print(f"{'='*60}\n")

        # Analyze audio
        features = self.analyzer.analyze(audio_path)

        # Setup visual elements
        self.setup_elements()

        # Create temporary video file
        temp_video = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name

        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(temp_video, fourcc, self.fps, (self.width, self.height))

        print(f"\n[SynestheticEngine] Generating {features.num_frames} frames...")
        print(f"[SynestheticEngine] Key: {features.key} {features.mode}")
        print(f"[SynestheticEngine] Tempo: {features.tempo:.1f} BPM\n")

        for frame_idx in range(features.num_frames):
            frame = self.render_frame(frame_idx, features)
            writer.write(frame)

            if frame_idx % 30 == 0:
                progress = (frame_idx + 1) / features.num_frames * 100
                print(f"Progress: {progress:.1f}% ({frame_idx + 1}/{features.num_frames})")
                if progress_callback:
                    progress_callback(progress)

        writer.release()
        print("\n[SynestheticEngine] Video frames complete, adding audio...")

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
            print(f"\n[SynestheticEngine] Output: {final_output}")
        except Exception as e:
            print(f"[SynestheticEngine] FFmpeg error: {e}")
            os.rename(temp_video, final_output)
        finally:
            if os.path.exists(temp_video):
                os.remove(temp_video)

        return final_output


def main():
    """Command-line interface."""
    import argparse

    parser = argparse.ArgumentParser(description='Synesthetic Audio Visualization Engine')
    parser.add_argument('audio', help='Path to input audio file')
    parser.add_argument('-o', '--output', default='synesthetic_output.mp4', help='Output video path')
    parser.add_argument('--width', type=int, default=1920, help='Video width')
    parser.add_argument('--height', type=int, default=1080, help='Video height')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second')
    parser.add_argument('--no-separation', action='store_true', help='Disable source separation')

    args = parser.parse_args()

    engine = SynestheticEngine(
        width=args.width,
        height=args.height,
        fps=args.fps,
        use_source_separation=not args.no_separation
    )

    engine.generate(args.audio, args.output)


if __name__ == '__main__':
    main()
