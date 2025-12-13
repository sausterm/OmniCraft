"""
Video Generator for Synesthetic Visualization
Generates audio-reactive visualization videos synced to input audio files.
"""

import numpy as np
import librosa
import cv2
import os
import subprocess
from typing import Optional, Tuple
import tempfile


class SynestheticVideoGenerator:
    """
    Generates particle-based visualization videos synced to audio.
    """

    def __init__(
        self,
        width: int = 1920,
        height: int = 1080,
        fps: int = 30,
        particle_count: int = 2000
    ):
        self.width = width
        self.height = height
        self.fps = fps
        self.particle_count = particle_count
        self.sample_rate = 22050
        self.hop_length = 512  # ~23ms per frame at 22050 Hz

    def analyze_audio(self, audio_path: str) -> dict:
        """
        Analyze audio file and extract frame-by-frame features.
        """
        print(f"Loading audio: {audio_path}")
        y, sr = librosa.load(audio_path, sr=self.sample_rate)
        duration = len(y) / sr

        print(f"Duration: {duration:.2f}s, Sample rate: {sr}")

        # Calculate hop length to match video fps
        # We want one feature frame per video frame
        samples_per_video_frame = sr / self.fps
        hop_length = int(samples_per_video_frame)

        print("Extracting audio features...")

        # Get mel spectrogram for frequency bands
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=128, hop_length=hop_length
        )
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Normalize mel spectrogram to 0-1
        mel_normalized = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)

        # Split into frequency bands (bass, mids, treble)
        n_mels = mel_normalized.shape[0]
        bass = np.mean(mel_normalized[:n_mels//3, :], axis=0)
        mids = np.mean(mel_normalized[n_mels//3:2*n_mels//3, :], axis=0)
        treble = np.mean(mel_normalized[2*n_mels//3:, :], axis=0)

        # RMS energy (volume)
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        rms_normalized = rms / (rms.max() + 1e-8)

        # Onset strength for beat detection
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
        onset_normalized = onset_env / (onset_env.max() + 1e-8)

        # Spectral centroid (brightness)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
        centroid_normalized = (centroid - centroid.min()) / (centroid.max() - centroid.min() + 1e-8)

        # Ensure all arrays have the same length
        min_frames = min(len(bass), len(mids), len(treble), len(rms_normalized),
                        len(onset_normalized), len(centroid_normalized))

        features = {
            'duration': duration,
            'num_frames': min_frames,
            'bass': bass[:min_frames],
            'mids': mids[:min_frames],
            'treble': treble[:min_frames],
            'volume': rms_normalized[:min_frames],
            'onset': onset_normalized[:min_frames],
            'brightness': centroid_normalized[:min_frames],
        }

        print(f"Extracted {min_frames} feature frames")
        return features

    def init_particles(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Initialize particle positions, velocities, and base colors.
        """
        # Random positions in a sphere
        theta = np.random.uniform(0, 2 * np.pi, self.particle_count)
        phi = np.random.uniform(0, np.pi, self.particle_count)
        radius = np.random.uniform(50, 300, self.particle_count)

        positions = np.zeros((self.particle_count, 2))
        positions[:, 0] = self.width / 2 + radius * np.sin(phi) * np.cos(theta)
        positions[:, 1] = self.height / 2 + radius * np.sin(phi) * np.sin(theta)

        # Store original radius for pulsation
        base_radius = radius.copy()

        # Random velocities for organic movement
        velocities = np.random.uniform(-1, 1, (self.particle_count, 2))

        return positions, velocities, base_radius

    def update_particles(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        base_radius: np.ndarray,
        frame_idx: int,
        bass: float,
        mids: float,
        treble: float,
        volume: float,
        onset: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update particle positions based on audio features.
        """
        center = np.array([self.width / 2, self.height / 2])

        # Calculate direction from center for each particle
        directions = positions - center
        distances = np.linalg.norm(directions, axis=1, keepdims=True) + 1e-8
        unit_directions = directions / distances

        # Pulsation based on bass
        pulsation = np.sin(frame_idx * 0.1 + np.arange(self.particle_count) * 0.01) * bass * 30
        target_radius = base_radius * (1 + volume * 0.5) + pulsation

        # Move particles toward target radius
        current_radius = distances.flatten()
        radius_diff = target_radius - current_radius
        positions += unit_directions * radius_diff[:, np.newaxis] * 0.1

        # Add rotation based on mids
        rotation_speed = 0.02 + mids * 0.05
        angle = rotation_speed
        cos_a, sin_a = np.cos(angle), np.sin(angle)

        centered = positions - center
        rotated = np.zeros_like(centered)
        rotated[:, 0] = centered[:, 0] * cos_a - centered[:, 1] * sin_a
        rotated[:, 1] = centered[:, 0] * sin_a + centered[:, 1] * cos_a
        positions = rotated + center

        # Add turbulence on beats
        if onset > 0.5:
            positions += np.random.uniform(-5, 5, positions.shape) * onset

        # Keep particles in bounds with soft boundaries
        margin = 50
        positions[:, 0] = np.clip(positions[:, 0], margin, self.width - margin)
        positions[:, 1] = np.clip(positions[:, 1], margin, self.height - margin)

        return positions, velocities

    def render_frame(
        self,
        positions: np.ndarray,
        bass: float,
        mids: float,
        treble: float,
        volume: float,
        brightness: float
    ) -> np.ndarray:
        """
        Render a single frame with particles.
        """
        # Create black background
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Calculate particle colors based on audio
        # Bass -> Red, Mids -> Green, Treble -> Blue
        base_color = np.array([
            int(treble * 255),   # Blue
            int(mids * 255),     # Green
            int(bass * 255)      # Red
        ])

        # Particle size based on volume
        base_size = int(2 + volume * 6)

        # Draw particles with glow effect
        for i in range(self.particle_count):
            x, y = int(positions[i, 0]), int(positions[i, 1])

            if 0 <= x < self.width and 0 <= y < self.height:
                # Vary color slightly per particle
                color_variation = np.sin(i * 0.1) * 0.2 + 0.8
                color = (base_color * color_variation).astype(np.uint8)
                color = tuple(map(int, color))

                # Size variation
                size = max(1, base_size + int(np.sin(i * 0.05) * 2))

                # Draw with additive-like effect (multiple small circles)
                cv2.circle(frame, (x, y), size, color, -1, cv2.LINE_AA)

        # Add bloom/glow effect
        if volume > 0.3:
            blur_amount = int(5 + volume * 10)
            if blur_amount % 2 == 0:
                blur_amount += 1
            blurred = cv2.GaussianBlur(frame, (blur_amount, blur_amount), 0)
            frame = cv2.addWeighted(frame, 1.0, blurred, 0.5, 0)

        # Add vignette
        rows, cols = frame.shape[:2]
        X = np.arange(0, cols)
        Y = np.arange(0, rows)
        X, Y = np.meshgrid(X, Y)
        center_x, center_y = cols / 2, rows / 2
        vignette = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
        vignette = 1 - vignette / vignette.max() * 0.5
        frame = (frame * vignette[:, :, np.newaxis]).astype(np.uint8)

        return frame

    def generate_video(
        self,
        audio_path: str,
        output_path: str,
        progress_callback: Optional[callable] = None
    ) -> str:
        """
        Generate visualization video synced to audio.

        Args:
            audio_path: Path to input audio file
            output_path: Path for output video file
            progress_callback: Optional callback for progress updates

        Returns:
            Path to final video with audio
        """
        # Analyze audio
        features = self.analyze_audio(audio_path)
        num_frames = features['num_frames']

        # Initialize particles
        positions, velocities, base_radius = self.init_particles()

        # Create temporary video file (without audio)
        temp_video = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name

        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(temp_video, fourcc, self.fps, (self.width, self.height))

        print(f"Generating {num_frames} frames...")

        for frame_idx in range(num_frames):
            # Get features for this frame
            bass = features['bass'][frame_idx]
            mids = features['mids'][frame_idx]
            treble = features['treble'][frame_idx]
            volume = features['volume'][frame_idx]
            onset = features['onset'][frame_idx]
            brightness = features['brightness'][frame_idx]

            # Update particles
            positions, velocities = self.update_particles(
                positions, velocities, base_radius,
                frame_idx, bass, mids, treble, volume, onset
            )

            # Render frame
            frame = self.render_frame(positions, bass, mids, treble, volume, brightness)

            # Write frame
            writer.write(frame)

            # Progress update
            if frame_idx % 30 == 0:
                progress = (frame_idx + 1) / num_frames * 100
                print(f"Progress: {progress:.1f}% ({frame_idx + 1}/{num_frames} frames)")
                if progress_callback:
                    progress_callback(progress)

        writer.release()
        print("Video frames generated, adding audio...")

        # Combine video with audio using ffmpeg
        final_output = output_path
        if not final_output.endswith('.mp4'):
            final_output += '.mp4'

        cmd = [
            'ffmpeg', '-y',
            '-i', temp_video,
            '-i', audio_path,
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-b:a', '192k',
            '-shortest',
            '-pix_fmt', 'yuv420p',
            final_output
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"Video saved to: {final_output}")
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg error: {e.stderr.decode()}")
            # Fall back to video without audio
            os.rename(temp_video, final_output)
            print(f"Saved video without audio to: {final_output}")
        except FileNotFoundError:
            print("FFmpeg not found. Saving video without audio.")
            os.rename(temp_video, final_output)
        finally:
            # Clean up temp file if it still exists
            if os.path.exists(temp_video):
                os.remove(temp_video)

        return final_output


def main():
    """Generate video from command line."""
    import argparse

    parser = argparse.ArgumentParser(description='Generate synesthetic visualization video')
    parser.add_argument('audio', help='Path to input audio file')
    parser.add_argument('-o', '--output', default='output.mp4', help='Output video path')
    parser.add_argument('--width', type=int, default=1920, help='Video width')
    parser.add_argument('--height', type=int, default=1080, help='Video height')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second')
    parser.add_argument('--particles', type=int, default=2000, help='Number of particles')

    args = parser.parse_args()

    generator = SynestheticVideoGenerator(
        width=args.width,
        height=args.height,
        fps=args.fps,
        particle_count=args.particles
    )

    generator.generate_video(args.audio, args.output)


if __name__ == '__main__':
    main()
