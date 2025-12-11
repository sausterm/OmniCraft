/**
 * Main Application
 * Synesthetic Audio Visualization System - Phase 1 POC
 */

import { AudioAnalyzer } from './audio/AudioAnalyzer.js';
import { ParticleVisualizer } from './visualizers/ParticleVisualizer.js';

class App {
  constructor() {
    this.audioAnalyzer = null;
    this.visualizer = null;
    this.isRunning = false;
    this.animationFrameId = null;
    
    this.initUI();
  }

  initUI() {
    const startBtn = document.getElementById('start-btn');
    const audioStatus = document.getElementById('audio-status');
    const vizStatus = document.getElementById('viz-status');
    const audioIndicator = document.getElementById('audio-indicator');
    const vizIndicator = document.getElementById('viz-indicator');

    this.elements = {
      startBtn,
      audioStatus,
      vizStatus,
      audioIndicator,
      vizIndicator
    };

    startBtn.addEventListener('click', () => this.start());
  }

  async start() {
    try {
      const { startBtn, audioStatus, vizStatus, audioIndicator, vizIndicator } = this.elements;
      
      startBtn.disabled = true;
      startBtn.textContent = 'Initializing...';

      // Initialize audio analyzer
      audioStatus.textContent = 'Initializing microphone...';
      this.audioAnalyzer = new AudioAnalyzer();
      await this.audioAnalyzer.initMicrophone();
      
      audioStatus.textContent = 'Active';
      audioIndicator.classList.add('active');
      console.log('✓ Audio analyzer ready');

      // Initialize visualizer
      vizStatus.textContent = 'Initializing renderer...';
      const container = document.getElementById('canvas-container');
      this.visualizer = new ParticleVisualizer(container);
      
      vizStatus.textContent = 'Active';
      vizIndicator.classList.add('active');
      console.log('✓ Visualizer ready');

      // Start animation loop
      this.isRunning = true;
      this.animate();

      startBtn.textContent = 'Running ✓';
      console.log('✓ Visualization started');

    } catch (error) {
      console.error('Failed to start:', error);
      alert('Failed to start visualization. Please ensure microphone access is granted.');
      
      this.elements.startBtn.disabled = false;
      this.elements.startBtn.textContent = 'Start Visualization';
      this.elements.audioStatus.textContent = 'Error';
      this.elements.vizStatus.textContent = 'Error';
    }
  }

  animate() {
    if (!this.isRunning) return;

    // Get audio features
    const audioFeatures = this.audioAnalyzer.getFeatures();

    // Update visualizer with audio data
    this.visualizer.update(audioFeatures);

    // Render the scene
    this.visualizer.render();

    // Continue animation loop
    this.animationFrameId = requestAnimationFrame(() => this.animate());
  }

  stop() {
    this.isRunning = false;
    
    if (this.animationFrameId) {
      cancelAnimationFrame(this.animationFrameId);
    }

    if (this.audioAnalyzer) {
      this.audioAnalyzer.dispose();
    }

    if (this.visualizer) {
      this.visualizer.dispose();
    }

    console.log('✓ Visualization stopped');
  }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
  console.log('🎵 Synesthetic Audio Visualization System');
  console.log('Phase 1: Real-time Audio Visualization POC');
  console.log('==========================================');
  
  const app = new App();
  
  // Clean up on page unload
  window.addEventListener('beforeunload', () => {
    app.stop();
  });
});
