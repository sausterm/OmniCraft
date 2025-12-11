/**
 * ParticleVisualizer
 * Real-time audio-reactive particle system using Three.js
 */

import * as THREE from 'three';

export class ParticleVisualizer {
  constructor(container) {
    this.container = container;
    this.scene = null;
    this.camera = null;
    this.renderer = null;
    this.particles = null;
    this.particleCount = 5000;
    this.isActive = false;
    
    this.init();
  }

  init() {
    // Create scene
    this.scene = new THREE.Scene();
    this.scene.fog = new THREE.FogExp2(0x000000, 0.001);

    // Create camera
    this.camera = new THREE.PerspectiveCamera(
      75,
      window.innerWidth / window.innerHeight,
      0.1,
      1000
    );
    this.camera.position.z = 100;

    // Create renderer
    this.renderer = new THREE.WebGLRenderer({ 
      antialias: true,
      alpha: true 
    });
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.container.appendChild(this.renderer.domElement);

    // Create particle system
    this.createParticles();

    // Add lights
    const ambientLight = new THREE.AmbientLight(0x404040);
    this.scene.add(ambientLight);

    const pointLight = new THREE.PointLight(0xffffff, 1, 100);
    pointLight.position.set(0, 0, 50);
    this.scene.add(pointLight);

    // Handle window resize
    window.addEventListener('resize', () => this.onWindowResize());

    this.isActive = true;
    console.log('✓ Particle visualizer initialized');
  }

  createParticles() {
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(this.particleCount * 3);
    const colors = new Float32Array(this.particleCount * 3);
    const sizes = new Float32Array(this.particleCount);

    // Initialize particle positions in a sphere
    for (let i = 0; i < this.particleCount; i++) {
      const i3 = i * 3;
      
      // Random position in sphere
      const radius = Math.random() * 50;
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.random() * Math.PI;
      
      positions[i3] = radius * Math.sin(phi) * Math.cos(theta);
      positions[i3 + 1] = radius * Math.sin(phi) * Math.sin(theta);
      positions[i3 + 2] = radius * Math.cos(phi);

      // Random colors
      colors[i3] = Math.random();
      colors[i3 + 1] = Math.random();
      colors[i3 + 2] = Math.random();

      // Random sizes
      sizes[i] = Math.random() * 2;
    }

    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1));

    // Create material
    const material = new THREE.PointsMaterial({
      size: 2,
      vertexColors: true,
      blending: THREE.AdditiveBlending,
      transparent: true,
      opacity: 0.8,
      sizeAttenuation: true
    });

    // Create particle system
    this.particles = new THREE.Points(geometry, material);
    this.scene.add(this.particles);
  }

  /**
   * Update visualization based on audio features
   */
  update(audioFeatures) {
    if (!this.particles || !audioFeatures) return;

    const { volume, bass, mids, treble, frequencyData } = audioFeatures;
    
    // Rotate based on volume
    this.particles.rotation.y += 0.001 + (volume / 255) * 0.01;
    this.particles.rotation.x += 0.0005;

    // Update particle positions and colors based on audio
    const positions = this.particles.geometry.attributes.position.array;
    const colors = this.particles.geometry.attributes.color.array;
    const sizes = this.particles.geometry.attributes.size.array;

    const bassIntensity = bass / 255;
    const midsIntensity = mids / 255;
    const trebleIntensity = treble / 255;

    for (let i = 0; i < this.particleCount; i++) {
      const i3 = i * 3;
      
      // Pulsate based on bass
      const pulsate = Math.sin(Date.now() * 0.001 + i) * bassIntensity * 5;
      const currentRadius = Math.sqrt(
        positions[i3] ** 2 + 
        positions[i3 + 1] ** 2 + 
        positions[i3 + 2] ** 2
      );
      
      if (currentRadius > 0) {
        const scale = (currentRadius + pulsate) / currentRadius;
        positions[i3] *= scale;
        positions[i3 + 1] *= scale;
        positions[i3 + 2] *= scale;
      }

      // Update colors based on frequency ranges
      colors[i3] = bassIntensity;       // Red from bass
      colors[i3 + 1] = midsIntensity;   // Green from mids
      colors[i3 + 2] = trebleIntensity; // Blue from treble

      // Update sizes based on volume
      sizes[i] = 1 + (volume / 255) * 2;
    }

    // Mark attributes as needing update
    this.particles.geometry.attributes.position.needsUpdate = true;
    this.particles.geometry.attributes.color.needsUpdate = true;
    this.particles.geometry.attributes.size.needsUpdate = true;

    // Update camera position for dynamic view
    this.camera.position.z = 100 + Math.sin(Date.now() * 0.0005) * 20;
    this.camera.lookAt(this.scene.position);
  }

  /**
   * Render the scene
   */
  render() {
    if (!this.isActive) return;
    this.renderer.render(this.scene, this.camera);
  }

  /**
   * Handle window resize
   */
  onWindowResize() {
    this.camera.aspect = window.innerWidth / window.innerHeight;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(window.innerWidth, window.innerHeight);
  }

  /**
   * Clean up resources
   */
  dispose() {
    this.isActive = false;
    
    if (this.particles) {
      this.particles.geometry.dispose();
      this.particles.material.dispose();
      this.scene.remove(this.particles);
    }
    
    if (this.renderer) {
      this.renderer.dispose();
      this.container.removeChild(this.renderer.domElement);
    }
    
    window.removeEventListener('resize', () => this.onWindowResize());
    console.log('✓ Particle visualizer disposed');
  }
}
