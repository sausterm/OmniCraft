/**
 * AudioAnalyzer
 * Handles audio input and real-time analysis using Web Audio API
 */

export class AudioAnalyzer {
  constructor() {
    this.audioContext = null;
    this.analyser = null;
    this.dataArray = null;
    this.bufferLength = null;
    this.source = null;
    this.isActive = false;
  }

  /**
   * Initialize audio input from microphone
   */
  async initMicrophone() {
    try {
      // Request microphone access
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false
        } 
      });
      
      // Create audio context
      this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
      
      // Create analyser node
      this.analyser = this.audioContext.createAnalyser();
      this.analyser.fftSize = 2048; // Higher = more frequency detail
      this.analyser.smoothingTimeConstant = 0.8; // 0-1, higher = smoother
      
      // Set up data arrays
      this.bufferLength = this.analyser.frequencyBinCount;
      this.dataArray = new Uint8Array(this.bufferLength);
      
      // Connect microphone to analyser
      this.source = this.audioContext.createMediaStreamSource(stream);
      this.source.connect(this.analyser);
      
      this.isActive = true;
      console.log('✓ Audio analyzer initialized');
      
      return true;
    } catch (error) {
      console.error('Audio initialization failed:', error);
      throw error;
    }
  }

  /**
   * Initialize audio from an audio file or element
   */
  initAudioElement(audioElement) {
    this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
    this.analyser = this.audioContext.createAnalyser();
    this.analyser.fftSize = 2048;
    this.analyser.smoothingTimeConstant = 0.8;
    
    this.bufferLength = this.analyser.frequencyBinCount;
    this.dataArray = new Uint8Array(this.bufferLength);
    
    this.source = this.audioContext.createMediaElementSource(audioElement);
    this.source.connect(this.analyser);
    this.analyser.connect(this.audioContext.destination);
    
    this.isActive = true;
    console.log('✓ Audio element connected');
  }

  /**
   * Get raw frequency data (0-255 for each frequency bin)
   */
  getFrequencyData() {
    if (!this.analyser) return null;
    this.analyser.getByteFrequencyData(this.dataArray);
    return this.dataArray;
  }

  /**
   * Get time domain data (waveform)
   */
  getTimeDomainData() {
    if (!this.analyser) return null;
    this.analyser.getByteTimeDomainData(this.dataArray);
    return this.dataArray;
  }

  /**
   * Get overall volume/energy (0-255)
   */
  getVolume() {
    const data = this.getFrequencyData();
    if (!data) return 0;
    
    const sum = data.reduce((acc, val) => acc + val, 0);
    return sum / data.length;
  }

  /**
   * Get bass energy (low frequencies)
   */
  getBass() {
    const data = this.getFrequencyData();
    if (!data) return 0;
    
    const bassRange = Math.floor(this.bufferLength * 0.1); // First 10%
    const sum = data.slice(0, bassRange).reduce((acc, val) => acc + val, 0);
    return sum / bassRange;
  }

  /**
   * Get mid-range energy
   */
  getMids() {
    const data = this.getFrequencyData();
    if (!data) return 0;
    
    const start = Math.floor(this.bufferLength * 0.1);
    const end = Math.floor(this.bufferLength * 0.5);
    const midsData = data.slice(start, end);
    const sum = midsData.reduce((acc, val) => acc + val, 0);
    return sum / midsData.length;
  }

  /**
   * Get treble energy (high frequencies)
   */
  getTreble() {
    const data = this.getFrequencyData();
    if (!data) return 0;
    
    const start = Math.floor(this.bufferLength * 0.5);
    const trebleData = data.slice(start);
    const sum = trebleData.reduce((acc, val) => acc + val, 0);
    return sum / trebleData.length;
  }

  /**
   * Get comprehensive audio features
   */
  getFeatures() {
    return {
      volume: this.getVolume(),
      bass: this.getBass(),
      mids: this.getMids(),
      treble: this.getTreble(),
      frequencyData: this.getFrequencyData(),
      waveform: this.getTimeDomainData()
    };
  }

  /**
   * Clean up resources
   */
  dispose() {
    if (this.source) {
      this.source.disconnect();
    }
    if (this.audioContext) {
      this.audioContext.close();
    }
    this.isActive = false;
    console.log('✓ Audio analyzer disposed');
  }
}
