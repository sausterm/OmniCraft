/**
 * API client for Artisan Paint-by-Numbers backend
 * Supports primary (local/ngrok) and fallback (cloud) backends
 */

// Primary backend (local ngrok tunnel)
const PRIMARY_API = process.env.NEXT_PUBLIC_API_URL || '';
// Fallback backend (Modal/Railway cloud)
const FALLBACK_API = process.env.NEXT_PUBLIC_FALLBACK_API_URL || '';

// Track which backend is active
let activeBackend: 'primary' | 'fallback' = 'primary';
let lastHealthCheck = 0;
const HEALTH_CHECK_INTERVAL = 30000; // 30 seconds

function getApiBase(): string {
  return activeBackend === 'primary' ? PRIMARY_API : FALLBACK_API;
}

async function checkBackendHealth(url: string): Promise<boolean> {
  if (!url) return false;
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 5000);
    const response = await fetch(`${url}/health`, {
      signal: controller.signal,
      headers: { 'ngrok-skip-browser-warning': 'true' },
    });
    clearTimeout(timeoutId);
    return response.ok;
  } catch {
    return false;
  }
}

async function ensureHealthyBackend(): Promise<string> {
  const now = Date.now();

  // Skip health check if we checked recently
  if (now - lastHealthCheck < HEALTH_CHECK_INTERVAL) {
    return getApiBase();
  }

  lastHealthCheck = now;

  // Try primary first
  if (PRIMARY_API && await checkBackendHealth(PRIMARY_API)) {
    activeBackend = 'primary';
    return PRIMARY_API;
  }

  // Fall back to cloud
  if (FALLBACK_API && await checkBackendHealth(FALLBACK_API)) {
    activeBackend = 'fallback';
    console.log('Switched to fallback backend');
    return FALLBACK_API;
  }

  // Return whatever we have
  return getApiBase();
}

// Legacy constant for backwards compatibility
const API_BASE = PRIMARY_API;

export interface ProcessConfig {
  model_size?: 'n' | 's' | 'm' | 'l' | 'x';
  confidence?: number;
  num_colors?: number;
  min_region_size?: number;
}

export interface Job {
  job_id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  created_at: string;
  filename: string;
  config?: ProcessConfig;
  error?: string;
  // Style transfer
  style_status?: 'queued' | 'processing' | 'completed' | 'failed' | null;
  style_config?: StyleConfig | null;
  styled_image_url?: string | null;
}

export interface StyleConfig {
  style: string;
  custom_prompt?: string;
  guidance_scale: number;
  control_strength: number;
  num_steps: number;
  processing_time?: number;
}

export interface StyleInfo {
  id: string;
  name: string;
  description: string;
}

export interface StyleTransferRequest {
  job_id: string;
  style: string;
  custom_prompt?: string;
  guidance_scale?: number;
  control_strength?: number;
  num_steps?: number;
}

export interface StyleTransferResponse {
  job_id: string;
  status: string;
  message: string;
  style: string;
  estimated_time_seconds: number;
}

export interface JobResults {
  job_id: string;
  status: string;
  total_steps: number;
  scene_analysis: {
    time_of_day: string;
    weather: string;
    setting: string;
    lighting_type: string;
    mood: string;
  };
  outputs: {
    cumulative: string[];
    context: string[];
    isolated: string[];
  };
}

// Raw API response format
interface ApiProcessingResult {
  job_id: string;
  scene_context: {
    time_of_day: string;
    weather: string;
    setting: string;
    lighting: string;
    mood: string;
    light_direction: string;
  };
  regions: Array<{
    name: string;
    subject_type: string;
    category: string;
    coverage: number;
    is_focal: boolean;
    substeps: number;
  }>;
  total_substeps: number;
  output_files: {
    cumulative: number;
    context: number;
    isolated: number;
  };
}

export interface Product {
  id: string;
  name: string;
  description: string;
  price: number;
  includes: string[];
  purchased: boolean;
}

export interface CheckoutResponse {
  checkout_url: string;
  session_id: string;
}

export interface PromoCodeResponse {
  valid: boolean;
  message: string;
  tier?: string;
}

export interface PaintingStep {
  step_number: number;
  region_name: string;
  technique: string;
  brush_type: string;
  stroke_motion: string;
  colors: string[];
  instruction: string;
  tips: string[];
}

export interface PaintingGuide {
  total_steps: number;
  scene_context: {
    time_of_day: string;
    weather: string;
    setting: string;
    lighting: string;
    mood: string;
  };
  steps: PaintingStep[];
}

class ApiClient {
  private async request<T>(
    endpoint: string,
    options?: RequestInit
  ): Promise<T> {
    const baseUrl = await ensureHealthyBackend();
    const response = await fetch(`${baseUrl}${endpoint}`, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        'ngrok-skip-browser-warning': 'true',
        ...options?.headers,
      },
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(error.detail || `HTTP ${response.status}`);
    }

    return response.json();
  }

  /**
   * Upload an image for processing
   */
  async uploadImage(file: File): Promise<Job> {
    const baseUrl = await ensureHealthyBackend();
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${baseUrl}/api/upload`, {
      method: 'POST',
      body: formData,
      headers: {
        'ngrok-skip-browser-warning': 'true',
      },
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Upload failed' }));
      throw new Error(error.detail || `HTTP ${response.status}`);
    }

    return response.json();
  }

  /**
   * Get job status
   */
  async getJob(jobId: string): Promise<Job> {
    return this.request<Job>(`/api/job/${jobId}`);
  }

  /**
   * Start processing a job
   */
  async processJob(jobId: string, config?: ProcessConfig): Promise<Job> {
    return this.request<Job>(`/api/process/${jobId}`, {
      method: 'POST',
      body: JSON.stringify(config || {}),
    });
  }

  /**
   * Get job status (polling endpoint)
   */
  async getStatus(jobId: string): Promise<Job> {
    return this.request<Job>(`/api/status/${jobId}`);
  }

  /**
   * Get job results after completion
   */
  async getResults(jobId: string): Promise<JobResults> {
    const apiResult = await this.request<ApiProcessingResult>(`/api/results/${jobId}`);

    // Transform API response to frontend format
    // Use local proxy route to avoid ngrok header issues with <img> tags
    const generateUrls = (type: string, count: number): string[] => {
      return Array.from({ length: count }, (_, i) =>
        `/api/image/${jobId}/${type}/${i}`
      );
    };

    return {
      job_id: apiResult.job_id,
      status: 'completed',
      total_steps: apiResult.total_substeps,
      scene_analysis: {
        time_of_day: apiResult.scene_context.time_of_day,
        weather: apiResult.scene_context.weather,
        setting: apiResult.scene_context.setting,
        lighting_type: apiResult.scene_context.lighting,
        mood: apiResult.scene_context.mood,
      },
      outputs: {
        cumulative: generateUrls('cumulative', apiResult.output_files.cumulative || 0),
        context: generateUrls('context', apiResult.output_files.context || 0),
        isolated: generateUrls('isolated', apiResult.output_files.isolated || 0),
      },
    };
  }

  /**
   * Get available products for a job
   */
  async getProducts(jobId: string): Promise<Product[]> {
    const response = await this.request<{ job_id: string; products: Product[]; purchased: string[] }>(
      `/api/products/${jobId}`
    );
    // Mark products as purchased based on the purchased array
    return (response.products || []).map(p => ({
      ...p,
      purchased: response.purchased?.includes(p.id) || false,
    }));
  }

  /**
   * Get preview image URL
   */
  getPreviewUrl(jobId: string): string {
    return `${getApiBase()}/api/preview/${jobId}`;
  }

  /**
   * Get download URL for a product
   */
  getDownloadUrl(jobId: string, productId: string): string {
    return `${getApiBase()}/api/download/${jobId}/${productId}`;
  }

  /**
   * Create Stripe checkout session
   */
  async createCheckout(
    jobId: string,
    productIds: string[],
    email: string,
    successUrl?: string,
    cancelUrl?: string
  ): Promise<CheckoutResponse> {
    return this.request<CheckoutResponse>('/api/checkout', {
      method: 'POST',
      body: JSON.stringify({
        job_id: jobId,
        product_ids: productIds,
        email,
        success_url: successUrl,
        cancel_url: cancelUrl,
      }),
    });
  }

  /**
   * Verify payment status
   */
  async verifyPayment(sessionId: string): Promise<{
    status: string;
    job_id: string;
    products: string[];
  }> {
    return this.request(`/api/verify/${sessionId}`);
  }

  /**
   * Validate and redeem a promo code
   */
  async validatePromoCode(code: string, jobId: string): Promise<PromoCodeResponse> {
    return this.request<PromoCodeResponse>('/api/promo/validate', {
      method: 'POST',
      body: JSON.stringify({ code, job_id: jobId }),
    });
  }

  /**
   * Get the painting guide with step-by-step instructions
   */
  async getPaintingGuide(jobId: string): Promise<PaintingGuide> {
    return this.request<PaintingGuide>(`/api/guide/${jobId}`);
  }

  /**
   * Get available style presets
   */
  async getStyles(): Promise<StyleInfo[]> {
    const response = await this.request<{ styles: StyleInfo[] }>('/api/styles');
    return response.styles;
  }

  /**
   * Apply style transfer to an image
   */
  async applyStyleTransfer(request: StyleTransferRequest): Promise<StyleTransferResponse> {
    return this.request<StyleTransferResponse>('/api/style-transfer', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  /**
   * Get style transfer status
   */
  async getStyleTransferStatus(jobId: string): Promise<{
    job_id: string;
    style_status: string;
    style_config?: StyleConfig;
    styled_image_url?: string;
    processing_time?: number;
    error?: string;
  }> {
    return this.request(`/api/style-transfer/${jobId}`);
  }

  /**
   * Remove style transfer from a job
   */
  async removeStyleTransfer(jobId: string): Promise<{ job_id: string; message: string }> {
    return this.request(`/api/style-transfer/${jobId}`, {
      method: 'DELETE',
    });
  }

  /**
   * Get original image URL
   */
  getOriginalImageUrl(jobId: string): string {
    return `${getApiBase()}/api/preview/${jobId}/original`;
  }

  /**
   * Get styled image URL
   */
  getStyledImageUrl(jobId: string): string {
    return `${getApiBase()}/api/preview/${jobId}/styled`;
  }

  /**
   * Get current backend status
   */
  getBackendStatus(): { active: 'primary' | 'fallback'; primary: string; fallback: string } {
    return {
      active: activeBackend,
      primary: PRIMARY_API,
      fallback: FALLBACK_API,
    };
  }

  /**
   * Poll for style transfer completion
   */
  async waitForStyleTransfer(
    jobId: string,
    onProgress?: (status: string) => void,
    maxWaitMs = 600000,  // 10 minutes for style transfer
    pollIntervalMs = 3000
  ): Promise<{ styled_image_url?: string; error?: string }> {
    const startTime = Date.now();

    while (Date.now() - startTime < maxWaitMs) {
      const result = await this.getStyleTransferStatus(jobId);
      onProgress?.(result.style_status);

      if (result.style_status === 'completed') {
        return { styled_image_url: result.styled_image_url };
      }

      if (result.style_status === 'failed') {
        return { error: result.error || 'Style transfer failed' };
      }

      await new Promise((resolve) => setTimeout(resolve, pollIntervalMs));
    }

    throw new Error('Style transfer timeout');
  }

  /**
   * Poll for job completion
   */
  async waitForCompletion(
    jobId: string,
    onProgress?: (job: Job) => void,
    maxWaitMs = 300000,
    pollIntervalMs = 2000
  ): Promise<Job> {
    const startTime = Date.now();

    while (Date.now() - startTime < maxWaitMs) {
      const job = await this.getStatus(jobId);
      onProgress?.(job);

      if (job.status === 'completed' || job.status === 'failed') {
        return job;
      }

      await new Promise((resolve) => setTimeout(resolve, pollIntervalMs));
    }

    throw new Error('Processing timeout');
  }
}

export const api = new ApiClient();
export default api;
