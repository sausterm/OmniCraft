/**
 * API client for Artisan Paint-by-Numbers backend
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL || '';

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

class ApiClient {
  private async request<T>(
    endpoint: string,
    options?: RequestInit
  ): Promise<T> {
    const response = await fetch(`${API_BASE}${endpoint}`, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
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
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${API_BASE}/api/upload`, {
      method: 'POST',
      body: formData,
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
    return this.request<JobResults>(`/api/results/${jobId}`);
  }

  /**
   * Get available products for a job
   */
  async getProducts(jobId: string): Promise<Product[]> {
    return this.request<Product[]>(`/api/products/${jobId}`);
  }

  /**
   * Get preview image URL
   */
  getPreviewUrl(jobId: string): string {
    return `${API_BASE}/api/preview/${jobId}`;
  }

  /**
   * Get download URL for a product
   */
  getDownloadUrl(jobId: string, productId: string): string {
    return `${API_BASE}/api/download/${jobId}/${productId}`;
  }

  /**
   * Create Stripe checkout session
   */
  async createCheckout(
    jobId: string,
    productIds: string[],
    successUrl?: string,
    cancelUrl?: string
  ): Promise<CheckoutResponse> {
    return this.request<CheckoutResponse>('/api/checkout', {
      method: 'POST',
      body: JSON.stringify({
        job_id: jobId,
        product_ids: productIds,
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
