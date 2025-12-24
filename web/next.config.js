/** @type {import('next').NextConfig} */

// Default to Modal backend if env var not set
const API_URL = process.env.NEXT_PUBLIC_API_URL || 'https://sausterm--artisan-api-fastapi-app.modal.run';

const nextConfig = {
  // Enable standalone output for Docker deployment
  output: 'standalone',

  // API rewrites to backend (excluding local API routes like /api/image)
  async rewrites() {
    return {
      beforeFiles: [],
      afterFiles: [
        { source: '/api/upload', destination: `${API_URL}/api/upload` },
        { source: '/api/process/:path*', destination: `${API_URL}/api/process/:path*` },
        { source: '/api/status/:path*', destination: `${API_URL}/api/status/:path*` },
        { source: '/api/results/:path*', destination: `${API_URL}/api/results/:path*` },
        { source: '/api/products/:path*', destination: `${API_URL}/api/products/:path*` },
        { source: '/api/preview/:path*', destination: `${API_URL}/api/preview/:path*` },
        { source: '/api/checkout', destination: `${API_URL}/api/checkout` },
        { source: '/api/guide/:path*', destination: `${API_URL}/api/guide/:path*` },
        { source: '/api/promo/:path*', destination: `${API_URL}/api/promo/:path*` },
        { source: '/api/styles', destination: `${API_URL}/api/styles` },
        { source: '/api/style-transfer', destination: `${API_URL}/api/style-transfer` },
        { source: '/api/style-transfer/:path*', destination: `${API_URL}/api/style-transfer/:path*` },
        { source: '/api/job/:path*', destination: `${API_URL}/api/job/:path*` },
        { source: '/api/health', destination: `${API_URL}/health` },
      ],
    };
  },
  images: {
    remotePatterns: [
      { protocol: 'http', hostname: 'localhost', port: '8000' },
      { protocol: 'https', hostname: '*.modal.run' },
      { protocol: 'https', hostname: '*.ngrok-free.app' },
      { protocol: 'https', hostname: '*.ngrok.io' },
    ],
  },
};

module.exports = nextConfig;
