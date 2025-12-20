/** @type {import('next').NextConfig} */
const nextConfig = {
  // Enable standalone output for Docker deployment
  output: 'standalone',

  // API rewrites to backend (excluding local API routes like /api/image)
  async rewrites() {
    return {
      beforeFiles: [
        // Don't rewrite local API routes - let Next.js handle them
      ],
      afterFiles: [
        // Rewrite other API calls to backend, but NOT /api/image (handled by our proxy)
        {
          source: '/api/upload',
          destination: process.env.NEXT_PUBLIC_API_URL
            ? `${process.env.NEXT_PUBLIC_API_URL}/api/upload`
            : 'http://localhost:8000/api/upload',
        },
        {
          source: '/api/process/:path*',
          destination: process.env.NEXT_PUBLIC_API_URL
            ? `${process.env.NEXT_PUBLIC_API_URL}/api/process/:path*`
            : 'http://localhost:8000/api/process/:path*',
        },
        {
          source: '/api/status/:path*',
          destination: process.env.NEXT_PUBLIC_API_URL
            ? `${process.env.NEXT_PUBLIC_API_URL}/api/status/:path*`
            : 'http://localhost:8000/api/status/:path*',
        },
        {
          source: '/api/results/:path*',
          destination: process.env.NEXT_PUBLIC_API_URL
            ? `${process.env.NEXT_PUBLIC_API_URL}/api/results/:path*`
            : 'http://localhost:8000/api/results/:path*',
        },
        {
          source: '/api/products/:path*',
          destination: process.env.NEXT_PUBLIC_API_URL
            ? `${process.env.NEXT_PUBLIC_API_URL}/api/products/:path*`
            : 'http://localhost:8000/api/products/:path*',
        },
        {
          source: '/api/preview/:path*',
          destination: process.env.NEXT_PUBLIC_API_URL
            ? `${process.env.NEXT_PUBLIC_API_URL}/api/preview/:path*`
            : 'http://localhost:8000/api/preview/:path*',
        },
        {
          source: '/api/checkout',
          destination: process.env.NEXT_PUBLIC_API_URL
            ? `${process.env.NEXT_PUBLIC_API_URL}/api/checkout`
            : 'http://localhost:8000/api/checkout',
        },
        {
          source: '/api/guide/:path*',
          destination: process.env.NEXT_PUBLIC_API_URL
            ? `${process.env.NEXT_PUBLIC_API_URL}/api/guide/:path*`
            : 'http://localhost:8000/api/guide/:path*',
        },
        {
          source: '/api/promo/:path*',
          destination: process.env.NEXT_PUBLIC_API_URL
            ? `${process.env.NEXT_PUBLIC_API_URL}/api/promo/:path*`
            : 'http://localhost:8000/api/promo/:path*',
        },
      ],
    };
  },
  images: {
    remotePatterns: [
      {
        protocol: 'http',
        hostname: 'localhost',
        port: '8000',
      },
    ],
  },
};

module.exports = nextConfig;
