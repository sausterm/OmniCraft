'use client';

import { useState, useEffect } from 'react';
import { useRouter, useParams, useSearchParams } from 'next/navigation';
import {
  Download,
  Check,
  Loader2,
  AlertCircle,
  Package,
  FileImage,
  FileText,
  Palette,
} from 'lucide-react';
import api, { Product } from '@/lib/api';

interface DownloadItem {
  id: string;
  name: string;
  description: string;
  icon: typeof FileImage;
  downloadUrl: string;
}

export default function DownloadPage() {
  const router = useRouter();
  const params = useParams();
  const searchParams = useSearchParams();
  const jobId = params.id as string;
  const sessionId = searchParams.get('session_id');

  const [products, setProducts] = useState<Product[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isVerifying, setIsVerifying] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [downloadingId, setDownloadingId] = useState<string | null>(null);

  useEffect(() => {
    async function loadAndVerify() {
      try {
        // If coming from Stripe, verify the payment first
        if (sessionId && sessionId !== 'free') {
          setIsVerifying(true);
          const verification = await api.verifyPayment(sessionId);
          if (verification.status !== 'paid') {
            setError('Payment not completed. Please try again.');
            setIsVerifying(false);
            return;
          }
          setIsVerifying(false);
        }

        // Load products to see what's purchased
        const productsData = await api.getProducts(jobId);
        setProducts(productsData);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load downloads');
      } finally {
        setIsLoading(false);
      }
    }

    loadAndVerify();
  }, [jobId, sessionId]);

  const purchasedProducts = products.filter((p) => p.purchased);

  const getProductIcon = (productId: string) => {
    switch (productId) {
      case 'basic':
      case 'standard':
      case 'premium':
        return Package;
      case 'paint_kit':
        return Palette;
      case 'mixing_guide':
        return FileText;
      default:
        return FileImage;
    }
  };

  const handleDownload = async (productId: string) => {
    setDownloadingId(productId);
    try {
      const url = api.getDownloadUrl(jobId, productId);
      // Open in new tab or trigger download
      window.open(url, '_blank');
    } finally {
      setTimeout(() => setDownloadingId(null), 1000);
    }
  };

  if (isLoading || isVerifying) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="w-12 h-12 text-primary-600 animate-spin mx-auto mb-4" />
          <p className="text-gray-600">
            {isVerifying ? 'Verifying your payment...' : 'Loading your downloads...'}
          </p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="card max-w-md text-center">
          <AlertCircle className="w-12 h-12 text-red-500 mx-auto mb-4" />
          <h2 className="font-semibold text-gray-900 mb-2">Error</h2>
          <p className="text-gray-600 mb-4">{error}</p>
          <button onClick={() => router.push(`/preview/${jobId}`)} className="btn-primary">
            Return to Preview
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 py-12">
      <div className="max-w-2xl mx-auto px-4 sm:px-6">
        {/* Success Header */}
        <div className="text-center mb-8">
          <div className="w-16 h-16 bg-accent-100 rounded-full flex items-center justify-center mx-auto mb-4">
            <Check className="w-8 h-8 text-accent-600" />
          </div>
          <h1 className="font-display text-3xl font-bold text-gray-900 mb-2">
            Thank You for Your Purchase!
          </h1>
          <p className="text-gray-600">
            Your paint-by-numbers package is ready to download.
          </p>
        </div>

        {/* Downloads */}
        <div className="card">
          <h2 className="font-semibold text-gray-900 mb-4">Your Downloads</h2>

          {purchasedProducts.length === 0 ? (
            <div className="text-center py-8">
              <Package className="w-12 h-12 text-gray-300 mx-auto mb-4" />
              <p className="text-gray-500 mb-4">No purchased items found.</p>
              <button
                onClick={() => router.push(`/preview/${jobId}`)}
                className="btn-primary"
              >
                View Available Products
              </button>
            </div>
          ) : (
            <div className="space-y-3">
              {purchasedProducts.map((product) => {
                const Icon = getProductIcon(product.id);
                const isDownloading = downloadingId === product.id;

                return (
                  <div
                    key={product.id}
                    className="flex items-center justify-between p-4 bg-gray-50 rounded-lg"
                  >
                    <div className="flex items-center gap-3">
                      <div className="w-10 h-10 bg-white rounded-lg flex items-center justify-center shadow-sm">
                        <Icon className="w-5 h-5 text-primary-600" />
                      </div>
                      <div>
                        <p className="font-medium text-gray-900">{product.name}</p>
                        <p className="text-sm text-gray-500">{product.description}</p>
                      </div>
                    </div>
                    <button
                      onClick={() => handleDownload(product.id)}
                      disabled={isDownloading}
                      className="btn-primary py-2 px-4 flex items-center gap-2"
                    >
                      {isDownloading ? (
                        <Loader2 className="w-4 h-4 animate-spin" />
                      ) : (
                        <Download className="w-4 h-4" />
                      )}
                      Download
                    </button>
                  </div>
                );
              })}
            </div>
          )}

          {/* What's Included */}
          {purchasedProducts.length > 0 && (
            <div className="mt-6 pt-6 border-t border-gray-200">
              <h3 className="font-medium text-gray-900 mb-3">What&apos;s Included</h3>
              <ul className="space-y-2">
                {purchasedProducts.flatMap((p) =>
                  p.includes.map((item) => (
                    <li
                      key={`${p.id}-${item}`}
                      className="flex items-center gap-2 text-sm text-gray-600"
                    >
                      <Check className="w-4 h-4 text-accent-500" />
                      {item.replace(/_/g, ' ')}
                    </li>
                  ))
                )}
              </ul>
            </div>
          )}
        </div>

        {/* Actions */}
        <div className="mt-6 flex flex-col sm:flex-row gap-4 justify-center">
          <button
            onClick={() => router.push('/upload')}
            className="btn-secondary"
          >
            Create Another Painting
          </button>
          <button
            onClick={() => router.push(`/preview/${jobId}`)}
            className="btn-secondary"
          >
            Purchase More Items
          </button>
        </div>

        {/* Support */}
        <p className="text-center text-sm text-gray-500 mt-8">
          Having issues? Contact us at support@artisan-paint.com
        </p>
      </div>
    </div>
  );
}
