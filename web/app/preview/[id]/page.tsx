'use client';

import { useState, useEffect, useCallback } from 'react';
import { useRouter, useParams } from 'next/navigation';
import { ArrowLeft, Loader2, AlertCircle, Sun, Cloud, MapPin, Lightbulb, Sparkles } from 'lucide-react';
import PreviewGallery from '@/components/PreviewGallery';
import ProductSelector from '@/components/ProductSelector';
import api, { JobResults, Product, PaintingGuide } from '@/lib/api';

export default function PreviewPage() {
  const router = useRouter();
  const params = useParams();
  const jobId = params.id as string;

  const [results, setResults] = useState<JobResults | null>(null);
  const [products, setProducts] = useState<Product[]>([]);
  const [paintingGuide, setPaintingGuide] = useState<PaintingGuide | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isCheckingOut, setIsCheckingOut] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [promoApplied, setPromoApplied] = useState(false);

  useEffect(() => {
    async function loadData() {
      try {
        const [resultsData, productsData] = await Promise.all([
          api.getResults(jobId),
          api.getProducts(jobId),
        ]);
        setResults(resultsData);
        setProducts(productsData);

        // Also fetch painting guide (non-blocking)
        try {
          const guideData = await api.getPaintingGuide(jobId);
          setPaintingGuide(guideData);
        } catch {
          // Guide not available, that's okay
          console.log('Painting guide not available');
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load results');
      } finally {
        setIsLoading(false);
      }
    }

    loadData();
  }, [jobId]);

  const handleCheckout = useCallback(
    async (selectedProductIds: string[]) => {
      setIsCheckingOut(true);
      setError(null);

      try {
        const successUrl = `${window.location.origin}/download/${jobId}`;
        const cancelUrl = window.location.href;

        const { checkout_url, session_id } = await api.createCheckout(
          jobId,
          selectedProductIds,
          successUrl,
          cancelUrl
        );

        // For free products, session_id will be "free"
        if (session_id === 'free') {
          router.push(`/download/${jobId}`);
        } else {
          // Redirect to Stripe
          window.location.href = checkout_url;
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Checkout failed');
      } finally {
        setIsCheckingOut(false);
      }
    },
    [jobId, router]
  );

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="w-12 h-12 text-primary-600 animate-spin mx-auto mb-4" />
          <p className="text-gray-600">Loading your results...</p>
        </div>
      </div>
    );
  }

  if (error && !results) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="card max-w-md text-center">
          <AlertCircle className="w-12 h-12 text-red-500 mx-auto mb-4" />
          <h2 className="font-semibold text-gray-900 mb-2">Error Loading Results</h2>
          <p className="text-gray-600 mb-4">{error}</p>
          <button onClick={() => router.push('/upload')} className="btn-primary">
            Try Again
          </button>
        </div>
      </div>
    );
  }

  const sceneIcons: Record<string, typeof Sun> = {
    time_of_day: Sun,
    weather: Cloud,
    setting: MapPin,
    lighting_type: Lightbulb,
    mood: Sparkles,
  };

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-6xl mx-auto px-4 sm:px-6">
        {/* Header */}
        <div className="mb-6">
          <button
            onClick={() => router.push('/upload')}
            className="flex items-center gap-2 text-gray-600 hover:text-gray-900 transition-colors mb-4"
          >
            <ArrowLeft className="w-4 h-4" />
            Create Another
          </button>
          <h1 className="font-display text-3xl font-bold text-gray-900">
            Your Paint-by-Numbers Preview
          </h1>
          <p className="text-gray-600 mt-2">
            {results?.total_steps} painting steps generated. Browse the preview and choose your package.
          </p>
        </div>

        {/* Error Banner */}
        {error && (
          <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg flex items-start gap-3">
            <AlertCircle className="w-5 h-5 text-red-500 flex-shrink-0 mt-0.5" />
            <p className="text-red-700">{error}</p>
          </div>
        )}

        <div className="grid lg:grid-cols-3 gap-8">
          {/* Preview Gallery */}
          <div className="lg:col-span-2">
            <div className="card">
              <h2 className="font-semibold text-gray-900 mb-4">Preview Gallery</h2>
              {results && (
                <PreviewGallery
                  outputs={results.outputs}
                  totalSteps={results.total_steps}
                  paintingGuide={paintingGuide || undefined}
                />
              )}
            </div>

            {/* Scene Analysis */}
            {results?.scene_analysis && (
              <div className="card mt-6">
                <h2 className="font-semibold text-gray-900 mb-4">Scene Analysis</h2>
                <div className="grid grid-cols-2 sm:grid-cols-5 gap-4">
                  {Object.entries(results.scene_analysis).map(([key, value]) => {
                    const Icon = sceneIcons[key] || Sparkles;
                    const label = key.replace(/_/g, ' ');

                    return (
                      <div
                        key={key}
                        className="bg-gray-50 rounded-lg p-3 text-center"
                      >
                        <Icon className="w-5 h-5 text-primary-600 mx-auto mb-1" />
                        <p className="text-xs text-gray-500 capitalize">{label}</p>
                        <p className="text-sm font-medium text-gray-900 capitalize">
                          {value}
                        </p>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </div>

          {/* Product Selection */}
          <div className="lg:col-span-1">
            <div className="card sticky top-24">
              <h2 className="font-semibold text-gray-900 mb-4">
                Select Your Package
              </h2>
              <ProductSelector
                products={products}
                onCheckout={handleCheckout}
                isLoading={isCheckingOut}
                jobId={jobId}
                onPromoApplied={(tier) => setPromoApplied(true)}
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
