'use client';

import { useRouter } from 'next/navigation';
import { Paintbrush, Layers, Palette, BookOpen, Sparkles, ArrowRight } from 'lucide-react';

const features = [
  {
    icon: Layers,
    title: 'Smart Layering',
    description: 'AI-powered scene analysis creates optimal painting order from background to foreground.',
  },
  {
    icon: Palette,
    title: 'Color Guidance',
    description: 'Get detailed color mixing instructions and paint brand recommendations.',
  },
  {
    icon: BookOpen,
    title: 'Step-by-Step Guide',
    description: 'Follow along with three view modes: cumulative, context, and isolated.',
  },
  {
    icon: Sparkles,
    title: 'Professional Results',
    description: 'YOLO semantic segmentation ensures accurate object detection and layering.',
  },
];

const products = [
  {
    name: 'Preview',
    price: 'Free',
    description: 'See what your painting will look like',
    features: ['Low-resolution preview', 'Basic scene analysis', 'Layer count estimate'],
    cta: 'Try Free',
    highlighted: false,
  },
  {
    name: 'Basic',
    price: '$4.99',
    description: 'Essential paint-by-numbers package',
    features: ['HD cumulative images', 'Scene analysis report', 'Numbered regions'],
    cta: 'Get Started',
    highlighted: false,
  },
  {
    name: 'Standard',
    price: '$9.99',
    description: 'Complete painting experience',
    features: ['All image views', 'Full painting guide', 'Color palette', 'Print-ready PDFs'],
    cta: 'Most Popular',
    highlighted: true,
  },
  {
    name: 'Premium',
    price: '$19.99',
    description: 'Everything you need to paint',
    features: [
      'All Standard features',
      'Paint brand recommendations',
      'Color mixing instructions',
      'Supply shopping list',
    ],
    cta: 'Go Premium',
    highlighted: false,
  },
];

export default function Home() {
  const router = useRouter();

  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <section className="relative overflow-hidden bg-gradient-to-b from-primary-50 to-white">
        <div className="absolute inset-0 bg-grid-gray-100/50 [mask-image:radial-gradient(ellipse_at_center,transparent_20%,black)]" />
        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-24 sm:py-32">
          <div className="text-center">
            <h1 className="font-display text-4xl sm:text-5xl lg:text-6xl font-bold text-gray-900 mb-6">
              Transform Photos into{' '}
              <span className="text-primary-600">Paint-by-Numbers</span>{' '}
              Masterpieces
            </h1>
            <p className="text-lg sm:text-xl text-gray-600 max-w-2xl mx-auto mb-8">
              Upload any image and our AI will create a beautiful, paintable artwork with
              numbered regions, color guides, and step-by-step instructions.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <button
                onClick={() => router.push('/upload')}
                className="btn-primary inline-flex items-center justify-center gap-2 text-lg px-8 py-3"
              >
                <Paintbrush className="w-5 h-5" />
                Create Your Painting
                <ArrowRight className="w-4 h-4" />
              </button>
              <a
                href="#pricing"
                className="btn-secondary inline-flex items-center justify-center gap-2 text-lg px-8 py-3"
              >
                View Pricing
              </a>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-12">
            <h2 className="font-display text-3xl sm:text-4xl font-bold text-gray-900 mb-4">
              AI-Powered Painting Intelligence
            </h2>
            <p className="text-gray-600 max-w-2xl mx-auto">
              Our advanced algorithms analyze your image to create the perfect painting experience.
            </p>
          </div>
          <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-8">
            {features.map((feature) => (
              <div key={feature.title} className="card text-center">
                <div className="w-12 h-12 bg-primary-100 rounded-xl flex items-center justify-center mx-auto mb-4">
                  <feature.icon className="w-6 h-6 text-primary-600" />
                </div>
                <h3 className="font-semibold text-gray-900 mb-2">{feature.title}</h3>
                <p className="text-gray-600 text-sm">{feature.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section className="py-20 bg-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-12">
            <h2 className="font-display text-3xl sm:text-4xl font-bold text-gray-900 mb-4">
              How It Works
            </h2>
          </div>
          <div className="grid md:grid-cols-3 gap-8">
            <div className="text-center">
              <div className="w-16 h-16 bg-primary-600 text-white rounded-full flex items-center justify-center mx-auto mb-4 text-2xl font-bold">
                1
              </div>
              <h3 className="font-semibold text-gray-900 mb-2">Upload Your Photo</h3>
              <p className="text-gray-600">
                Choose any photo - pets, landscapes, portraits, or anything you want to paint.
              </p>
            </div>
            <div className="text-center">
              <div className="w-16 h-16 bg-primary-600 text-white rounded-full flex items-center justify-center mx-auto mb-4 text-2xl font-bold">
                2
              </div>
              <h3 className="font-semibold text-gray-900 mb-2">AI Processes Your Image</h3>
              <p className="text-gray-600">
                Our AI analyzes the scene, detects objects, and creates optimal painting layers.
              </p>
            </div>
            <div className="text-center">
              <div className="w-16 h-16 bg-primary-600 text-white rounded-full flex items-center justify-center mx-auto mb-4 text-2xl font-bold">
                3
              </div>
              <h3 className="font-semibold text-gray-900 mb-2">Download & Paint</h3>
              <p className="text-gray-600">
                Get your numbered guide, step-by-step images, and start creating your masterpiece.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Pricing Section */}
      <section id="pricing" className="py-20 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-12">
            <h2 className="font-display text-3xl sm:text-4xl font-bold text-gray-900 mb-4">
              Simple, Transparent Pricing
            </h2>
            <p className="text-gray-600 max-w-2xl mx-auto">
              Start with a free preview, then choose the package that fits your needs.
            </p>
          </div>
          <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-6">
            {products.map((product) => (
              <div
                key={product.name}
                className={`card relative ${
                  product.highlighted
                    ? 'border-2 border-primary-500 ring-4 ring-primary-100'
                    : ''
                }`}
              >
                {product.highlighted && (
                  <div className="absolute -top-3 left-1/2 -translate-x-1/2 bg-primary-600 text-white text-xs font-medium px-3 py-1 rounded-full">
                    Most Popular
                  </div>
                )}
                <div className="text-center mb-4">
                  <h3 className="font-semibold text-gray-900 text-lg">{product.name}</h3>
                  <p className="text-3xl font-bold text-gray-900 mt-2">{product.price}</p>
                  <p className="text-gray-500 text-sm mt-1">{product.description}</p>
                </div>
                <ul className="space-y-2 mb-6">
                  {product.features.map((feature) => (
                    <li key={feature} className="flex items-start gap-2 text-sm text-gray-600">
                      <svg
                        className="w-5 h-5 text-accent-500 flex-shrink-0"
                        fill="none"
                        viewBox="0 0 24 24"
                        stroke="currentColor"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M5 13l4 4L19 7"
                        />
                      </svg>
                      {feature}
                    </li>
                  ))}
                </ul>
                <button
                  onClick={() => router.push('/upload')}
                  className={`w-full py-2 px-4 rounded-lg font-medium transition-colors ${
                    product.highlighted
                      ? 'bg-primary-600 hover:bg-primary-700 text-white'
                      : 'bg-gray-100 hover:bg-gray-200 text-gray-900'
                  }`}
                >
                  {product.cta}
                </button>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 bg-primary-600">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <h2 className="font-display text-3xl sm:text-4xl font-bold text-white mb-4">
            Ready to Create Your Masterpiece?
          </h2>
          <p className="text-primary-100 text-lg mb-8">
            Upload your photo now and see the magic of AI-powered paint-by-numbers.
          </p>
          <button
            onClick={() => router.push('/upload')}
            className="bg-white text-primary-600 hover:bg-primary-50 font-medium py-3 px-8 rounded-lg inline-flex items-center gap-2 transition-colors"
          >
            <Paintbrush className="w-5 h-5" />
            Start Creating
            <ArrowRight className="w-4 h-4" />
          </button>
        </div>
      </section>
    </div>
  );
}
