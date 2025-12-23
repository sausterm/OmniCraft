import type { Metadata, Viewport } from 'next';
import { Inter, Playfair_Display } from 'next/font/google';
import './globals.css';
import Providers from '@/components/Providers';
import AuthButton from '@/components/AuthButton';

const inter = Inter({
  subsets: ['latin'],
  variable: '--font-inter',
});

const playfair = Playfair_Display({
  subsets: ['latin'],
  variable: '--font-playfair',
});

export const viewport: Viewport = {
  width: 'device-width',
  initialScale: 1,
  maximumScale: 5,
  userScalable: true,
  themeColor: '#4f46e5',
};

export const metadata: Metadata = {
  title: 'Artisan Paint-by-Numbers',
  description: 'Transform your photos into beautiful paint-by-numbers masterpieces with AI-powered semantic segmentation',
  keywords: ['paint by numbers', 'AI art', 'photo to painting', 'art generator'],
  appleWebApp: {
    capable: true,
    statusBarStyle: 'default',
    title: 'Artisan',
  },
  formatDetection: {
    telephone: false,
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={`${inter.variable} ${playfair.variable}`}>
      <body className="min-h-screen font-sans antialiased">
        <Providers>
          <header className="border-b border-gray-100 bg-white/80 backdrop-blur-sm sticky top-0 z-50">
            <nav className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-14 sm:h-16 flex items-center justify-between">
              <a href="/" className="flex items-center gap-2 touch-manipulation">
                <div className="w-8 h-8 bg-gradient-to-br from-primary-500 to-primary-700 rounded-lg flex items-center justify-center">
                  <svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 21a4 4 0 01-4-4V5a2 2 0 012-2h4a2 2 0 012 2v12a4 4 0 01-4 4zm0 0h12a2 2 0 002-2v-4a2 2 0 00-2-2h-2.343M11 7.343l1.657-1.657a2 2 0 012.828 0l2.829 2.829a2 2 0 010 2.828l-8.486 8.485M7 17h.01" />
                  </svg>
                </div>
                <span className="font-display text-lg sm:text-xl font-semibold text-gray-900">Artisan</span>
              </a>
              <div className="flex items-center gap-3 sm:gap-6">
                <a href="/upload" className="text-gray-600 hover:text-gray-900 text-sm font-medium transition-colors min-h-[44px] flex items-center touch-manipulation">
                  Create
                </a>
                <a href="#pricing" className="hidden sm:flex text-gray-600 hover:text-gray-900 text-sm font-medium transition-colors min-h-[44px] items-center touch-manipulation">
                  Pricing
                </a>
                <AuthButton />
              </div>
            </nav>
          </header>
          <main>{children}</main>
          <footer className="border-t border-gray-100 bg-white mt-auto">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
              <p className="text-center text-gray-500 text-sm">
                &copy; {new Date().getFullYear()} Artisan Paint-by-Numbers. All rights reserved.
              </p>
            </div>
          </footer>
        </Providers>
      </body>
    </html>
  );
}
