import { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Terms of Service - Artisan Paint-by-Numbers',
  description: 'Terms of Service for Artisan Paint-by-Numbers',
};

export default function TermsPage() {
  return (
    <div className="min-h-screen bg-gray-50 py-12">
      <div className="max-w-3xl mx-auto px-4 sm:px-6">
        <h1 className="font-display text-3xl font-bold text-gray-900 mb-8">
          Terms of Service
        </h1>

        <div className="prose prose-gray max-w-none">
          <p className="text-sm text-gray-500 mb-6">Last updated: December 24, 2024</p>

          <section className="mb-8">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">1. Agreement to Terms</h2>
            <p className="text-gray-600 mb-4">
              By accessing or using Artisan Paint-by-Numbers ("Service"), you agree to be bound by these
              Terms of Service. If you do not agree to these terms, please do not use our Service.
            </p>
          </section>

          <section className="mb-8">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">2. Description of Service</h2>
            <p className="text-gray-600 mb-4">
              Artisan Paint-by-Numbers is an AI-powered service that transforms uploaded photos into
              paint-by-numbers guides. Our Service includes:
            </p>
            <ul className="list-disc pl-6 text-gray-600 mb-4">
              <li>Image processing and analysis</li>
              <li>Generation of numbered painting regions</li>
              <li>Color palette and mixing recommendations</li>
              <li>Step-by-step painting instructions</li>
              <li>Downloadable digital files (images, PDFs)</li>
            </ul>
            <p className="text-gray-600">
              All products are delivered as digital downloads. No physical goods are shipped.
            </p>
          </section>

          <section className="mb-8">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">3. User Responsibilities</h2>
            <p className="text-gray-600 mb-4">You agree to:</p>
            <ul className="list-disc pl-6 text-gray-600 mb-4">
              <li>Only upload images you own or have rights to use</li>
              <li>Not upload illegal, harmful, or inappropriate content</li>
              <li>Not attempt to reverse-engineer or exploit our Service</li>
              <li>Provide accurate payment and contact information</li>
            </ul>
          </section>

          <section className="mb-8">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">4. Intellectual Property</h2>
            <p className="text-gray-600 mb-4">
              <strong>Your Content:</strong> You retain ownership of images you upload. By uploading,
              you grant us a limited license to process your images solely to provide the Service.
            </p>
            <p className="text-gray-600 mb-4">
              <strong>Generated Content:</strong> Upon purchase, you receive a personal, non-exclusive
              license to use the generated paint-by-numbers guides for personal, non-commercial purposes.
            </p>
            <p className="text-gray-600">
              <strong>Our Service:</strong> The Artisan platform, algorithms, and branding remain our
              intellectual property.
            </p>
          </section>

          <section className="mb-8">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">5. Payments and Refunds</h2>
            <p className="text-gray-600 mb-4">
              <strong>Pricing:</strong> All prices are in USD and displayed before purchase.
              Prices may change without notice.
            </p>
            <p className="text-gray-600 mb-4">
              <strong>Payment Processing:</strong> Payments are processed securely through Stripe.
              We do not store your credit card information.
            </p>
            <p className="text-gray-600">
              <strong>Refunds:</strong> Due to the digital nature of our products, we generally do not
              offer refunds once files have been downloaded. If you experience technical issues preventing
              you from accessing your purchase, please contact us and we will work to resolve the issue
              or provide a refund at our discretion.
            </p>
          </section>

          <section className="mb-8">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">6. Disclaimer of Warranties</h2>
            <p className="text-gray-600 mb-4">
              The Service is provided "as is" without warranties of any kind. We do not guarantee that:
            </p>
            <ul className="list-disc pl-6 text-gray-600 mb-4">
              <li>The Service will be uninterrupted or error-free</li>
              <li>Results will meet your specific expectations</li>
              <li>Generated guides will be suitable for all skill levels</li>
            </ul>
          </section>

          <section className="mb-8">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">7. Limitation of Liability</h2>
            <p className="text-gray-600">
              To the maximum extent permitted by law, we shall not be liable for any indirect,
              incidental, special, or consequential damages arising from your use of the Service.
              Our total liability shall not exceed the amount you paid for the specific product
              giving rise to the claim.
            </p>
          </section>

          <section className="mb-8">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">8. Changes to Terms</h2>
            <p className="text-gray-600">
              We may update these Terms from time to time. Continued use of the Service after
              changes constitutes acceptance of the new Terms.
            </p>
          </section>

          <section className="mb-8">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">9. Contact</h2>
            <p className="text-gray-600">
              For questions about these Terms, please contact us at support@artisan.com.
            </p>
          </section>
        </div>

        <div className="mt-12 pt-8 border-t border-gray-200">
          <a href="/" className="text-primary-600 hover:text-primary-700 font-medium">
            &larr; Back to Home
          </a>
        </div>
      </div>
    </div>
  );
}
