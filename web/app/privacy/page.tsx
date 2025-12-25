import { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Privacy Policy - Artisan Paint-by-Numbers',
  description: 'Privacy Policy for Artisan Paint-by-Numbers',
};

export default function PrivacyPage() {
  return (
    <div className="min-h-screen bg-gray-50 py-12">
      <div className="max-w-3xl mx-auto px-4 sm:px-6">
        <h1 className="font-display text-3xl font-bold text-gray-900 mb-8">
          Privacy Policy
        </h1>

        <div className="prose prose-gray max-w-none">
          <p className="text-sm text-gray-500 mb-6">Last updated: December 24, 2024</p>

          <section className="mb-8">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">1. Introduction</h2>
            <p className="text-gray-600">
              Artisan Paint-by-Numbers ("we", "our", "us") respects your privacy and is committed
              to protecting your personal data. This Privacy Policy explains how we collect, use,
              and safeguard your information when you use our Service.
            </p>
          </section>

          <section className="mb-8">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">2. Information We Collect</h2>

            <h3 className="text-lg font-medium text-gray-900 mt-4 mb-2">Information You Provide</h3>
            <ul className="list-disc pl-6 text-gray-600 mb-4">
              <li><strong>Images:</strong> Photos you upload for processing</li>
              <li><strong>Email Address:</strong> For order confirmations and delivery</li>
              <li><strong>Payment Information:</strong> Processed securely by Stripe (we do not store card details)</li>
            </ul>

            <h3 className="text-lg font-medium text-gray-900 mt-4 mb-2">Automatically Collected Information</h3>
            <ul className="list-disc pl-6 text-gray-600">
              <li><strong>Usage Data:</strong> Pages visited, features used, time spent</li>
              <li><strong>Device Information:</strong> Browser type, operating system</li>
              <li><strong>IP Address:</strong> For security and fraud prevention</li>
            </ul>
          </section>

          <section className="mb-8">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">3. How We Use Your Information</h2>
            <p className="text-gray-600 mb-4">We use your information to:</p>
            <ul className="list-disc pl-6 text-gray-600">
              <li>Process your uploaded images and generate paint-by-numbers guides</li>
              <li>Process payments and deliver your purchases</li>
              <li>Send order confirmations and download links</li>
              <li>Improve our Service and develop new features</li>
              <li>Prevent fraud and ensure security</li>
              <li>Comply with legal obligations</li>
            </ul>
          </section>

          <section className="mb-8">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">4. Image Storage and Retention</h2>
            <p className="text-gray-600 mb-4">
              <strong>Uploaded Images:</strong> Your uploaded images are temporarily stored to process
              your order. Images are automatically deleted after 30 days.
            </p>
            <p className="text-gray-600 mb-4">
              <strong>Generated Content:</strong> Your paint-by-numbers guides are stored for 30 days
              to allow you to re-download your purchases. After this period, files may be deleted.
            </p>
            <p className="text-gray-600">
              <strong>No Training:</strong> We do not use your images to train AI models without
              explicit consent.
            </p>
          </section>

          <section className="mb-8">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">5. Information Sharing</h2>
            <p className="text-gray-600 mb-4">We do not sell your personal information. We may share data with:</p>
            <ul className="list-disc pl-6 text-gray-600">
              <li><strong>Stripe:</strong> For payment processing</li>
              <li><strong>Cloud Providers:</strong> For hosting and processing (Modal, Vercel)</li>
              <li><strong>Legal Authorities:</strong> When required by law</li>
            </ul>
          </section>

          <section className="mb-8">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">6. Data Security</h2>
            <p className="text-gray-600">
              We implement industry-standard security measures including encryption in transit (HTTPS),
              secure payment processing through Stripe, and limited access to personal data. However,
              no method of transmission over the Internet is 100% secure.
            </p>
          </section>

          <section className="mb-8">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">7. Your Rights</h2>
            <p className="text-gray-600 mb-4">You have the right to:</p>
            <ul className="list-disc pl-6 text-gray-600">
              <li>Access the personal data we hold about you</li>
              <li>Request correction of inaccurate data</li>
              <li>Request deletion of your data</li>
              <li>Opt out of marketing communications</li>
            </ul>
            <p className="text-gray-600 mt-4">
              To exercise these rights, contact us at support@artisan.com.
            </p>
          </section>

          <section className="mb-8">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">8. Cookies</h2>
            <p className="text-gray-600">
              We use essential cookies to maintain your session and remember your preferences.
              We may use analytics cookies to understand how visitors use our site. You can
              control cookies through your browser settings.
            </p>
          </section>

          <section className="mb-8">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">9. Children's Privacy</h2>
            <p className="text-gray-600">
              Our Service is not intended for children under 13. We do not knowingly collect
              personal information from children under 13.
            </p>
          </section>

          <section className="mb-8">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">10. Changes to This Policy</h2>
            <p className="text-gray-600">
              We may update this Privacy Policy from time to time. We will notify you of significant
              changes by posting the new policy on this page with an updated date.
            </p>
          </section>

          <section className="mb-8">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">11. Contact Us</h2>
            <p className="text-gray-600">
              For questions about this Privacy Policy or our data practices, please contact us at
              support@artisan.com.
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
