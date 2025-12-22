import { Mail, ArrowLeft } from 'lucide-react';
import Link from 'next/link';

export default function CheckEmailPage() {
  return (
    <div className="min-h-screen bg-gray-50 flex items-center justify-center py-12 px-4">
      <div className="max-w-md w-full text-center">
        <div className="bg-white rounded-2xl shadow-lg p-8">
          <div className="w-16 h-16 bg-primary-100 rounded-full flex items-center justify-center mx-auto mb-6">
            <Mail className="w-8 h-8 text-primary-600" />
          </div>

          <h1 className="font-display text-2xl font-bold text-gray-900 mb-4">
            Check your email
          </h1>

          <p className="text-gray-600 mb-6">
            We sent you a magic link to sign in. Click the link in your email to continue.
          </p>

          <div className="bg-amber-50 border border-amber-200 rounded-lg p-4 mb-6">
            <p className="text-amber-800 text-sm">
              <strong>Tip:</strong> Check your spam folder if you don't see the email within a few minutes.
            </p>
          </div>

          <Link
            href="/login"
            className="flex items-center justify-center gap-2 text-gray-600 hover:text-gray-900"
          >
            <ArrowLeft className="w-4 h-4" />
            Back to login
          </Link>
        </div>
      </div>
    </div>
  );
}
