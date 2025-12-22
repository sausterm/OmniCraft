'use client';

import { useSession, signOut } from 'next-auth/react';
import { User, LogOut, Loader2 } from 'lucide-react';
import Link from 'next/link';

export default function AuthButton() {
  const { data: session, status } = useSession();

  if (status === 'loading') {
    return (
      <div className="flex items-center gap-2 text-gray-500">
        <Loader2 className="w-4 h-4 animate-spin" />
      </div>
    );
  }

  if (session?.user) {
    return (
      <div className="flex items-center gap-3">
        <Link
          href="/dashboard"
          className="flex items-center gap-2 text-gray-700 hover:text-gray-900"
        >
          <User className="w-4 h-4" />
          <span className="hidden sm:inline">{session.user.email}</span>
        </Link>
        <button
          onClick={() => signOut({ callbackUrl: '/' })}
          className="flex items-center gap-1 text-gray-500 hover:text-gray-700 text-sm"
        >
          <LogOut className="w-4 h-4" />
          <span className="hidden sm:inline">Sign out</span>
        </button>
      </div>
    );
  }

  return (
    <Link
      href="/login"
      className="flex items-center gap-2 px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 text-sm font-medium"
    >
      <User className="w-4 h-4" />
      Sign in
    </Link>
  );
}
