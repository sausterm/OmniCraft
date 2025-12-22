'use client';

import { useSession } from 'next-auth/react';
import { useRouter } from 'next/navigation';
import { useEffect, useState } from 'react';
import { Loader2, Image, Calendar, Package, ArrowRight, Plus } from 'lucide-react';
import Link from 'next/link';

interface UserJob {
  id: string;
  filename: string;
  status: string;
  createdAt: string;
  products: string[];
}

export default function DashboardPage() {
  const { data: session, status } = useSession();
  const router = useRouter();
  const [jobs, setJobs] = useState<UserJob[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    if (status === 'unauthenticated') {
      router.push('/login');
    }
  }, [status, router]);

  useEffect(() => {
    // For now, we'll show a placeholder - in production this would fetch from API
    setIsLoading(false);
    // Placeholder data
    setJobs([]);
  }, [session]);

  if (status === 'loading' || isLoading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="w-12 h-12 text-primary-600 animate-spin mx-auto mb-4" />
          <p className="text-gray-600">Loading your dashboard...</p>
        </div>
      </div>
    );
  }

  if (!session) {
    return null;
  }

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-4xl mx-auto px-4 sm:px-6">
        {/* Header */}
        <div className="mb-8">
          <h1 className="font-display text-3xl font-bold text-gray-900">
            Welcome back!
          </h1>
          <p className="text-gray-600 mt-2">
            {session.user?.email}
          </p>
        </div>

        {/* Quick Actions */}
        <div className="grid sm:grid-cols-2 gap-4 mb-8">
          <Link
            href="/upload"
            className="bg-gradient-to-br from-primary-500 to-primary-600 rounded-xl p-6 text-white hover:from-primary-600 hover:to-primary-700 transition-all group"
          >
            <div className="flex items-center gap-3 mb-2">
              <Plus className="w-6 h-6" />
              <span className="font-semibold text-lg">Create New</span>
            </div>
            <p className="text-primary-100 text-sm">
              Upload a new image and generate a paint-by-numbers kit
            </p>
            <ArrowRight className="w-5 h-5 mt-4 group-hover:translate-x-1 transition-transform" />
          </Link>

          <div className="bg-white rounded-xl p-6 border border-gray-200">
            <div className="flex items-center gap-3 mb-2">
              <Package className="w-6 h-6 text-gray-600" />
              <span className="font-semibold text-lg text-gray-900">Your Stats</span>
            </div>
            <div className="grid grid-cols-2 gap-4 mt-4">
              <div>
                <p className="text-3xl font-bold text-primary-600">{jobs.length}</p>
                <p className="text-sm text-gray-500">Projects</p>
              </div>
              <div>
                <p className="text-3xl font-bold text-accent-600">
                  {jobs.filter(j => j.products.length > 0).length}
                </p>
                <p className="text-sm text-gray-500">Purchased</p>
              </div>
            </div>
          </div>
        </div>

        {/* Projects List */}
        <div className="bg-white rounded-xl border border-gray-200">
          <div className="p-4 border-b border-gray-200">
            <h2 className="font-semibold text-gray-900">Your Projects</h2>
          </div>

          {jobs.length === 0 ? (
            <div className="p-12 text-center">
              <Image className="w-12 h-12 text-gray-300 mx-auto mb-4" />
              <h3 className="font-medium text-gray-900 mb-2">No projects yet</h3>
              <p className="text-gray-500 mb-6">
                Create your first paint-by-numbers project to get started
              </p>
              <Link
                href="/upload"
                className="inline-flex items-center gap-2 px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700"
              >
                <Plus className="w-4 h-4" />
                Create Project
              </Link>
            </div>
          ) : (
            <div className="divide-y divide-gray-200">
              {jobs.map((job) => (
                <Link
                  key={job.id}
                  href={`/preview/${job.id}`}
                  className="flex items-center gap-4 p-4 hover:bg-gray-50 transition-colors"
                >
                  <div className="w-16 h-16 bg-gray-100 rounded-lg flex items-center justify-center">
                    <Image className="w-6 h-6 text-gray-400" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="font-medium text-gray-900 truncate">
                      {job.filename}
                    </p>
                    <div className="flex items-center gap-4 text-sm text-gray-500 mt-1">
                      <span className="flex items-center gap-1">
                        <Calendar className="w-4 h-4" />
                        {new Date(job.createdAt).toLocaleDateString()}
                      </span>
                      <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${
                        job.status === 'completed'
                          ? 'bg-green-100 text-green-700'
                          : job.status === 'processing'
                          ? 'bg-blue-100 text-blue-700'
                          : 'bg-gray-100 text-gray-700'
                      }`}>
                        {job.status}
                      </span>
                    </div>
                  </div>
                  <ArrowRight className="w-5 h-5 text-gray-400" />
                </Link>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
