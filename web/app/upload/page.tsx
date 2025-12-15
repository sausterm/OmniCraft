'use client';

import { useState, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import { ArrowLeft, Loader2, AlertCircle } from 'lucide-react';
import ImageUploader from '@/components/ImageUploader';
import ConfigPanel from '@/components/ConfigPanel';
import api, { ProcessConfig, Job } from '@/lib/api';

type Stage = 'upload' | 'configure' | 'processing';

export default function UploadPage() {
  const router = useRouter();
  const [stage, setStage] = useState<Stage>('upload');
  const [job, setJob] = useState<Job | null>(null);
  const [config, setConfig] = useState<ProcessConfig>({
    model_size: 'm',
    confidence: 0.25,
    num_colors: 32,
    min_region_size: 100,
  });
  const [isUploading, setIsUploading] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState<string>('Starting...');

  const handleUpload = useCallback(async (file: File) => {
    setIsUploading(true);
    setError(null);

    try {
      const uploadedJob = await api.uploadImage(file);
      setJob(uploadedJob);
      setStage('configure');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed');
    } finally {
      setIsUploading(false);
    }
  }, []);

  const handleProcess = useCallback(async () => {
    if (!job) return;

    setIsProcessing(true);
    setStage('processing');
    setError(null);

    try {
      // Start processing
      await api.processJob(job.job_id, config);

      // Poll for completion
      const completedJob = await api.waitForCompletion(
        job.job_id,
        (status) => {
          if (status.status === 'processing') {
            setProgress('Processing image with AI...');
          }
        }
      );

      if (completedJob.status === 'failed') {
        throw new Error(completedJob.error || 'Processing failed');
      }

      // Redirect to preview
      router.push(`/preview/${job.job_id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Processing failed');
      setStage('configure');
    } finally {
      setIsProcessing(false);
    }
  }, [job, config, router]);

  return (
    <div className="min-h-screen bg-gray-50 py-12">
      <div className="max-w-2xl mx-auto px-4 sm:px-6">
        {/* Header */}
        <div className="mb-8">
          <button
            onClick={() => router.push('/')}
            className="flex items-center gap-2 text-gray-600 hover:text-gray-900 transition-colors mb-4"
          >
            <ArrowLeft className="w-4 h-4" />
            Back to Home
          </button>
          <h1 className="font-display text-3xl font-bold text-gray-900">
            Create Your Painting
          </h1>
          <p className="text-gray-600 mt-2">
            Upload your photo and customize the processing options.
          </p>
        </div>

        {/* Progress Steps */}
        <div className="flex items-center justify-center mb-8">
          {['Upload', 'Configure', 'Process'].map((step, index) => {
            const stepIndex = ['upload', 'configure', 'processing'].indexOf(stage);
            const isActive = index <= stepIndex;
            const isCurrent = index === stepIndex;

            return (
              <div key={step} className="flex items-center">
                <div
                  className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium transition-colors ${
                    isActive
                      ? 'bg-primary-600 text-white'
                      : 'bg-gray-200 text-gray-500'
                  } ${isCurrent ? 'ring-4 ring-primary-100' : ''}`}
                >
                  {index + 1}
                </div>
                <span
                  className={`ml-2 text-sm font-medium ${
                    isActive ? 'text-gray-900' : 'text-gray-500'
                  }`}
                >
                  {step}
                </span>
                {index < 2 && (
                  <div
                    className={`w-12 h-0.5 mx-4 ${
                      index < stepIndex ? 'bg-primary-600' : 'bg-gray-200'
                    }`}
                  />
                )}
              </div>
            );
          })}
        </div>

        {/* Error Display */}
        {error && (
          <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg flex items-start gap-3">
            <AlertCircle className="w-5 h-5 text-red-500 flex-shrink-0 mt-0.5" />
            <div>
              <p className="font-medium text-red-800">Error</p>
              <p className="text-red-700 text-sm">{error}</p>
            </div>
          </div>
        )}

        {/* Stage Content */}
        {stage === 'upload' && (
          <div className="card">
            <h2 className="font-semibold text-gray-900 mb-4">
              Step 1: Upload Your Image
            </h2>
            <ImageUploader onUpload={handleUpload} isUploading={isUploading} />
          </div>
        )}

        {stage === 'configure' && job && (
          <div className="space-y-6">
            <div className="card">
              <h2 className="font-semibold text-gray-900 mb-4">
                Step 2: Configure Processing
              </h2>
              <div className="bg-gray-50 rounded-lg p-4 flex items-center gap-4 mb-4">
                <div className="w-16 h-16 bg-gray-200 rounded-lg flex items-center justify-center">
                  <span className="text-2xl">🖼️</span>
                </div>
                <div>
                  <p className="font-medium text-gray-900">{job.filename}</p>
                  <p className="text-sm text-gray-500">
                    Ready to process • Job ID: {job.job_id.slice(0, 8)}...
                  </p>
                </div>
              </div>
            </div>

            <ConfigPanel
              config={config}
              onChange={setConfig}
              disabled={isProcessing}
            />

            <div className="flex gap-4">
              <button
                onClick={() => {
                  setStage('upload');
                  setJob(null);
                }}
                className="btn-secondary"
                disabled={isProcessing}
              >
                Change Image
              </button>
              <button
                onClick={handleProcess}
                className="btn-primary flex-1"
                disabled={isProcessing}
              >
                Start Processing
              </button>
            </div>
          </div>
        )}

        {stage === 'processing' && (
          <div className="card text-center py-12">
            <Loader2 className="w-12 h-12 text-primary-600 animate-spin mx-auto mb-4" />
            <h2 className="font-semibold text-gray-900 mb-2">
              Processing Your Image
            </h2>
            <p className="text-gray-600 mb-4">{progress}</p>
            <div className="w-full bg-gray-200 rounded-full h-2 max-w-xs mx-auto">
              <div className="bg-primary-600 h-2 rounded-full animate-pulse w-2/3" />
            </div>
            <p className="text-sm text-gray-500 mt-4">
              This may take 1-3 minutes depending on image complexity.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
