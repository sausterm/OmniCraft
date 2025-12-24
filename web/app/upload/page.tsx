'use client';

import { useState, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import { ArrowLeft, Loader2, AlertCircle, Check, ChevronDown, ChevronUp } from 'lucide-react';
import ImageUploader from '@/components/ImageUploader';
import ConfigPanel from '@/components/ConfigPanel';
import StyleSelector from '@/components/StyleSelector';
import PresetSelector, { PresetType, getPresetConfig } from '@/components/PresetSelector';
import api, { ProcessConfig, Job } from '@/lib/api';

// Simplified 2-step flow: upload (with config) -> style -> processing
type Stage = 'upload' | 'style' | 'processing';

export default function UploadPage() {
  const router = useRouter();
  const [stage, setStage] = useState<Stage>('upload');
  const [job, setJob] = useState<Job | null>(null);
  const [selectedPreset, setSelectedPreset] = useState<PresetType>('balanced');
  const [config, setConfig] = useState<ProcessConfig>(getPresetConfig('balanced'));
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState<string>('Starting...');
  const [styledImageUrl, setStyledImageUrl] = useState<string | null>(null);
  const [styleApplied, setStyleApplied] = useState(false);

  const handleUpload = useCallback(async (file: File) => {
    setIsUploading(true);
    setError(null);

    try {
      const uploadedJob = await api.uploadImage(file);
      setJob(uploadedJob);
      // Stay on upload stage - preset selector appears below
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed');
    } finally {
      setIsUploading(false);
    }
  }, []);

  const handlePresetChange = useCallback((preset: PresetType, presetConfig: ProcessConfig) => {
    setSelectedPreset(preset);
    setConfig(presetConfig);
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
      setStage('style');
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

        {/* Progress Steps - Simplified 2-step flow */}
        <div className="flex items-center justify-center mb-8">
          {[
            { label: 'Upload & Configure', stage: 'upload' as Stage },
            { label: 'Style & Generate', stage: 'style' as Stage },
          ].map((step, index) => {
            const stages: Stage[] = ['upload', 'style', 'processing'];
            const currentStageIndex = stages.indexOf(stage);
            const stepStageIndex = stages.indexOf(step.stage);
            const isActive = stepStageIndex <= currentStageIndex;
            const isCurrent = step.stage === stage || (stage === 'processing' && step.stage === 'style');

            return (
              <div key={step.label} className="flex items-center flex-shrink-0">
                <div
                  className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium transition-colors ${
                    isActive
                      ? 'bg-primary-600 text-white'
                      : 'bg-gray-200 text-gray-500'
                  } ${isCurrent ? 'ring-4 ring-primary-100' : ''}`}
                >
                  {isActive && stepStageIndex < currentStageIndex ? (
                    <Check className="w-4 h-4" />
                  ) : (
                    index + 1
                  )}
                </div>
                <span
                  className={`ml-2 text-sm font-medium ${
                    isActive ? 'text-gray-900' : 'text-gray-500'
                  }`}
                >
                  {step.label}
                </span>
                {index < 1 && (
                  <div
                    className={`w-12 sm:w-16 h-0.5 mx-3 sm:mx-4 ${
                      stepStageIndex < currentStageIndex ? 'bg-primary-600' : 'bg-gray-200'
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
          <div className="space-y-6">
            {/* Upload Card */}
            <div className="card">
              <h2 className="font-semibold text-gray-900 mb-4">
                {job ? 'Your Image' : 'Step 1: Upload Your Image'}
              </h2>

              {job ? (
                <div className="space-y-4">
                  {/* Image preview */}
                  <div className="relative aspect-video bg-gray-100 rounded-lg overflow-hidden">
                    <img
                      src={api.getOriginalImageUrl(job.job_id)}
                      alt="Uploaded"
                      className="w-full h-full object-contain"
                    />
                  </div>
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="font-medium text-gray-900">{job.filename}</p>
                      <p className="text-sm text-gray-500">Ready to process</p>
                    </div>
                    <button
                      onClick={() => setJob(null)}
                      className="text-sm text-gray-500 hover:text-gray-700"
                    >
                      Change image
                    </button>
                  </div>
                </div>
              ) : (
                <ImageUploader onUpload={handleUpload} isUploading={isUploading} />
              )}
            </div>

            {/* Preset Selector - appears after upload */}
            {job && (
              <>
                <div className="card">
                  <PresetSelector
                    selectedPreset={selectedPreset}
                    onPresetChange={handlePresetChange}
                    disabled={isProcessing}
                  />

                  {/* Advanced Settings Toggle */}
                  <button
                    onClick={() => setShowAdvanced(!showAdvanced)}
                    className="flex items-center gap-1 text-sm text-gray-500 hover:text-gray-700 mt-4"
                  >
                    {showAdvanced ? (
                      <ChevronUp className="w-4 h-4" />
                    ) : (
                      <ChevronDown className="w-4 h-4" />
                    )}
                    Advanced settings
                  </button>

                  {showAdvanced && (
                    <div className="mt-4 pt-4 border-t border-gray-100">
                      <ConfigPanel
                        config={config}
                        onChange={setConfig}
                        disabled={isProcessing}
                        inline
                      />
                    </div>
                  )}
                </div>

                {/* Continue Button - Mobile sticky */}
                <div className="sticky-cta-wrapper">
                  <button
                    onClick={() => setStage('style')}
                    className="btn-primary w-full"
                    disabled={isProcessing}
                  >
                    Continue to Style Selection
                  </button>
                </div>
              </>
            )}
          </div>
        )}

        {stage === 'style' && job && (
          <div className="space-y-6">
            {/* Image Preview */}
            <div className="card">
              <h2 className="font-semibold text-gray-900 mb-4">
                Step 2: Choose Your Style
              </h2>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div>
                  <p className="text-sm text-gray-500 mb-2">Original Image</p>
                  <div className="aspect-square bg-gray-100 rounded-lg overflow-hidden">
                    <img
                      src={api.getOriginalImageUrl(job.job_id)}
                      alt="Original"
                      className="w-full h-full object-cover"
                    />
                  </div>
                </div>
                {styledImageUrl && (
                  <div>
                    <p className="text-sm text-gray-500 mb-2 flex items-center gap-1">
                      Styled Preview
                      <Check className="w-4 h-4 text-green-500" />
                    </p>
                    <div className="aspect-square bg-gray-100 rounded-lg overflow-hidden">
                      <img
                        src={styledImageUrl}
                        alt="Styled"
                        className="w-full h-full object-cover"
                      />
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Style Selector */}
            <StyleSelector
              jobId={job.job_id}
              onStyleApplied={(url) => {
                setStyledImageUrl(url);
                setStyleApplied(true);
              }}
              onStyleRemoved={() => {
                setStyledImageUrl(null);
                setStyleApplied(false);
              }}
              currentStyleConfig={styleApplied ? { style: 'applied' } : null}
            />

            {/* Navigation Buttons - Mobile sticky */}
            <div className="sticky-cta-wrapper">
              <div className="flex gap-3">
                <button
                  onClick={() => setStage('upload')}
                  className="btn-secondary"
                >
                  Back
                </button>
                <button
                  onClick={handleProcess}
                  className="btn-primary flex-1"
                  disabled={isProcessing}
                >
                  {styleApplied ? 'Generate Painting Guide' : 'Generate without Style'}
                </button>
              </div>
            </div>
          </div>
        )}

        {stage === 'processing' && (
          <div className="card text-center py-12">
            <Loader2 className="w-12 h-12 text-primary-600 animate-spin mx-auto mb-4" />
            <h2 className="font-semibold text-gray-900 mb-2">
              {styleApplied ? 'Processing Your Styled Image' : 'Processing Your Image'}
            </h2>
            <p className="text-gray-600 mb-4">{progress}</p>
            <div className="w-full bg-gray-200 rounded-full h-2 max-w-xs mx-auto">
              <div className="bg-primary-600 h-2 rounded-full animate-pulse w-2/3" />
            </div>
            <p className="text-sm text-gray-500 mt-4">
              This may take 1-3 minutes depending on image complexity.
            </p>
            {styleApplied && (
              <p className="text-xs text-primary-600 mt-2">
                Using styled image for paint-by-numbers generation
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
// Build: 1766597836
