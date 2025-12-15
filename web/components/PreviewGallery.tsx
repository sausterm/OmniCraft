'use client';

import { useState } from 'react';
import { ChevronLeft, ChevronRight, Layers, Eye, Square } from 'lucide-react';

type ViewMode = 'cumulative' | 'context' | 'isolated';

interface PreviewGalleryProps {
  outputs: {
    cumulative: string[];
    context: string[];
    isolated: string[];
  };
  totalSteps: number;
  apiBase?: string;
}

const VIEW_MODES: { value: ViewMode; label: string; icon: typeof Layers; description: string }[] = [
  {
    value: 'cumulative',
    label: 'Cumulative',
    icon: Layers,
    description: 'Progressive build-up of the painting',
  },
  {
    value: 'context',
    label: 'Context',
    icon: Eye,
    description: 'Current step highlighted in full image',
  },
  {
    value: 'isolated',
    label: 'Isolated',
    icon: Square,
    description: 'Current step region on white canvas',
  },
];

export default function PreviewGallery({
  outputs,
  totalSteps,
  apiBase = '',
}: PreviewGalleryProps) {
  const [currentStep, setCurrentStep] = useState(0);
  const [viewMode, setViewMode] = useState<ViewMode>('cumulative');

  const currentImages = outputs[viewMode];
  const maxStep = currentImages.length - 1;

  const goToStep = (step: number) => {
    setCurrentStep(Math.max(0, Math.min(step, maxStep)));
  };

  const currentImageUrl = currentImages[currentStep]
    ? `${apiBase}${currentImages[currentStep]}`
    : null;

  return (
    <div className="space-y-4">
      {/* View Mode Selector */}
      <div className="flex gap-2 justify-center">
        {VIEW_MODES.map((mode) => (
          <button
            key={mode.value}
            onClick={() => setViewMode(mode.value)}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              viewMode === mode.value
                ? 'bg-primary-600 text-white'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            <mode.icon className="w-4 h-4" />
            {mode.label}
          </button>
        ))}
      </div>

      {/* Image Display */}
      <div className="relative aspect-video bg-gray-100 rounded-xl overflow-hidden">
        {currentImageUrl ? (
          <img
            src={currentImageUrl}
            alt={`Step ${currentStep + 1} - ${viewMode} view`}
            className="w-full h-full object-contain"
          />
        ) : (
          <div className="flex items-center justify-center h-full text-gray-500">
            No image available
          </div>
        )}

        {/* Navigation Arrows */}
        <button
          onClick={() => goToStep(currentStep - 1)}
          disabled={currentStep === 0}
          className="absolute left-2 top-1/2 -translate-y-1/2 p-2 bg-black/50 hover:bg-black/70 disabled:bg-black/20 disabled:cursor-not-allowed rounded-full text-white transition-colors"
        >
          <ChevronLeft className="w-6 h-6" />
        </button>
        <button
          onClick={() => goToStep(currentStep + 1)}
          disabled={currentStep === maxStep}
          className="absolute right-2 top-1/2 -translate-y-1/2 p-2 bg-black/50 hover:bg-black/70 disabled:bg-black/20 disabled:cursor-not-allowed rounded-full text-white transition-colors"
        >
          <ChevronRight className="w-6 h-6" />
        </button>

        {/* Step Counter */}
        <div className="absolute bottom-4 left-1/2 -translate-x-1/2 bg-black/70 text-white px-4 py-2 rounded-full text-sm font-medium">
          Step {currentStep + 1} of {totalSteps}
        </div>
      </div>

      {/* Step Slider */}
      <div className="px-4">
        <input
          type="range"
          min="0"
          max={maxStep}
          value={currentStep}
          onChange={(e) => goToStep(parseInt(e.target.value))}
          className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-primary-600"
        />
        <div className="flex justify-between text-xs text-gray-500 mt-1">
          <span>Start</span>
          <span>Finish</span>
        </div>
      </div>

      {/* View Mode Description */}
      <p className="text-center text-sm text-gray-500">
        {VIEW_MODES.find((m) => m.value === viewMode)?.description}
      </p>

      {/* Thumbnail Strip */}
      <div className="flex gap-2 overflow-x-auto pb-2 px-1">
        {currentImages.slice(0, 10).map((_, index) => (
          <button
            key={index}
            onClick={() => goToStep(index)}
            className={`flex-shrink-0 w-16 h-16 rounded-lg border-2 transition-all ${
              currentStep === index
                ? 'border-primary-500 ring-2 ring-primary-200'
                : 'border-gray-200 hover:border-gray-300'
            }`}
          >
            <div className="w-full h-full bg-gray-100 rounded-md flex items-center justify-center text-xs text-gray-500 font-medium">
              {index + 1}
            </div>
          </button>
        ))}
        {currentImages.length > 10 && (
          <div className="flex-shrink-0 w-16 h-16 rounded-lg bg-gray-100 flex items-center justify-center text-xs text-gray-500">
            +{currentImages.length - 10} more
          </div>
        )}
      </div>
    </div>
  );
}
