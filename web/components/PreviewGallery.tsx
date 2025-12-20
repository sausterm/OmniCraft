'use client';

import { useState, useEffect, useCallback } from 'react';
import { ChevronLeft, ChevronRight, Layers, Eye, Square, Paintbrush, Palette, Lightbulb, Keyboard } from 'lucide-react';

type ViewMode = 'cumulative' | 'context' | 'isolated';

interface PaintingStep {
  step_number: number;
  region_name: string;
  technique: string;
  brush_type: string;
  stroke_motion: string;
  colors: string[];
  instruction: string;
  tips: string[];
}

interface PreviewGalleryProps {
  outputs: {
    cumulative: string[];
    context: string[];
    isolated: string[];
  };
  totalSteps: number;
  apiBase?: string;
  paintingGuide?: {
    steps: PaintingStep[];
    scene_context?: {
      time_of_day: string;
      weather: string;
      setting: string;
      lighting: string;
      mood: string;
    };
  };
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

// Bob Ross-style encouragements
const BOB_ROSS_QUOTES = [
  "There are no mistakes, only happy little accidents.",
  "We don't make mistakes, just happy little accidents.",
  "Let's get crazy!",
  "Beat the devil out of it!",
  "Happy little trees...",
  "Anyone can paint.",
  "Talent is a pursued interest.",
  "You can do anything you want to do.",
];

export default function PreviewGallery({
  outputs,
  totalSteps,
  apiBase = '',
  paintingGuide,
}: PreviewGalleryProps) {
  const [currentStep, setCurrentStep] = useState(0);
  const [viewMode, setViewMode] = useState<ViewMode>('cumulative');

  // Handle undefined or empty outputs
  const currentImages = outputs?.[viewMode] ?? [];
  const maxStep = Math.max(0, currentImages.length - 1);

  // Get current step instructions
  const currentInstruction = paintingGuide?.steps?.[currentStep];

  // If no images available, show placeholder
  if (!outputs || (!outputs.cumulative?.length && !outputs.context?.length && !outputs.isolated?.length)) {
    return (
      <div className="text-center py-12 text-gray-500">
        <p>No preview images available yet.</p>
        <p className="text-sm mt-2">Processing may still be in progress.</p>
      </div>
    );
  }

  const goToStep = useCallback((step: number) => {
    setCurrentStep(Math.max(0, Math.min(step, maxStep)));
  }, [maxStep]);

  // Keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'ArrowLeft') {
        goToStep(currentStep - 1);
      } else if (e.key === 'ArrowRight') {
        goToStep(currentStep + 1);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [currentStep, goToStep]);

  const currentImageUrl = currentImages[currentStep]
    ? `${apiBase}${currentImages[currentStep]}`
    : null;

  // Get a random Bob Ross quote for this step
  const bobQuote = BOB_ROSS_QUOTES[currentStep % BOB_ROSS_QUOTES.length];

  return (
    <div className="space-y-4">
      {/* Scene Context (if available) */}
      {paintingGuide?.scene_context && currentStep === 0 && (
        <div className="bg-amber-50 border border-amber-200 rounded-xl p-4 mb-4">
          <h3 className="font-semibold text-amber-800 mb-2 flex items-center gap-2">
            <Lightbulb className="w-5 h-5" />
            Scene Analysis
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-2 text-sm">
            <div>
              <span className="text-amber-600">Time:</span>{' '}
              <span className="font-medium">{paintingGuide.scene_context.time_of_day}</span>
            </div>
            <div>
              <span className="text-amber-600">Weather:</span>{' '}
              <span className="font-medium">{paintingGuide.scene_context.weather}</span>
            </div>
            <div>
              <span className="text-amber-600">Setting:</span>{' '}
              <span className="font-medium">{paintingGuide.scene_context.setting}</span>
            </div>
            <div>
              <span className="text-amber-600">Lighting:</span>{' '}
              <span className="font-medium">{paintingGuide.scene_context.lighting}</span>
            </div>
            <div>
              <span className="text-amber-600">Mood:</span>{' '}
              <span className="font-medium">{paintingGuide.scene_context.mood}</span>
            </div>
          </div>
        </div>
      )}

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

      {/* Main Content: Image + Instructions Side by Side on large screens */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Image Display (2/3 width on large screens) */}
        <div className="lg:col-span-2">
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
            <div className="absolute bottom-4 left-1/2 -translate-x-1/2 flex items-center gap-2">
              <div className="bg-black/70 text-white px-4 py-2 rounded-full text-sm font-medium">
                Step {currentStep + 1} of {totalSteps}
              </div>
              <div className="bg-black/50 text-white/70 px-2 py-1 rounded-full text-xs flex items-center gap-1">
                <Keyboard className="w-3 h-3" />
                <span className="hidden sm:inline">Arrow keys</span>
              </div>
            </div>
          </div>
        </div>

        {/* Instructions Panel (1/3 width on large screens) */}
        <div className="bg-gradient-to-br from-green-50 to-blue-50 rounded-xl p-4 border border-green-200">
          {currentInstruction ? (
            <div className="space-y-4">
              {/* Step Title */}
              <div>
                <h3 className="font-bold text-lg text-gray-800">
                  {currentInstruction.region_name}
                </h3>
                <p className="text-sm text-gray-500">{currentInstruction.technique}</p>
              </div>

              {/* Bob Ross Quote */}
              <div className="bg-white/50 rounded-lg p-3 italic text-gray-600 text-sm border-l-4 border-green-400">
                "{bobQuote}"
                <div className="text-xs text-gray-400 mt-1">- Bob Ross</div>
              </div>

              {/* Main Instruction */}
              <div>
                <h4 className="font-semibold text-gray-700 flex items-center gap-2 mb-2">
                  <Paintbrush className="w-4 h-4" />
                  Instructions
                </h4>
                <p className="text-gray-600 text-sm leading-relaxed">
                  {currentInstruction.instruction || "Apply paint to this region using the recommended technique."}
                </p>
              </div>

              {/* Brush & Stroke */}
              <div className="grid grid-cols-2 gap-2 text-sm">
                <div className="bg-white/50 rounded-lg p-2">
                  <span className="text-gray-500">Brush:</span>
                  <div className="font-medium text-gray-700">
                    {currentInstruction.brush_type?.replace(/_/g, ' ') || 'Medium flat'}
                  </div>
                </div>
                <div className="bg-white/50 rounded-lg p-2">
                  <span className="text-gray-500">Stroke:</span>
                  <div className="font-medium text-gray-700">
                    {currentInstruction.stroke_motion?.replace(/_/g, ' ') || 'Smooth'}
                  </div>
                </div>
              </div>

              {/* Colors */}
              {currentInstruction.colors && currentInstruction.colors.length > 0 && (
                <div>
                  <h4 className="font-semibold text-gray-700 flex items-center gap-2 mb-2">
                    <Palette className="w-4 h-4" />
                    Colors
                  </h4>
                  <div className="flex flex-wrap gap-2">
                    {currentInstruction.colors.map((color, i) => {
                      // Try to parse color name to CSS color
                      const colorMap: Record<string, string> = {
                        'phthalo blue': '#000F89',
                        'titanium white': '#FAFAFA',
                        'van dyke brown': '#664228',
                        'alizarin crimson': '#E32636',
                        'sap green': '#507D2A',
                        'cadmium yellow': '#FFF600',
                        'prussian blue': '#003153',
                        'midnight black': '#0C090A',
                        'bright red': '#FF0000',
                        'indian yellow': '#E3A857',
                        'yellow ochre': '#CC7722',
                        'dark sienna': '#3C1414',
                      };
                      const cssColor = colorMap[color.toLowerCase()] || '#9CA3AF';
                      const isLight = ['titanium white', 'cadmium yellow', 'indian yellow', 'yellow ochre'].includes(color.toLowerCase());

                      return (
                        <span
                          key={i}
                          className="px-2 py-1 rounded-full text-xs font-medium flex items-center gap-1.5 border"
                          style={{ backgroundColor: cssColor + '20' }}
                        >
                          <span
                            className="w-3 h-3 rounded-full border"
                            style={{
                              backgroundColor: cssColor,
                              borderColor: isLight ? '#D1D5DB' : cssColor
                            }}
                          />
                          {color}
                        </span>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* Tips */}
              {currentInstruction.tips && currentInstruction.tips.length > 0 && (
                <div>
                  <h4 className="font-semibold text-gray-700 flex items-center gap-2 mb-2">
                    <Lightbulb className="w-4 h-4" />
                    Pro Tips
                  </h4>
                  <ul className="space-y-1">
                    {currentInstruction.tips.map((tip, i) => (
                      <li key={i} className="text-sm text-gray-600 flex items-start gap-2">
                        <span className="text-green-500 mt-1">•</span>
                        {tip}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          ) : (
            <div className="text-center py-8 text-gray-500">
              <Paintbrush className="w-12 h-12 mx-auto mb-3 opacity-30" />
              <p className="font-medium">Step {currentStep + 1}</p>
              <p className="text-sm mt-1">Instructions loading...</p>
              <p className="text-xs mt-4 italic">"{bobQuote}"</p>
            </div>
          )}
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
        {currentImages.slice(0, 12).map((imgPath, index) => {
          const thumbUrl = imgPath ? `${apiBase}${imgPath}` : null;
          return (
            <button
              key={index}
              onClick={() => goToStep(index)}
              className={`flex-shrink-0 w-16 h-16 rounded-lg border-2 transition-all overflow-hidden ${
                currentStep === index
                  ? 'border-primary-500 ring-2 ring-primary-200'
                  : 'border-gray-200 hover:border-gray-300'
              }`}
            >
              {thumbUrl ? (
                <img
                  src={thumbUrl}
                  alt={`Step ${index + 1}`}
                  className="w-full h-full object-cover"
                />
              ) : (
                <div className="w-full h-full bg-gray-100 flex items-center justify-center text-xs text-gray-500 font-medium">
                  {index + 1}
                </div>
              )}
            </button>
          );
        })}
        {currentImages.length > 12 && (
          <div className="flex-shrink-0 w-16 h-16 rounded-lg bg-gray-100 flex items-center justify-center text-xs text-gray-500">
            +{currentImages.length - 12} more
          </div>
        )}
      </div>
    </div>
  );
}
