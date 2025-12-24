'use client';

import { useState, useEffect } from 'react';
import { Palette, Sparkles, Loader2, X, ChevronDown, ChevronUp, Wand2, Camera, Check, AlertTriangle } from 'lucide-react';
import type { StyleInfo, StyleTransferRequest } from '@/lib/api';
import api from '@/lib/api';

interface StyleSelectorProps {
  jobId: string;
  onStyleApplied?: (styledImageUrl: string) => void;
  onStyleRemoved?: () => void;
  currentStyleConfig?: {
    style: string;
    custom_prompt?: string;
  } | null;
  disabled?: boolean;
}

// Style reliability ratings
type ReliabilityLevel = 'high' | 'medium' | 'low' | 'none';

interface StyleMetadata {
  reliability: ReliabilityLevel;
  badge?: string;
  badgeColor?: string;
  bestFor?: string;
}

const STYLE_METADATA: Record<string, StyleMetadata> = {
  photo_realistic: {
    reliability: 'none',
    badge: 'Original',
    badgeColor: 'bg-blue-500',
    bestFor: 'Keep your original image'
  },
  oil_painting: {
    reliability: 'high',
    badge: 'Recommended',
    badgeColor: 'bg-green-500',
    bestFor: 'Works great with all images'
  },
  van_gogh: {
    reliability: 'high',
    badge: 'Recommended',
    badgeColor: 'bg-green-500',
    bestFor: 'Best for landscapes & nature'
  },
  sketch: {
    reliability: 'medium',
    bestFor: 'Portraits & simple subjects'
  },
  anime: {
    reliability: 'medium',
    bestFor: 'Characters & illustrations'
  },
  pop_art: {
    reliability: 'medium',
    bestFor: 'Bold subjects & portraits'
  },
  watercolor: {
    reliability: 'low',
    badge: 'Experimental',
    badgeColor: 'bg-amber-500',
    bestFor: 'Simple scenes only'
  },
  britto_style: {
    reliability: 'low',
    badge: 'Experimental',
    badgeColor: 'bg-amber-500',
    bestFor: 'Simple, bold subjects'
  },
  custom: {
    reliability: 'medium',
    bestFor: 'Describe your own style'
  },
};

// Style card images (placeholders - can be replaced with actual thumbnails)
const STYLE_IMAGES: Record<string, string> = {
  photo_realistic: '/styles/photo.jpg',
  pop_art: '/styles/pop_art.jpg',
  britto_style: '/styles/britto.jpg',
  van_gogh: '/styles/van_gogh.jpg',
  picasso_cubist: '/styles/picasso.jpg',
  anime: '/styles/anime.jpg',
  watercolor: '/styles/watercolor.jpg',
  oil_painting: '/styles/oil.jpg',
  sketch: '/styles/sketch.jpg',
  custom: '/styles/custom.jpg',
};

// Fallback gradient backgrounds for styles without images
const STYLE_GRADIENTS: Record<string, string> = {
  photo_realistic: 'from-slate-400 via-slate-500 to-slate-600',
  pop_art: 'from-pink-500 via-red-500 to-yellow-500',
  britto_style: 'from-yellow-400 via-red-500 to-blue-500',
  van_gogh: 'from-blue-600 via-yellow-500 to-orange-500',
  picasso_cubist: 'from-gray-600 via-blue-400 to-brown-500',
  anime: 'from-pink-400 via-purple-500 to-blue-500',
  watercolor: 'from-blue-300 via-purple-300 to-pink-300',
  oil_painting: 'from-amber-700 via-orange-600 to-red-700',
  sketch: 'from-gray-300 via-gray-400 to-gray-500',
  custom: 'from-violet-500 via-purple-500 to-fuchsia-500',
};

// Helper to reorder styles by reliability
function reorderStylesByReliability(styles: StyleInfo[]): StyleInfo[] {
  const reliabilityOrder: Record<ReliabilityLevel, number> = {
    none: 0, // photo_realistic first
    high: 1,
    medium: 2,
    low: 3,
  };

  return [...styles].sort((a, b) => {
    // Custom always goes last
    if (a.id === 'custom') return 1;
    if (b.id === 'custom') return -1;

    const aReliability = STYLE_METADATA[a.id]?.reliability ?? 'medium';
    const bReliability = STYLE_METADATA[b.id]?.reliability ?? 'medium';

    return reliabilityOrder[aReliability] - reliabilityOrder[bReliability];
  });
}

export default function StyleSelector({
  jobId,
  onStyleApplied,
  onStyleRemoved,
  currentStyleConfig,
  disabled = false,
}: StyleSelectorProps) {
  const [styles, setStyles] = useState<StyleInfo[]>([]);
  const [selectedStyle, setSelectedStyle] = useState<string | null>(
    currentStyleConfig?.style || null
  );
  const [customPrompt, setCustomPrompt] = useState(
    currentStyleConfig?.custom_prompt || ''
  );
  const [isApplying, setIsApplying] = useState(false);
  const [isRemoving, setIsRemoving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [guidanceScale, setGuidanceScale] = useState(7.5);
  const [controlStrength, setControlStrength] = useState(1.0);
  const [numSteps, setNumSteps] = useState(30);
  const [styleStatus, setStyleStatus] = useState<string | null>(null);

  // Load available styles on mount
  useEffect(() => {
    const loadStyles = async () => {
      try {
        const loadedStyles = await api.getStyles();
        // Add photo_realistic if not present and reorder by reliability
        const hasPhotoRealistic = loadedStyles.some(s => s.id === 'photo_realistic');
        const orderedStyles = hasPhotoRealistic ? loadedStyles : [
          { id: 'photo_realistic', name: 'Photo-Realistic', description: 'Use your original image as-is' },
          ...loadedStyles
        ];
        setStyles(reorderStylesByReliability(orderedStyles));
      } catch (err) {
        console.error('Failed to load styles:', err);
        // Use default styles as fallback, ordered by reliability
        setStyles([
          // High reliability / special
          { id: 'photo_realistic', name: 'Photo-Realistic', description: 'Use your original image as-is' },
          { id: 'oil_painting', name: 'Oil Painting', description: 'Rich colors, visible brushstrokes' },
          { id: 'van_gogh', name: 'Van Gogh', description: 'Post-impressionist swirling brushstrokes' },
          // Medium reliability
          { id: 'sketch', name: 'Pencil Sketch', description: 'Hand-drawn style with linework' },
          { id: 'anime', name: 'Anime', description: 'Japanese animation style' },
          { id: 'pop_art', name: 'Pop Art', description: 'Bold colors, thick black outlines' },
          // Low reliability (experimental)
          { id: 'watercolor', name: 'Watercolor', description: 'Soft edges, translucent colors' },
          { id: 'britto_style', name: 'Romero Britto', description: 'Geometric pop art with vibrant colors' },
          // Custom always last
          { id: 'custom', name: 'Custom Style', description: 'Describe your own style' },
        ]);
      }
    };
    loadStyles();
  }, []);

  const handleApplyStyle = async () => {
    if (!selectedStyle || disabled) return;

    // Photo-realistic = skip style transfer, use original image
    if (selectedStyle === 'photo_realistic') {
      // Notify parent that we're using original (no style)
      onStyleRemoved?.();
      return;
    }

    // Validate custom prompt if needed
    if (selectedStyle === 'custom' && !customPrompt.trim()) {
      setError('Please enter a custom style description');
      return;
    }

    setIsApplying(true);
    setError(null);
    setStyleStatus('Starting style transfer...');

    try {
      const request: StyleTransferRequest = {
        job_id: jobId,
        style: selectedStyle,
        guidance_scale: guidanceScale,
        control_strength: controlStrength,
        num_steps: numSteps,
      };

      if (selectedStyle === 'custom' && customPrompt.trim()) {
        request.custom_prompt = customPrompt.trim();
      }

      await api.applyStyleTransfer(request);
      setStyleStatus('Processing... This takes 3-5 minutes');

      // Poll for completion
      const result = await api.waitForStyleTransfer(
        jobId,
        (status) => {
          if (status === 'processing') {
            setStyleStatus('Applying artistic style...');
          } else if (status === 'queued') {
            setStyleStatus('Queued, starting soon...');
          }
        },
        600000, // 10 min timeout
        3000 // 3 second poll interval
      );

      if (result.error) {
        setError(result.error);
        setStyleStatus(null);
      } else if (result.styled_image_url) {
        setStyleStatus(null);
        onStyleApplied?.(result.styled_image_url);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Style transfer failed');
      setStyleStatus(null);
    } finally {
      setIsApplying(false);
    }
  };

  const handleRemoveStyle = async () => {
    if (disabled) return;

    setIsRemoving(true);
    setError(null);

    try {
      await api.removeStyleTransfer(jobId);
      setSelectedStyle(null);
      setCustomPrompt('');
      onStyleRemoved?.();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to remove style');
    } finally {
      setIsRemoving(false);
    }
  };

  const hasAppliedStyle = currentStyleConfig?.style != null;

  return (
    <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
      {/* Header */}
      <div className="p-4 border-b border-gray-100 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Palette className="w-5 h-5 text-primary-600" />
          <h3 className="font-semibold text-gray-900">Apply Artistic Style</h3>
          <span className="text-xs text-gray-500 bg-gray-100 px-2 py-0.5 rounded-full">
            Optional
          </span>
        </div>
        {hasAppliedStyle && (
          <button
            onClick={handleRemoveStyle}
            disabled={isRemoving || isApplying}
            className="flex items-center gap-1 text-sm text-gray-500 hover:text-red-600 transition-colors"
          >
            {isRemoving ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <X className="w-4 h-4" />
            )}
            Remove style
          </button>
        )}
      </div>

      {/* Style Grid */}
      <div className="p-4">
        <p className="text-sm text-gray-600 mb-4">
          Transform your image with an artistic style before generating paint-by-numbers.
          This creates a unique, stylized painting experience.
        </p>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-4">
          {styles.map((style) => {
            const metadata = STYLE_METADATA[style.id];
            const hasBadge = metadata?.badge;

            return (
              <button
                key={style.id}
                onClick={() => setSelectedStyle(style.id)}
                disabled={disabled || isApplying}
                className={`relative group rounded-lg overflow-hidden transition-all ${
                  selectedStyle === style.id
                    ? 'ring-2 ring-primary-500 ring-offset-2'
                    : 'hover:ring-2 hover:ring-gray-300 hover:ring-offset-1'
                } ${disabled || isApplying ? 'opacity-50 cursor-not-allowed' : ''}`}
              >
                {/* Background gradient */}
                <div
                  className={`aspect-square bg-gradient-to-br ${
                    STYLE_GRADIENTS[style.id] || 'from-gray-400 to-gray-600'
                  } flex items-center justify-center`}
                >
                  {style.id === 'custom' ? (
                    <Wand2 className="w-8 h-8 text-white/80" />
                  ) : style.id === 'photo_realistic' ? (
                    <Camera className="w-8 h-8 text-white/80" />
                  ) : (
                    <Sparkles className="w-8 h-8 text-white/80" />
                  )}
                </div>

                {/* Label overlay with badge */}
                <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/70 to-transparent p-2">
                  <p className="text-white text-sm font-medium truncate">
                    {style.name}
                  </p>
                  {hasBadge && (
                    <span
                      className={`inline-flex items-center gap-1 text-[10px] font-medium text-white px-1.5 py-0.5 rounded ${metadata.badgeColor} mt-1`}
                    >
                      {metadata.badge === 'Recommended' && (
                        <Check className="w-2.5 h-2.5" />
                      )}
                      {metadata.badge === 'Experimental' && (
                        <AlertTriangle className="w-2.5 h-2.5" />
                      )}
                      {metadata.badge}
                    </span>
                  )}
                </div>

                {/* Selected check */}
                {selectedStyle === style.id && (
                  <div className="absolute top-2 right-2 w-6 h-6 bg-primary-500 rounded-full flex items-center justify-center">
                    <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                    </svg>
                  </div>
                )}
              </button>
            );
          })}
        </div>

        {/* Custom prompt input */}
        {selectedStyle === 'custom' && (
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Describe your style
            </label>
            <textarea
              value={customPrompt}
              onChange={(e) => setCustomPrompt(e.target.value)}
              placeholder="e.g., impressionist style with soft pastel colors and dreamy atmosphere, or art deco geometric patterns with gold accents..."
              rows={3}
              disabled={disabled || isApplying}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-primary-500 focus:border-transparent disabled:bg-gray-100"
            />
          </div>
        )}

        {/* Selected style description */}
        {selectedStyle && selectedStyle !== 'custom' && (
          <div className="mb-4 p-3 bg-gray-50 rounded-lg">
            <p className="text-sm text-gray-600">
              <span className="font-medium text-gray-900">
                {styles.find(s => s.id === selectedStyle)?.name}:
              </span>{' '}
              {styles.find(s => s.id === selectedStyle)?.description}
            </p>
            {STYLE_METADATA[selectedStyle]?.bestFor && (
              <p className="text-xs text-gray-500 mt-1">
                Best for: {STYLE_METADATA[selectedStyle].bestFor}
              </p>
            )}
            {selectedStyle === 'photo_realistic' && (
              <p className="text-xs text-blue-600 mt-1 font-medium">
                This will skip style transfer and use your original image.
              </p>
            )}
          </div>
        )}

        {/* Advanced settings toggle */}
        <button
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="flex items-center gap-1 text-sm text-gray-500 hover:text-gray-700 mb-3"
        >
          {showAdvanced ? (
            <ChevronUp className="w-4 h-4" />
          ) : (
            <ChevronDown className="w-4 h-4" />
          )}
          Advanced settings
        </button>

        {/* Advanced settings */}
        {showAdvanced && (
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-4 p-3 bg-gray-50 rounded-lg">
            <div>
              <label className="block text-xs font-medium text-gray-600 mb-1">
                Style Strength ({guidanceScale})
              </label>
              <input
                type="range"
                min={3}
                max={15}
                step={0.5}
                value={guidanceScale}
                onChange={(e) => setGuidanceScale(parseFloat(e.target.value))}
                disabled={disabled || isApplying}
                className="w-full"
              />
              <div className="flex justify-between text-xs text-gray-400">
                <span>Subtle</span>
                <span>Strong</span>
              </div>
            </div>

            <div>
              <label className="block text-xs font-medium text-gray-600 mb-1">
                Style Intensity (Level {controlStrength <= 0.7 ? '1 - Heavy' : controlStrength <= 0.9 ? '2' : controlStrength <= 1.1 ? '3 - Balanced' : controlStrength <= 1.3 ? '4' : '5 - Subtle'})
              </label>
              <input
                type="range"
                min={0.5}
                max={1.5}
                step={0.25}
                value={controlStrength}
                onChange={(e) => setControlStrength(parseFloat(e.target.value))}
                disabled={disabled || isApplying}
                className="w-full"
              />
              <div className="flex justify-between text-xs text-gray-400">
                <span>More Style</span>
                <span>Keep Original</span>
              </div>
            </div>

            <div>
              <label className="block text-xs font-medium text-gray-600 mb-1">
                Quality ({numSteps} steps)
              </label>
              <input
                type="range"
                min={20}
                max={50}
                step={5}
                value={numSteps}
                onChange={(e) => setNumSteps(parseInt(e.target.value))}
                disabled={disabled || isApplying}
                className="w-full"
              />
              <div className="flex justify-between text-xs text-gray-400">
                <span>Fast</span>
                <span>High</span>
              </div>
            </div>
          </div>
        )}

        {/* Error message */}
        {error && (
          <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg">
            <p className="text-sm text-red-700">{error}</p>
          </div>
        )}

        {/* Status message */}
        {styleStatus && (
          <div className="mb-4 p-3 bg-blue-50 border border-blue-200 rounded-lg flex items-center gap-2">
            <Loader2 className="w-4 h-4 text-blue-600 animate-spin" />
            <p className="text-sm text-blue-700">{styleStatus}</p>
          </div>
        )}

        {/* Apply button */}
        <button
          onClick={handleApplyStyle}
          disabled={!selectedStyle || disabled || isApplying}
          className={`w-full py-3 rounded-lg font-medium transition-all flex items-center justify-center gap-2 ${
            !selectedStyle || disabled || isApplying
              ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
              : 'bg-gradient-to-r from-primary-500 to-primary-600 text-white hover:from-primary-600 hover:to-primary-700'
          }`}
        >
          {isApplying ? (
            <>
              <Loader2 className="w-5 h-5 animate-spin" />
              Applying Style...
            </>
          ) : selectedStyle === 'photo_realistic' ? (
            <>
              <Camera className="w-5 h-5" />
              Use Original Image
            </>
          ) : (
            <>
              <Sparkles className="w-5 h-5" />
              {hasAppliedStyle ? 'Apply New Style' : 'Apply Style'}
            </>
          )}
        </button>

        {/* Processing time note */}
        {selectedStyle && selectedStyle !== 'photo_realistic' && !hasAppliedStyle && (
          <p className="text-xs text-gray-500 text-center mt-2">
            Style transfer typically takes 3-5 minutes
          </p>
        )}
      </div>
    </div>
  );
}
