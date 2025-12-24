'use client';

import { Settings, Info } from 'lucide-react';
import type { ProcessConfig } from '@/lib/api';

interface ConfigPanelProps {
  config: ProcessConfig;
  onChange: (config: ProcessConfig) => void;
  disabled?: boolean;
  /** When true, shows as inline content without card wrapper */
  inline?: boolean;
}

const MODEL_SIZES = [
  { value: 'n', label: 'Nano', description: 'Fastest, basic accuracy' },
  { value: 's', label: 'Small', description: 'Fast, good accuracy' },
  { value: 'm', label: 'Medium', description: 'Balanced speed/accuracy' },
  { value: 'l', label: 'Large', description: 'Slower, high accuracy' },
  { value: 'x', label: 'Extra Large', description: 'Slowest, best accuracy' },
] as const;

export default function ConfigPanel({
  config,
  onChange,
  disabled = false,
  inline = false,
}: ConfigPanelProps) {
  const updateConfig = (updates: Partial<ProcessConfig>) => {
    onChange({ ...config, ...updates });
  };

  const content = (
    <div className="space-y-6">
      {/* Model Size */}
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Detection Model
          <span className="ml-1 text-gray-400 font-normal">
            (affects accuracy & speed)
          </span>
        </label>
        <div className="grid grid-cols-5 gap-2">
          {MODEL_SIZES.map((model) => (
            <button
              key={model.value}
              onClick={() => updateConfig({ model_size: model.value })}
              disabled={disabled}
              className={`p-2 rounded-lg border text-center transition-all ${
                config.model_size === model.value
                  ? 'border-primary-500 bg-primary-50 text-primary-700'
                  : 'border-gray-200 hover:border-gray-300 text-gray-700'
              } disabled:opacity-50 disabled:cursor-not-allowed`}
            >
              <span className="block font-medium text-sm">{model.label}</span>
            </button>
          ))}
        </div>
        <p className="mt-1 text-xs text-gray-500">
          {MODEL_SIZES.find((m) => m.value === config.model_size)?.description}
        </p>
      </div>

      {/* Confidence Threshold */}
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Detection Confidence
          <span className="ml-2 text-primary-600 font-semibold">
            {Math.round((config.confidence || 0.25) * 100)}%
          </span>
        </label>
        <input
          type="range"
          min="10"
          max="90"
          value={(config.confidence || 0.25) * 100}
          onChange={(e) =>
            updateConfig({ confidence: parseInt(e.target.value) / 100 })
          }
          disabled={disabled}
          className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-primary-600 disabled:opacity-50"
        />
        <div className="flex justify-between text-xs text-gray-500 mt-1">
          <span>More objects (lower threshold)</span>
          <span>Fewer, confident (higher threshold)</span>
        </div>
      </div>

      {/* Number of Colors */}
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Color Palette Size
          <span className="ml-2 text-primary-600 font-semibold">
            {config.num_colors || 32} colors
          </span>
        </label>
        <input
          type="range"
          min="8"
          max="64"
          step="4"
          value={config.num_colors || 32}
          onChange={(e) =>
            updateConfig({ num_colors: parseInt(e.target.value) })
          }
          disabled={disabled}
          className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-primary-600 disabled:opacity-50"
        />
        <div className="flex justify-between text-xs text-gray-500 mt-1">
          <span>Simpler (fewer colors)</span>
          <span>More detailed (more colors)</span>
        </div>
      </div>

      {/* Minimum Region Size */}
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Minimum Region Size
          <span className="ml-2 text-primary-600 font-semibold">
            {config.min_region_size || 100} pixels
          </span>
        </label>
        <input
          type="range"
          min="25"
          max="500"
          step="25"
          value={config.min_region_size || 100}
          onChange={(e) =>
            updateConfig({ min_region_size: parseInt(e.target.value) })
          }
          disabled={disabled}
          className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-primary-600 disabled:opacity-50"
        />
        <div className="flex justify-between text-xs text-gray-500 mt-1">
          <span>More detail (smaller regions)</span>
          <span>Simpler (larger regions)</span>
        </div>
      </div>

      {/* Info Box */}
      <div className="bg-blue-50 border border-blue-100 rounded-lg p-4 flex gap-3">
        <Info className="w-5 h-5 text-blue-500 flex-shrink-0 mt-0.5" />
        <div className="text-sm text-blue-700">
          <p className="font-medium mb-1">Fine-tune your results</p>
          <p className="text-blue-600">
            These settings let you control the detail level. Most users won't
            need to change these.
          </p>
        </div>
      </div>
    </div>
  );

  // If inline, return content directly; otherwise wrap in card
  if (inline) {
    return content;
  }

  return (
    <div className="card">
      <div className="flex items-center gap-2 mb-4">
        <Settings className="w-5 h-5 text-gray-500" />
        <span className="font-medium text-gray-900">Advanced Processing Options</span>
      </div>
      {content}
    </div>
  );
}
