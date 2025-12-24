'use client';

import { Zap, Star, Palette, Check } from 'lucide-react';
import type { ProcessConfig } from '@/lib/api';

export type PresetType = 'quick' | 'balanced' | 'detailed';

interface PresetSelectorProps {
  selectedPreset: PresetType;
  onPresetChange: (preset: PresetType, config: ProcessConfig) => void;
  disabled?: boolean;
}

interface Preset {
  id: PresetType;
  name: string;
  description: string;
  time: string;
  icon: React.ReactNode;
  recommended?: boolean;
  config: ProcessConfig;
}

const PRESETS: Preset[] = [
  {
    id: 'quick',
    name: 'Quick Preview',
    description: 'Fast processing, good for testing',
    time: '~30 seconds',
    icon: <Zap className="w-5 h-5" />,
    config: {
      model_size: 'n',
      confidence: 0.30,
      num_colors: 16,
      min_region_size: 150,
    },
  },
  {
    id: 'balanced',
    name: 'Balanced',
    description: 'Best quality/speed tradeoff',
    time: '~1-2 minutes',
    icon: <Star className="w-5 h-5" />,
    recommended: true,
    config: {
      model_size: 'm',
      confidence: 0.25,
      num_colors: 32,
      min_region_size: 100,
    },
  },
  {
    id: 'detailed',
    name: 'High Detail',
    description: 'Maximum detail capture',
    time: '~3-5 minutes',
    icon: <Palette className="w-5 h-5" />,
    config: {
      model_size: 'l',
      confidence: 0.20,
      num_colors: 48,
      min_region_size: 50,
    },
  },
];

export function getPresetConfig(preset: PresetType): ProcessConfig {
  return PRESETS.find((p) => p.id === preset)?.config ?? PRESETS[1].config;
}

export default function PresetSelector({
  selectedPreset,
  onPresetChange,
  disabled = false,
}: PresetSelectorProps) {
  return (
    <div className="space-y-3">
      <label className="block text-sm font-medium text-gray-700">
        Processing Quality
      </label>
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        {PRESETS.map((preset) => {
          const isSelected = selectedPreset === preset.id;
          return (
            <button
              key={preset.id}
              onClick={() => onPresetChange(preset.id, preset.config)}
              disabled={disabled}
              className={`relative flex flex-col items-start p-4 rounded-xl border-2 transition-all text-left min-h-[100px] ${
                isSelected
                  ? 'border-primary-500 bg-primary-50 ring-2 ring-primary-200'
                  : 'border-gray-200 bg-white hover:border-gray-300 hover:bg-gray-50'
              } ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
            >
              {/* Recommended badge */}
              {preset.recommended && (
                <span className="absolute -top-2 -right-2 px-2 py-0.5 text-xs font-medium bg-primary-500 text-white rounded-full">
                  Recommended
                </span>
              )}

              {/* Selected checkmark */}
              {isSelected && (
                <div className="absolute top-3 right-3 w-5 h-5 bg-primary-500 rounded-full flex items-center justify-center">
                  <Check className="w-3 h-3 text-white" />
                </div>
              )}

              {/* Icon and name */}
              <div className="flex items-center gap-2 mb-1">
                <span
                  className={`${
                    isSelected ? 'text-primary-600' : 'text-gray-500'
                  }`}
                >
                  {preset.icon}
                </span>
                <span
                  className={`font-semibold ${
                    isSelected ? 'text-primary-900' : 'text-gray-900'
                  }`}
                >
                  {preset.name}
                </span>
              </div>

              {/* Description */}
              <p
                className={`text-sm ${
                  isSelected ? 'text-primary-700' : 'text-gray-500'
                }`}
              >
                {preset.description}
              </p>

              {/* Time estimate */}
              <p
                className={`text-xs mt-auto pt-2 ${
                  isSelected ? 'text-primary-600' : 'text-gray-400'
                }`}
              >
                {preset.time}
              </p>
            </button>
          );
        })}
      </div>
    </div>
  );
}
