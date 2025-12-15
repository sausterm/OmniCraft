'use client';

import { useState, useCallback, useRef } from 'react';
import { Upload, Image as ImageIcon, X, AlertCircle } from 'lucide-react';

interface ImageUploaderProps {
  onUpload: (file: File) => void;
  isUploading?: boolean;
  maxSizeMB?: number;
  acceptedTypes?: string[];
}

export default function ImageUploader({
  onUpload,
  isUploading = false,
  maxSizeMB = 10,
  acceptedTypes = ['image/jpeg', 'image/png', 'image/webp'],
}: ImageUploaderProps) {
  const [isDragOver, setIsDragOver] = useState(false);
  const [preview, setPreview] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const validateFile = useCallback(
    (file: File): string | null => {
      if (!acceptedTypes.includes(file.type)) {
        return `Invalid file type. Please upload: ${acceptedTypes.map((t) => t.split('/')[1]).join(', ')}`;
      }
      if (file.size > maxSizeMB * 1024 * 1024) {
        return `File too large. Maximum size is ${maxSizeMB}MB`;
      }
      return null;
    },
    [acceptedTypes, maxSizeMB]
  );

  const handleFile = useCallback(
    (file: File) => {
      const validationError = validateFile(file);
      if (validationError) {
        setError(validationError);
        return;
      }

      setError(null);
      setSelectedFile(file);

      // Create preview
      const reader = new FileReader();
      reader.onload = (e) => {
        setPreview(e.target?.result as string);
      };
      reader.readAsDataURL(file);
    },
    [validateFile]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragOver(false);

      const file = e.dataTransfer.files[0];
      if (file) {
        handleFile(file);
      }
    },
    [handleFile]
  );

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  }, []);

  const handleInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) {
        handleFile(file);
      }
    },
    [handleFile]
  );

  const handleClear = useCallback(() => {
    setPreview(null);
    setSelectedFile(null);
    setError(null);
    if (inputRef.current) {
      inputRef.current.value = '';
    }
  }, []);

  const handleUploadClick = useCallback(() => {
    if (selectedFile) {
      onUpload(selectedFile);
    }
  }, [selectedFile, onUpload]);

  return (
    <div className="w-full">
      {!preview ? (
        <div
          className={`dropzone ${isDragOver ? 'dragover' : ''}`}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onClick={() => inputRef.current?.click()}
        >
          <input
            ref={inputRef}
            type="file"
            accept={acceptedTypes.join(',')}
            onChange={handleInputChange}
            className="hidden"
          />
          <Upload className="w-12 h-12 text-gray-400 mx-auto mb-4" />
          <p className="text-gray-700 font-medium mb-1">
            Drop your image here, or click to browse
          </p>
          <p className="text-gray-500 text-sm">
            Supports JPG, PNG, WebP up to {maxSizeMB}MB
          </p>
        </div>
      ) : (
        <div className="relative">
          <div className="relative aspect-video bg-gray-100 rounded-xl overflow-hidden">
            <img
              src={preview}
              alt="Preview"
              className="w-full h-full object-contain"
            />
            <button
              onClick={handleClear}
              className="absolute top-2 right-2 p-2 bg-black/50 hover:bg-black/70 rounded-full text-white transition-colors"
              disabled={isUploading}
            >
              <X className="w-4 h-4" />
            </button>
          </div>
          <div className="mt-4 flex items-center justify-between">
            <div className="flex items-center gap-2 text-gray-600">
              <ImageIcon className="w-4 h-4" />
              <span className="text-sm truncate max-w-[200px]">
                {selectedFile?.name}
              </span>
              <span className="text-xs text-gray-400">
                ({(selectedFile?.size || 0 / 1024 / 1024).toFixed(1)}MB)
              </span>
            </div>
            <button
              onClick={handleUploadClick}
              disabled={isUploading}
              className="btn-primary disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isUploading ? (
                <>
                  <svg
                    className="animate-spin -ml-1 mr-2 h-4 w-4 text-white inline"
                    fill="none"
                    viewBox="0 0 24 24"
                  >
                    <circle
                      className="opacity-25"
                      cx="12"
                      cy="12"
                      r="10"
                      stroke="currentColor"
                      strokeWidth="4"
                    />
                    <path
                      className="opacity-75"
                      fill="currentColor"
                      d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                    />
                  </svg>
                  Uploading...
                </>
              ) : (
                'Upload & Continue'
              )}
            </button>
          </div>
        </div>
      )}

      {error && (
        <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg flex items-start gap-2">
          <AlertCircle className="w-5 h-5 text-red-500 flex-shrink-0 mt-0.5" />
          <p className="text-red-700 text-sm">{error}</p>
        </div>
      )}
    </div>
  );
}
