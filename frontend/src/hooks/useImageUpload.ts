// Copyright (c) 2026 Harsh Dwivedi
// Licensed under the Harsh Non-Commercial Attribution License (HNCAL) v1.0

/**
 * PieceWise — Image Upload State Hook
 * Manages file selection, preview URL lifecycle, and validation
 * for both the reference image and pieces image upload zones.
 */

import { useCallback, useEffect, useState } from 'react'
import type { UploadedImage } from '@/types/solver'

const MAX_SIZE_MB = 50
const ALLOWED_TYPES = ['image/jpeg', 'image/png', 'image/webp']

export interface UseImageUploadReturn {
  image: UploadedImage | null
  error: string | null
  onDrop: (files: File[]) => void
  clear: () => void
}

export function useImageUpload(): UseImageUploadReturn {
  const [image, setImage] = useState<UploadedImage | null>(null)
  const [error, setError] = useState<string | null>(null)

  // Revoke preview URL on unmount or replacement to avoid memory leaks
  useEffect(() => {
    return () => {
      if (image?.previewUrl) URL.revokeObjectURL(image.previewUrl)
    }
  }, [image])

  const onDrop = useCallback((files: File[]) => {
    const file = files[0]
    if (!file) return

    // Client-side validation mirrors backend validator.py
    if (!ALLOWED_TYPES.includes(file.type)) {
      setError(`Unsupported format "${file.type}". Please upload JPEG, PNG, or WebP.`)
      return
    }
    if (file.size > MAX_SIZE_MB * 1024 * 1024) {
      setError(`File exceeds ${MAX_SIZE_MB} MB limit.`)
      return
    }

    // Revoke previous preview
    if (image?.previewUrl) URL.revokeObjectURL(image.previewUrl)

    setError(null)
    setImage({ file, previewUrl: URL.createObjectURL(file) })
  }, [image])

  const clear = useCallback(() => {
    if (image?.previewUrl) URL.revokeObjectURL(image.previewUrl)
    setImage(null)
    setError(null)
  }, [image])

  return { image, error, onDrop, clear }
}