// Copyright (c) 2026 Harsh Dwivedi
// Licensed under the Harsh Non-Commercial Attribution License (HNCAL) v1.0

import { useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { useImageUpload } from '@/hooks/useImageUpload'
import { useSolverJob } from '@/hooks/useSolverJob'
import { useSolverStore } from '@/store/solverStore'
import styles from './UploadPanel.module.css'

function DropZone({
  label,
  hint,
  image,
  error,
  onDrop,
  onClear,
}: {
  label: string
  hint: string
  image: { previewUrl: string; file: File } | null
  error: string | null
  onDrop: (files: File[]) => void
  onClear: () => void
}) {
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/jpeg': [], 'image/png': [], 'image/webp': [] },
    maxFiles: 1,
    multiple: false,
  })

  return (
    <div className={styles.zone}>
      <p className={styles.zoneLabel}>{label}</p>
      <div
        {...getRootProps()}
        className={`${styles.dropArea} ${isDragActive ? styles.active : ''} ${error ? styles.hasError : ''}`}
      >
        <input {...getInputProps()} />
        {image ? (
          <div className={styles.preview}>
            <img src={image.previewUrl} alt={label} className={styles.previewImg} />
            <div className={styles.previewMeta}>
              <span>{image.file.name}</span>
              <span>{(image.file.size / 1024 / 1024).toFixed(1)} MB</span>
            </div>
          </div>
        ) : (
          <div className={styles.placeholder}>
            <span className={styles.icon}>🖼️</span>
            <p>{isDragActive ? 'Drop it here…' : hint}</p>
            <p className={styles.subHint}>JPEG · PNG · WebP · max 50 MB</p>
          </div>
        )}
      </div>
      {error && <p className={styles.error}>{error}</p>}
      {image && (
        <button className={styles.clearBtn} onClick={onClear} type="button">
          Remove
        </button>
      )}
    </div>
  )
}

export function UploadPanel() {
  const ref = useImageUpload()
  const pieces = useImageUpload()
  const { solve } = useSolverJob()
  const reset = useSolverStore((s) => s.reset)

  const canSolve = ref.image !== null && pieces.image !== null

  const handleSolve = useCallback(() => {
    if (!ref.image || !pieces.image) return
    solve(ref.image.file, pieces.image.file)
  }, [ref.image, pieces.image, solve])

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <h1 className={styles.title}>🧩 PieceWise</h1>
        <p className={styles.tagline}>Scatter to solution, one piece at a time.</p>
      </div>

      <div className={styles.zones}>
        <DropZone
          label="Reference Image"
          hint="Drop the completed puzzle image here"
          image={ref.image}
          error={ref.error}
          onDrop={ref.onDrop}
          onClear={ref.clear}
        />
        <div className={styles.divider}>→</div>
        <DropZone
          label="Scattered Pieces"
          hint="Drop the photo of your scattered pieces here"
          image={pieces.image}
          error={pieces.error}
          onDrop={pieces.onDrop}
          onClear={pieces.clear}
        />
      </div>

      <button
        className={`${styles.solveBtn} ${canSolve ? styles.solveBtnReady : ''}`}
        onClick={handleSolve}
        disabled={!canSolve}
        type="button"
      >
        {canSolve ? '🔍 Solve Puzzle' : 'Upload both images to continue'}
      </button>
    </div>
  )
}