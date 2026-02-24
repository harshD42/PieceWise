// Copyright (c) 2026 Harsh Dwivedi
// Licensed under the Harsh Non-Commercial Attribution License (HNCAL) v1.0

import { useCallback, useRef } from 'react'
import type { StepManifestEntry } from '@/types/solver'
import styles from './SolutionViewer.module.css'

interface Props {
  overlayUrl: string
  steps: StepManifestEntry[]
  activeStepIndex: number
  onCellClick: (row: number, col: number) => void
}

export function GridOverlay({ overlayUrl, steps, activeStepIndex, onCellClick }: Props) {
  const imgRef = useRef<HTMLImageElement>(null)
  const activeStep = steps[activeStepIndex]

  const handleClick = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      const img = imgRef.current
      if (!img) return

      // Infer grid dimensions from steps
      const maxRow = Math.max(...steps.map((s) => s.grid_pos[0]))
      const maxCol = Math.max(...steps.map((s) => s.grid_pos[1]))
      const nRows = maxRow + 1
      const nCols = maxCol + 1

      const rect = img.getBoundingClientRect()
      const x = e.clientX - rect.left
      const y = e.clientY - rect.top
      const col = Math.floor((x / rect.width) * nCols)
      const row = Math.floor((y / rect.height) * nRows)

      onCellClick(
        Math.max(0, Math.min(row, nRows - 1)),
        Math.max(0, Math.min(col, nCols - 1)),
      )
    },
    [steps, onCellClick],
  )

  return (
    <div className={styles.gridWrapper} onClick={handleClick} role="button" tabIndex={0}>
      <img
        ref={imgRef}
        src={overlayUrl}
        alt="Reference overlay"
        className={styles.gridImg}
        draggable={false}
      />
      {/* Active step highlight box */}
      {activeStep && (
        <ActiveHighlight
          step={activeStep}
          steps={steps}
          imgRef={imgRef}
        />
      )}
    </div>
  )
}

function ActiveHighlight({
  step,
  steps,
  imgRef,
}: {
  step: StepManifestEntry
  steps: StepManifestEntry[]
  imgRef: React.RefObject<HTMLImageElement>
}) {
  // Compute cell position as % of image dimensions
  const maxRow = Math.max(...steps.map((s) => s.grid_pos[0]))
  const maxCol = Math.max(...steps.map((s) => s.grid_pos[1]))
  const nRows = maxRow + 1
  const nCols = maxCol + 1

  const [row, col] = step.grid_pos
  const top = `${(row / nRows) * 100}%`
  const left = `${(col / nCols) * 100}%`
  const width = `${(1 / nCols) * 100}%`
  const height = `${(1 / nRows) * 100}%`

  return (
    <div
      className={styles.activeCell}
      style={{ top, left, width, height }}
      aria-hidden="true"
    />
  )
}